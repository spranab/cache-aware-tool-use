"""Multi-tenant cache-scaling validation.

Simulates K concurrent users, each making N sequential tool calls, with
per-user personalized system prompts. Measures cache hit rates and effective
cost for Arm A (direct schema injection) vs Arm D (goal-delegation broker).

Hypothesis: A's cache footprint is O(K) and hit rate decays with K; D's broker
cache footprint is O(1) — stable across all users — so its hit rate stays near 100%.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from openai import AsyncOpenAI
from tabulate import tabulate

from bench.arms import (
    DELEGATE_TOOL,
    _BROKER_SYS,
    _DELEGATE_SYS,
    _FORCE_TOOL_SYS,
)
from bench.common import (
    API_KEY,
    BASE_URL,
    BROKER_MODEL,
    REASONER_MODEL,
    RUN_DIR,
    ToolDef,
    write_jsonl,
)
from bench.inflate import inflate_tools
from bench.personas import persona_text
from bench.seed_cases import SEED_CASES

# DeepSeek pricing (per million tokens)
PRICE_INPUT_MISS = 0.27
PRICE_INPUT_HIT = 0.027
PRICE_OUTPUT = 1.10


def _usage_stats(resp) -> dict:
    u = resp.usage
    pt = int(getattr(u, "prompt_tokens", 0) or 0)
    ct = int(getattr(u, "completion_tokens", 0) or 0)
    hit = int(getattr(u, "prompt_cache_hit_tokens", 0) or 0)
    miss = int(getattr(u, "prompt_cache_miss_tokens", 0) or 0)
    if hit == 0 and miss == 0:
        details = getattr(u, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details else None
        if cached is not None:
            hit = int(cached)
            miss = max(0, pt - hit)
    return {"prompt_tokens": pt, "completion_tokens": ct, "cache_hit": hit, "cache_miss": miss}


def _extract_delegate_goal(resp) -> str | None:
    msg = resp.choices[0].message
    if not getattr(msg, "tool_calls", None):
        return None
    tc = msg.tool_calls[0]
    if tc.function.name != "delegate":
        return None
    try:
        return json.loads(tc.function.arguments or "{}").get("goal")
    except json.JSONDecodeError:
        return None


async def _call_a(client, sem, user_id: int, case_idx: int, tools: list[ToolDef]) -> dict:
    case = SEED_CASES[case_idx % len(SEED_CASES)]
    system = persona_text(user_id) + "\n\n" + _FORCE_TOOL_SYS
    async with sem:
        t0 = time.time()
        resp = await client.chat.completions.create(
            model=REASONER_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": case.user_goal},
            ],
            tools=[t.to_openai() for t in tools],
            tool_choice="auto",
        )
        latency = time.time() - t0
    s = _usage_stats(resp)
    s.update(arm="A", user_id=user_id, case_idx=case_idx, latency=latency)
    return s


async def _call_d(client, sem, user_id: int, case_idx: int, tools: list[ToolDef]) -> dict:
    case = SEED_CASES[case_idx % len(SEED_CASES)]
    reasoner_system = persona_text(user_id) + "\n\n" + _DELEGATE_SYS
    async with sem:
        t0 = time.time()
        r_resp = await client.chat.completions.create(
            model=REASONER_MODEL,
            messages=[
                {"role": "system", "content": reasoner_system},
                {"role": "user", "content": case.user_goal},
            ],
            tools=[DELEGATE_TOOL],
            tool_choice="auto",
        )
    r_stats = _usage_stats(r_resp)
    goal = _extract_delegate_goal(r_resp) or case.user_goal

    async with sem:
        b_resp = await client.chat.completions.create(
            model=BROKER_MODEL,
            messages=[
                {"role": "system", "content": _BROKER_SYS},  # stable, no personalization
                {"role": "user", "content": goal},
            ],
            tools=[t.to_openai() for t in tools],
            tool_choice="auto",
        )
        latency = time.time() - t0
    b_stats = _usage_stats(b_resp)

    return {
        "arm": "D",
        "user_id": user_id,
        "case_idx": case_idx,
        "prompt_tokens": r_stats["prompt_tokens"] + b_stats["prompt_tokens"],
        "completion_tokens": r_stats["completion_tokens"] + b_stats["completion_tokens"],
        "cache_hit": r_stats["cache_hit"] + b_stats["cache_hit"],
        "cache_miss": r_stats["cache_miss"] + b_stats["cache_miss"],
        "reasoner_pt": r_stats["prompt_tokens"],
        "reasoner_hit": r_stats["cache_hit"],
        "reasoner_miss": r_stats["cache_miss"],
        "broker_pt": b_stats["prompt_tokens"],
        "broker_hit": b_stats["cache_hit"],
        "broker_miss": b_stats["cache_miss"],
        "latency": latency,
    }


async def _user_session(client, sem, user_id: int, calls: int, tools: list[ToolDef], arm: str) -> list[dict]:
    fn = _call_a if arm == "A" else _call_d
    results = []
    for call_idx in range(calls):
        r = await fn(client, sem, user_id, call_idx, tools)
        r["call_idx"] = call_idx
        results.append(r)
    return results


async def run_scenario(n_users: int, calls_per_user: int, n_tools: int,
                       concurrency: int, seed: int) -> dict:
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    sem = asyncio.Semaphore(concurrency)
    base_tools = inflate_tools(SEED_CASES[0].tools, n_tools, seed=seed)

    print(f"\n--- K={n_users} users, {calls_per_user} calls/user, N={n_tools} tools, concurrency={concurrency} ---")

    out: dict = {"n_users": n_users, "calls_per_user": calls_per_user, "n_tools": n_tools}

    for arm in ("A", "D"):
        t0 = time.time()
        tasks = [_user_session(client, sem, uid, calls_per_user, base_tools, arm)
                 for uid in range(n_users)]
        per_user = await asyncio.gather(*tasks)
        wall = time.time() - t0
        flat = [r for batch in per_user for r in batch]

        total_hit = sum(r["cache_hit"] for r in flat)
        total_miss = sum(r["cache_miss"] for r in flat)
        total_in = sum(r["prompt_tokens"] for r in flat)
        total_out = sum(r["completion_tokens"] for r in flat)
        n_calls = len(flat)

        cost_input = (total_miss * PRICE_INPUT_MISS + total_hit * PRICE_INPUT_HIT) / 1_000_000
        cost_output = total_out * PRICE_OUTPUT / 1_000_000
        cost_total = cost_input + cost_output

        hit_rate = total_hit / (total_hit + total_miss) if (total_hit + total_miss) else 0.0

        out[arm] = {
            "n_calls": n_calls,
            "total_in": total_in,
            "total_out": total_out,
            "cache_hit": total_hit,
            "cache_miss": total_miss,
            "hit_rate": hit_rate,
            "cost_input": cost_input,
            "cost_output": cost_output,
            "cost_total": cost_total,
            "cost_per_user": cost_total / n_users,
            "wall_seconds": wall,
            "results": flat,
        }
        print(f"  arm={arm}: {n_calls} calls in {wall:.1f}s, "
              f"in={total_in} (hit {hit_rate:.1%}), "
              f"cost=${cost_total:.4f} (${cost_total / n_users * 1000:.3f}/1k-user)")
    return out


def _print_summary(scenarios: list[dict]) -> None:
    rows = []
    for s in scenarios:
        for arm in ("A", "D"):
            a = s[arm]
            rows.append([
                s["n_users"], arm, a["n_calls"],
                f"{a['hit_rate']:.1%}",
                a["total_in"], a["total_out"],
                f"${a['cost_input']:.4f}",
                f"${a['cost_output']:.4f}",
                f"${a['cost_total']:.4f}",
                f"${a['cost_per_user']*1000:.3f}",
                f"{a['wall_seconds']:.1f}s",
            ])
    print()
    print(tabulate(rows, headers=[
        "K-users", "arm", "calls", "hit-rate",
        "in-tok", "out-tok",
        "$-input", "$-output", "$-total",
        "$/1k-user", "wall",
    ]))

    # Cross-arm comparison
    print("\nA-vs-D cost ratio per K:")
    cross = []
    for s in scenarios:
        ratio_in = s["A"]["cost_input"] / s["D"]["cost_input"] if s["D"]["cost_input"] else float("inf")
        ratio_total = s["A"]["cost_total"] / s["D"]["cost_total"] if s["D"]["cost_total"] else float("inf")
        cross.append([
            s["n_users"],
            f"{s['A']['hit_rate']:.1%}", f"{s['D']['hit_rate']:.1%}",
            f"${s['A']['cost_per_user']*1000:.3f}",
            f"${s['D']['cost_per_user']*1000:.3f}",
            f"{ratio_in:.2f}x",
            f"{ratio_total:.2f}x",
        ])
    print(tabulate(cross, headers=[
        "K-users", "A hit-rate", "D hit-rate",
        "A $/1k-user", "D $/1k-user",
        "A/D input cost", "A/D total cost",
    ]))


async def main_async():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", default="10,50,200",
                    help="Comma-separated K values to test")
    ap.add_argument("--calls", type=int, default=3, help="Calls per user")
    ap.add_argument("--tools", type=int, default=200, help="Total tool count")
    ap.add_argument("--concurrency", type=int, default=8,
                    help="Max concurrent API requests")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ks = [int(x) for x in args.users.split(",") if x.strip()]
    scenarios = []
    for k in ks:
        s = await run_scenario(k, args.calls, args.tools, args.concurrency, args.seed)
        scenarios.append(s)

    _print_summary(scenarios)

    ts = time.strftime("%Y%m%dT%H%M%S")
    out_path = RUN_DIR / f"multitenant_{ts}.jsonl"
    flat_records = []
    for s in scenarios:
        for arm in ("A", "D"):
            for r in s[arm]["results"]:
                r2 = {**r, "n_users": s["n_users"], "n_tools": s["n_tools"]}
                flat_records.append(r2)
    write_jsonl(out_path, flat_records)
    print(f"\nWrote {out_path}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
