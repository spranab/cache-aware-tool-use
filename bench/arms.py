from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI

from bench.common import (
    API_KEY,
    BASE_URL,
    BROKER_MODEL,
    PROVIDER,
    REASONER_MODEL,
    ArmResult,
    TestCase,
    ToolDef,
)
from bench.score import score_arguments

_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

_FORCE_TOOL_SYS = (
    "You have access to tools. You MUST call exactly one tool that satisfies the user's request. "
    "Do not respond with prose; emit a single tool call with correct arguments."
)


def _extract_tool_call(resp) -> tuple[str | None, dict[str, Any] | None]:
    if not resp.choices:
        return None, None
    msg = resp.choices[0].message
    tcs = getattr(msg, "tool_calls", None)
    if not tcs:
        return None, None
    tc = tcs[0]
    name = tc.function.name
    raw = tc.function.arguments or "{}"
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        args = {"_raw": raw}
    return name, args


def _usage(resp) -> tuple[int, int, dict]:
    u = getattr(resp, "usage", None)
    if u is None:
        return 0, 0, {}
    pt = int(getattr(u, "prompt_tokens", 0) or 0)
    ct = int(getattr(u, "completion_tokens", 0) or 0)
    cache: dict = {}
    # DeepSeek native fields
    hit = getattr(u, "prompt_cache_hit_tokens", None)
    miss = getattr(u, "prompt_cache_miss_tokens", None)
    if hit is not None or miss is not None:
        cache["cache_hit"] = int(hit or 0)
        cache["cache_miss"] = int(miss or 0)
    else:
        # OpenAI-style nested details
        details = getattr(u, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) if details else None
        if cached is not None:
            cache["cache_hit"] = int(cached)
            cache["cache_miss"] = max(0, pt - int(cached))
    return pt, ct, cache


def _make_result(
    case: TestCase,
    arm: str,
    tool_name: str | None,
    tool_args: dict[str, Any] | None,
    prompt_tok: int,
    completion_tok: int,
    latency_s: float,
    error: str | None = None,
    extra: dict | None = None,
) -> ArmResult:
    correct = tool_name == case.expected_tool
    acc = score_arguments(case.expected_args, tool_args) if correct else 0.0
    return ArmResult(
        case_id=case.case_id,
        arm=arm,
        selected_tool=tool_name,
        selected_args=tool_args,
        tool_selection_correct=correct,
        arg_accuracy=acc,
        prompt_tokens=prompt_tok,
        completion_tokens=completion_tok,
        latency_s=latency_s,
        error=error,
        extra=extra or {},
    )


# ---------------- Arm A: all tools inlined ----------------
def run_arm_a(case: TestCase) -> ArmResult:
    t0 = time.time()
    try:
        resp = _client.chat.completions.create(
            model=REASONER_MODEL,
            messages=[
                {"role": "system", "content": _FORCE_TOOL_SYS},
                {"role": "user", "content": case.user_goal},
            ],
            tools=[t.to_openai() for t in case.tools],
            tool_choice="auto",
        )
        latency = time.time() - t0
        name, args = _extract_tool_call(resp)
        pt, ct, cache = _usage(resp)
        return _make_result(case, "A", name, args, pt, ct, latency, extra=cache)
    except Exception as e:
        return _make_result(case, "A", None, None, 0, 0, time.time() - t0, error=str(e))


# ---------------- Arm B: top-k lexical retrieval ----------------
def _tokenize(s: str) -> set[str]:
    return {w.strip(".,?!:;'\"()").lower() for w in s.split() if len(w) > 2}


def _retrieve_topk(goal: str, tools: list[ToolDef], k: int = 3) -> list[ToolDef]:
    goal_toks = _tokenize(goal)
    scored = []
    for t in tools:
        tool_toks = _tokenize(t.name.replace("_", " ") + " " + t.description)
        scored.append((len(goal_toks & tool_toks), t))
    scored.sort(key=lambda x: -x[0])
    return [t for _, t in scored[:k]]


def run_arm_b(case: TestCase, k: int = 3) -> ArmResult:
    retrieved = _retrieve_topk(case.user_goal, case.tools, k=k)
    t0 = time.time()
    try:
        resp = _client.chat.completions.create(
            model=REASONER_MODEL,
            messages=[
                {"role": "system", "content": _FORCE_TOOL_SYS},
                {"role": "user", "content": case.user_goal},
            ],
            tools=[t.to_openai() for t in retrieved],
            tool_choice="auto",
        )
        latency = time.time() - t0
        name, args = _extract_tool_call(resp)
        pt, ct, cache = _usage(resp)
        return _make_result(
            case, "B", name, args, pt, ct, latency,
            extra={"retrieved": [t.name for t in retrieved], "k": k, **cache},
        )
    except Exception as e:
        return _make_result(case, "B", None, None, 0, 0, time.time() - t0, error=str(e))


# ---------------- Arm D: goal-delegated broker ----------------
DELEGATE_TOOL = {
    "type": "function",
    "function": {
        "name": "delegate",
        "description": (
            "Request an external action via the capability broker. Pass your goal in "
            "natural language. The broker will choose the right tool, construct arguments, "
            "execute, and return a structured result. You do not need to know which tools exist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "What you want done, in natural language.",
                },
            },
            "required": ["goal"],
        },
    },
}

_DELEGATE_SYS = (
    "You have a single tool, `delegate`, that performs any external action. "
    "Call `delegate` exactly once with a clear natural-language statement of the user's goal. "
    "Do not respond with prose."
)

_BROKER_SYS = (
    "You are a capability broker. Given a user goal, choose the single most appropriate tool "
    "from the available set and call it with correct arguments. Do not ask clarifying questions. "
    "Emit exactly one tool call."
)


def _run_broker(goal: str, tools: list[ToolDef]) -> tuple[str | None, dict[str, Any] | None, int, int, dict]:
    resp = _client.chat.completions.create(
        model=BROKER_MODEL,
        messages=[
            {"role": "system", "content": _BROKER_SYS},
            {"role": "user", "content": goal},
        ],
        tools=[t.to_openai() for t in tools],
        tool_choice="auto",
    )
    name, args = _extract_tool_call(resp)
    pt, ct, cache = _usage(resp)
    return name, args, pt, ct, cache


def run_arm_d(case: TestCase) -> ArmResult:
    t0 = time.time()
    try:
        reasoner = _client.chat.completions.create(
            model=REASONER_MODEL,
            messages=[
                {"role": "system", "content": _DELEGATE_SYS},
                {"role": "user", "content": case.user_goal},
            ],
            tools=[DELEGATE_TOOL],
            tool_choice="auto",
        )
        r_name, r_args = _extract_tool_call(reasoner)
        r_pt, r_ct, r_cache = _usage(reasoner)
        delegate_goal = (r_args or {}).get("goal") if r_name == "delegate" else None

        if not delegate_goal:
            return _make_result(
                case, "D", None, None, r_pt, r_ct, time.time() - t0,
                error=f"reasoner did not call delegate (got {r_name!r})",
            )

        b_name, b_args, b_pt, b_ct, b_cache = _run_broker(delegate_goal, case.tools)
        latency = time.time() - t0
        # Combined cache figures (reasoner + broker)
        combined_hit = (r_cache.get("cache_hit", 0) or 0) + (b_cache.get("cache_hit", 0) or 0)
        combined_miss = (r_cache.get("cache_miss", 0) or 0) + (b_cache.get("cache_miss", 0) or 0)
        extra = {
            "delegate_goal": delegate_goal,
            "reasoner_input_tokens": r_pt,
            "reasoner_output_tokens": r_ct,
            "broker_input_tokens": b_pt,
            "broker_output_tokens": b_ct,
            "reasoner_cache_hit": r_cache.get("cache_hit", 0),
            "broker_cache_hit": b_cache.get("cache_hit", 0),
        }
        if combined_hit or combined_miss:
            extra["cache_hit"] = combined_hit
            extra["cache_miss"] = combined_miss
        return _make_result(
            case, "D", b_name, b_args,
            r_pt + b_pt, r_ct + b_ct, latency,
            extra=extra,
        )
    except Exception as e:
        return _make_result(case, "D", None, None, 0, 0, time.time() - t0, error=str(e))


ARMS = {"A": run_arm_a, "B": run_arm_b, "D": run_arm_d}


def provider_info() -> str:
    return f"provider={PROVIDER} reasoner={REASONER_MODEL} broker={BROKER_MODEL} base_url={BASE_URL}"
