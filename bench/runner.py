from __future__ import annotations

import argparse
import time
from dataclasses import replace

from tabulate import tabulate

from bench.arms import ARMS, provider_info
from bench.common import RUN_DIR, TestCase, write_jsonl
from bench.inflate import inflate_tools
from bench.seed_cases import SEED_CASES


def _inflate_case(case: TestCase, target_n: int, seed: int) -> TestCase:
    return replace(case, tools=inflate_tools(case.tools, target_n, seed=seed))


def _aggregate(results, arm_keys, label_extras: dict | None = None):
    rows = []
    for arm in arm_keys:
        arm_rs = [r for r in results if r.arm == arm]
        n = len(arm_rs)
        if n == 0:
            continue
        sel_acc = sum(1 for r in arm_rs if r.tool_selection_correct) / n
        arg_acc = sum(r.arg_accuracy for r in arm_rs) / n
        avg_in = sum(r.prompt_tokens for r in arm_rs) / n
        avg_out = sum(r.completion_tokens for r in arm_rs) / n
        avg_lat = sum(r.latency_s for r in arm_rs) / n
        hits = sum(r.extra.get("cache_hit", 0) or 0 for r in arm_rs)
        in_total = sum(r.prompt_tokens for r in arm_rs)
        hit_pct = (hits / in_total) if in_total else 0.0
        errs = sum(1 for r in arm_rs if r.error)
        row = []
        if label_extras:
            row.extend(label_extras.values())
        row += [
            arm, n,
            f"{sel_acc:.1%}", f"{arg_acc:.1%}",
            f"{avg_in:.0f}", f"{avg_out:.0f}",
            f"{hit_pct:.1%}",
            f"{avg_lat:.2f}s", errs,
        ]
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="agent-gateway Phase 0 smoke runner")
    ap.add_argument("--arms", default="A,B,D", help="Comma-separated arm list (A,B,D)")
    ap.add_argument("--cases", type=int, default=None, help="Limit number of cases")
    ap.add_argument(
        "--inflate-sizes",
        default=None,
        help="Comma-separated tool-count targets (e.g. 10,50,200). "
             "If unset, runs at native tool count.",
    )
    ap.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each (case, arm) sequentially N times to test prompt-cache amortization. "
             "Calls within the same (case, arm) run back-to-back so cache locality holds.",
    )
    ap.add_argument("--seed", type=int, default=0, help="Seed for distractor inflation")
    args = ap.parse_args()

    arm_keys = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arm_keys:
        if a not in ARMS:
            raise SystemExit(f"unknown arm: {a} (have {sorted(ARMS)})")

    cases = SEED_CASES[: args.cases] if args.cases else SEED_CASES
    sizes: list[int | None]
    if args.inflate_sizes:
        sizes = [int(s) for s in args.inflate_sizes.split(",") if s.strip()]
    else:
        sizes = [None]

    print(provider_info())
    total_calls = len(cases) * len(arm_keys) * len(sizes) * args.repeat
    print(
        f"Running {len(cases)} cases x {len(arm_keys)} arms x {len(sizes)} sizes "
        f"x {args.repeat} repeats = {total_calls} reasoner trips"
    )
    if "D" in arm_keys:
        print(f"  (+ {len(cases) * len(sizes) * args.repeat} broker trips for arm D)")

    all_results = []
    aggregate_rows = []

    for size in sizes:
        size_label = f"N={size}" if size else "native"
        print(f"\n=== {size_label} ===")
        size_results = []

        # Outer loop: case x arm, so all repeats for one (case, arm) run back-to-back
        # — this preserves cache locality so prompt caching can kick in.
        for case in cases:
            target_case = _inflate_case(case, size, args.seed) if size else case
            n_tools = len(target_case.tools)
            for arm in arm_keys:
                for repeat_idx in range(args.repeat):
                    print(f"  [{arm} r{repeat_idx}] {case.case_id} (tools={n_tools}) ... ",
                          end="", flush=True)
                    r = ARMS[arm](target_case)
                    r.extra["size"] = size or n_tools
                    r.extra["repeat_idx"] = repeat_idx
                    tag = "OK" if r.error is None else f"ERR({r.error[:40]})"
                    hit = r.extra.get("cache_hit", 0) or 0
                    print(
                        f"{tag}  tool={r.selected_tool} acc={r.arg_accuracy:.2f} "
                        f"tok={r.prompt_tokens}({hit}hit)+{r.completion_tokens} "
                        f"t={r.latency_s:.2f}s"
                    )
                    size_results.append(r)
        all_results.extend(size_results)

        # Per-repeat aggregation: shows the cost-drop curve
        if args.repeat > 1:
            for repeat_idx in range(args.repeat):
                rs = [r for r in size_results if r.extra.get("repeat_idx") == repeat_idx]
                aggregate_rows.extend(
                    _aggregate(rs, arm_keys, label_extras={"size": size_label, "rep": f"r{repeat_idx}"})
                )
        else:
            aggregate_rows.extend(_aggregate(size_results, arm_keys, label_extras={"size": size_label}))

    ts = time.strftime("%Y%m%dT%H%M%S")
    out_path = RUN_DIR / f"run_{ts}.jsonl"
    write_jsonl(out_path, [r.to_dict() for r in all_results])
    print(f"\nWrote {out_path}")

    headers_base = ["arm", "n", "tool-sel", "arg-acc", "in-tok", "out-tok", "cache-hit", "latency", "errs"]
    if args.repeat > 1:
        headers = ["size", "rep"] + headers_base
    else:
        headers = ["size"] + headers_base

    print()
    print(tabulate(aggregate_rows, headers=headers))


if __name__ == "__main__":
    main()
