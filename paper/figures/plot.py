"""Generate paper figures from benchmark run logs.

Reads JSONL files in ../../runs/ and produces PNG figures into the current directory.

Run from agent-gateway/ with:
    python paper/figures/plot.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs"
FIG_DIR = Path(__file__).resolve().parent

PRICE_HIT = 0.027
PRICE_MISS = 0.27
PRICE_OUT = 1.10


def load_multitenant_runs():
    """Load every multitenant_*.jsonl run, return list of dicts keyed by K and arm."""
    runs = []
    for f in sorted(RUN_DIR.glob("multitenant_*.jsonl")):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        runs.append({"path": f, "rows": rows})
    return runs


def aggregate_by_K_arm(rows):
    by = defaultdict(lambda: defaultdict(lambda: {
        "n_calls": 0, "in_tok": 0, "out_tok": 0,
        "cache_hit": 0, "cache_miss": 0, "latency": 0.0,
        "n_users": 0,
    }))
    seen_users = defaultdict(set)
    for r in rows:
        K = r.get("n_users") or r.get("K") or 0
        arm = r["arm"]
        b = by[K][arm]
        b["n_calls"] += 1
        b["in_tok"] += r["prompt_tokens"]
        b["out_tok"] += r["completion_tokens"]
        b["cache_hit"] += r.get("cache_hit", 0) or 0
        b["cache_miss"] += r.get("cache_miss", 0) or 0
        b["latency"] += r.get("latency", 0.0)
        seen_users[(K, arm)].add(r.get("user_id"))
    for (K, arm), users in seen_users.items():
        by[K][arm]["n_users"] = len(users)
    return by


def compute_metrics(agg):
    out = {}
    for K, by_arm in agg.items():
        out[K] = {}
        for arm, b in by_arm.items():
            n = b["n_calls"]
            total_billed = b["cache_hit"] * PRICE_HIT + b["cache_miss"] * PRICE_MISS
            cost_input = total_billed / 1_000_000
            cost_output = b["out_tok"] * PRICE_OUT / 1_000_000
            cost_total = cost_input + cost_output
            hit_pool = b["cache_hit"] + b["cache_miss"]
            out[K][arm] = {
                "n_calls": n,
                "n_users": b["n_users"],
                "hit_rate": (b["cache_hit"] / hit_pool) if hit_pool else 0.0,
                "in_tok": b["in_tok"],
                "out_tok": b["out_tok"],
                "cost_input": cost_input,
                "cost_output": cost_output,
                "cost_total": cost_total,
                "cost_per_user_per_1k": (cost_total / b["n_users"]) * 1000 if b["n_users"] else 0,
                "avg_latency": b["latency"] / n if n else 0,
            }
    return out


ARM_COLORS = {"A": "#d62728", "B": "#7f7f7f", "D": "#1f77b4"}
ARM_LABELS = {
    "A": "A — Naive direct injection",
    "B": "B — Top-k lexical retrieval",
    "D": "D — Goal-delegation broker",
}


def plot_hit_rate_vs_K(metrics, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    Ks = sorted(metrics.keys())
    for arm in ("A", "D"):
        ys = [metrics[K].get(arm, {}).get("hit_rate", 0) * 100 for K in Ks]
        ax.plot(Ks, ys, marker="o", linewidth=2, label=ARM_LABELS[arm], color=ARM_COLORS[arm])
    ax.set_xscale("log")
    ax.set_xlabel("K — number of personalized tenants")
    ax.set_ylabel("Prompt-cache hit rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Cache hit rate vs tenant count (DeepSeek, N=200 tools, P≈50 token persona)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    ax.axhline(98.4, color="#1f77b4", linestyle=":", alpha=0.4, label="_D asymptote")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cost_per_user_vs_K(metrics, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    Ks = sorted(metrics.keys())
    for arm in ("A", "D"):
        ys = [metrics[K].get(arm, {}).get("cost_per_user_per_1k", 0) for K in Ks]
        ax.plot(Ks, ys, marker="o", linewidth=2, label=ARM_LABELS[arm], color=ARM_COLORS[arm])
    ax.set_xscale("log")
    ax.set_xlabel("K — number of personalized tenants")
    ax.set_ylabel("Cost per 1,000 user-sessions (USD)")
    ax.set_title("Per-tenant cost vs tenant count (DeepSeek, N=200 tools)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cost_ratio_vs_K(metrics, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    Ks = sorted(metrics.keys())
    ratios_total = []
    ratios_input = []
    for K in Ks:
        A, D = metrics[K].get("A", {}), metrics[K].get("D", {})
        rt = (A.get("cost_total", 0) / D["cost_total"]) if D.get("cost_total") else None
        ri = (A.get("cost_input", 0) / D["cost_input"]) if D.get("cost_input") else None
        ratios_total.append(rt)
        ratios_input.append(ri)
    ax.plot(Ks, ratios_input, marker="s", linewidth=2, label="Input-cost ratio (A / D)", color="#ff7f0e")
    ax.plot(Ks, ratios_total, marker="o", linewidth=2, label="Total-cost ratio (A / D)", color="#1f77b4")
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Parity")
    ax.set_xscale("log")
    ax.set_xlabel("K — number of personalized tenants")
    ax.set_ylabel("Cost ratio (A naive / D delegated)")
    ax.set_title("Cost advantage of delegation grows with tenant count")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cost_breakdown_at_K(metrics, K, out_path):
    if K not in metrics:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    arms = ["A", "D"]
    in_costs = [metrics[K][a]["cost_input"] for a in arms]
    out_costs = [metrics[K][a]["cost_output"] for a in arms]
    x = list(range(len(arms)))
    ax.bar(x, in_costs, label="Input (cached + uncached)", color="#1f77b4")
    ax.bar(x, out_costs, bottom=in_costs, label="Output", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABELS[a] for a in arms])
    ax.set_ylabel(f"Total cost at K={K} (USD)")
    ax.set_title(f"Cost decomposition at K={K} ({metrics[K]['A']['n_calls']} calls per arm)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_theoretical_scaling(out_path):
    """Theoretical cache-miss cost (no measurement, just the cost-model intuition)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    K = list(range(1, 1001))
    S = 10_000
    miss_cost_per_K_tokens_A = [k * S for k in K]
    miss_cost_per_K_tokens_D = [S for _ in K]
    ax.plot(K, miss_cost_per_K_tokens_A, label="A naive — O(K·S)", linewidth=2, color="#d62728")
    ax.plot(K, miss_cost_per_K_tokens_D, label="A′ cache-aware / D broker — O(S)", linewidth=2, color="#1f77b4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K — number of personalized tenants")
    ax.set_ylabel("Total schema cache-miss tokens (cold population)")
    ax.set_title("Cache-miss cost scaling (theoretical, S = 10,000 schema tokens)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    runs = load_multitenant_runs()
    if not runs:
        print(f"No multitenant_*.jsonl found in {RUN_DIR}")
        return

    print(f"Loaded {len(runs)} multitenant run(s):")
    all_rows = []
    for r in runs:
        print(f"  {r['path'].name}: {len(r['rows'])} records")
        all_rows.extend(r["rows"])

    agg = aggregate_by_K_arm(all_rows)
    metrics = compute_metrics(agg)

    Ks = sorted(metrics.keys())
    print(f"\nMetrics across K={Ks}:")
    for K in Ks:
        for arm, m in metrics[K].items():
            print(f"  K={K} arm={arm}: hit={m['hit_rate']:.1%} "
                  f"$/1k-user={m['cost_per_user_per_1k']:.3f} "
                  f"calls={m['n_calls']}")

    plot_hit_rate_vs_K(metrics, FIG_DIR / "fig_hit_rate_vs_k.png")
    plot_cost_per_user_vs_K(metrics, FIG_DIR / "fig_cost_per_user_vs_k.png")
    plot_cost_ratio_vs_K(metrics, FIG_DIR / "fig_cost_ratio_vs_k.png")
    if max(Ks) >= 100:
        plot_cost_breakdown_at_K(metrics, max(Ks), FIG_DIR / "fig_cost_breakdown.png")
    plot_theoretical_scaling(FIG_DIR / "fig_theoretical_scaling.png")

    # Emit a results.json for the paper to reference
    out = {str(K): {arm: dict(m) for arm, m in metrics[K].items()} for K in Ks}
    (FIG_DIR / "measured_results.json").write_text(json.dumps(out, indent=2))

    print(f"\nWrote figures + measured_results.json to {FIG_DIR}")


if __name__ == "__main__":
    main()
