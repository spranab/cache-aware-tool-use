# Cache-Aware Tool Use in Multi-Tenant LLM Systems

Companion artifact for the paper *Cache-Aware Tool Use in Multi-Tenant LLM Systems: A Cost Model, Cross-Provider Measurement, and Architectural Decision Framework* by Pranab Sarkar (Independent Researcher, ORCID [0009-0009-8683-1481](https://orcid.org/0009-0009-8683-1481)).

The paper studies a previously unexamined cache-economics problem in multi-tenant LLM tool use: when tenant-specific personalization precedes large stable tool schemas in the prompt, prefix-cache fragmentation causes schema cost to scale O(K) in tenant count. Cache-aware prompt structure recovers O(1) for shared tool catalogs. Goal-delegation behind a unified broker catalog recovers O(1) even when tool catalogs are tenant-variable — the prevailing case in multi-tenant SaaS.

## Headline result

Multi-tenant simulation on DeepSeek with K personalized tenants over 200 tools, 3 sequential calls per tenant:

| K       | Naive direct (A) hit rate | Goal-delegation (D) hit rate | A / D total cost ratio |
|---------|---------------------------|-----------------------------|------------------------|
| 10      | 82.7%                     | 98.7%                       | 1.69×                  |
| 50      | 72.8%                     | 98.3%                       | 2.19×                  |
| 200     | 74.4%                     | 98.4%                       | 2.10×                  |

Delegation's per-user cost is invariant in K (architectural O(1) confirmed empirically); naive injection's cost grows then plateaus around 74% hit rate due to per-tenant cache fragmentation.

See [paper/PAPER.md](paper/PAPER.md) for the full manuscript including formal cost model, six-architecture comparison, cross-provider taxonomy, and decision framework.

## Repository layout

```
.
├── paper/
│   ├── PAPER.md                # manuscript (CC-BY-4.0)
│   └── figures/
│       ├── plot.py             # regenerates all figures from runs/*.jsonl
│       ├── fig_*.png           # generated figures
│       └── measured_results.json
├── bench/
│   ├── arms.py                 # Arm A (naive direct), B (top-m retrieval), D (broker)
│   ├── common.py               # provider config, types
│   ├── inflate.py              # tool-set distractor inflation
│   ├── multitenant.py          # async multi-tenant simulator
│   ├── personas.py             # deterministic synthetic personas
│   ├── runner.py               # single-tenant CLI
│   ├── score.py                # AST-based argument scoring
│   └── seed_cases.py           # 5 seed test cases
├── requirements.txt
├── .env.example                # copy to .env and populate API keys
├── LICENSE                     # Apache-2.0 (code)
└── LICENSE-PAPER               # CC-BY-4.0 (manuscript)
```

## Reproducing the measurements

```bash
git clone https://github.com/spranab/cache-aware-tool-use.git
cd cache-aware-tool-use
pip install -r requirements.txt
cp .env.example .env
# populate DEEPSEEK_API_KEY (and optionally OPENAI_API_KEY) in .env

# Single-tenant smoke test (~$0.05 on DeepSeek)
python -m bench.runner --inflate-sizes 200 --repeat 5

# Multi-tenant scaling (Table 1 of the paper, ~$1 on DeepSeek)
python -m bench.multitenant --users 10,50,200 --calls 3 --tools 200

# Regenerate figures from run logs
python paper/figures/plot.py
```

`runs/*.jsonl` is git-ignored; each invocation appends a fresh log. Figures regenerate deterministically from the most recent logs.

## Status

This is a pre-registered draft (v0.3). Cross-provider replication (Anthropic, OpenAI, self-hosted vLLM), BFCL accuracy evaluation, and per-tenant tool-overlap sweeps are in progress. Seven falsifiability conditions are listed in [§9 of the paper](paper/PAPER.md#9-threats-to-validity).

## Citation

```bibtex
@misc{sarkar2026cacheaware,
  author       = {Sarkar, Pranab},
  title        = {Cache-Aware Tool Use in Multi-Tenant {LLM} Systems:
                  A Cost Model, Cross-Provider Measurement, and
                  Architectural Decision Framework},
  year         = {2026},
  howpublished = {GitHub: spranab/cache-aware-tool-use},
  note         = {Pre-registered draft v0.3}
}
```

Companion paper: Sarkar, P. (2026). *Skill as Memory, Not Document: A Database-Native Substrate for Agent Skill Catalogs.* Zenodo. [doi:10.5281/zenodo.20128887](https://doi.org/10.5281/zenodo.20128887)

## License

- Source code under Apache License 2.0 (see [LICENSE](LICENSE))
- Manuscript text and figures under CC-BY-4.0 (see [LICENSE-PAPER](LICENSE-PAPER))
