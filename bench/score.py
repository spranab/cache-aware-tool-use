from __future__ import annotations

from typing import Any


def _normalize(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip().lower()
    return v


def score_arguments(expected: dict[str, Any], actual: dict[str, Any] | None) -> float:
    """Fraction of expected keys present in actual with matching (normalized) values.

    Smoke-test scorer. Phase 1 swaps in BFCL's official AST scorer.
    """
    if not expected:
        return 1.0
    if actual is None:
        return 0.0
    hits = 0
    for k, v_exp in expected.items():
        if k in actual and _normalize(actual[k]) == _normalize(v_exp):
            hits += 1
    return hits / len(expected)
