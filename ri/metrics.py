"""Initial metrics stored with every data point: pass@k, pass^k and efficiency."""

from __future__ import annotations

from math import comb
from statistics import mean, median

PASS_AT_K = (1, 4, 8, 16, 32, 64, 128)
PASS_HAT_K = (2, 4, 8)
BUDGETS = (1024, 2048, 4096, 8192)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Chance that at least one of k attempts (drawn without replacement from n) is correct."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Chance that all k attempts (drawn without replacement from n) are correct."""
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)


def summarize(attempts: list[dict]) -> dict:
    """attempts: [{'tokens': int, 'truncated': bool, 'correct': bool}, ...]"""
    n = len(attempts)
    c = sum(a["correct"] for a in attempts)
    toks = [a["tokens"] for a in attempts]
    tok_c = [a["tokens"] for a in attempts if a["correct"]]
    tok_w = [a["tokens"] for a in attempts if not a["correct"]]
    return {
        "n_attempts": n,
        "n_correct": c,
        "pass@k": {str(k): round(pass_at_k(n, c, k), 6) for k in PASS_AT_K if k <= n},
        "pass^k": {str(k): round(pass_hat_k(n, c, k), 6) for k in PASS_HAT_K if k <= n},
        "efficiency": {
            "mean_tokens": round(mean(toks), 1),
            "median_tokens": median(toks),
            "mean_tokens_correct": round(mean(tok_c), 1) if tok_c else None,
            "mean_tokens_wrong": round(mean(tok_w), 1) if tok_w else None,
            "truncation_rate": round(sum(a["truncated"] for a in attempts) / n, 4),
            "tokens_per_correct": round(sum(toks) / c, 1) if c else None,
            "success_within_tokens": {
                str(b): round(sum(a["correct"] and a["tokens"] <= b for a in attempts) / n, 4)
                for b in BUDGETS
            },
        },
    }


def pool_summary(items: list[dict]) -> dict:
    """Means over items of the per-item initial metrics (what each JSON file's header shows)."""
    ms = [it["initial_metrics"] for it in items]
    if not ms:
        return {}
    keys_at = sorted({k for m in ms for k in m["pass@k"]}, key=int)
    keys_hat = sorted({k for m in ms for k in m["pass^k"]}, key=int)
    return {
        "n_items": len(items),
        "mean_pass@k": {k: round(mean(m["pass@k"][k] for m in ms if k in m["pass@k"]), 6) for k in keys_at},
        "mean_pass^k": {k: round(mean(m["pass^k"][k] for m in ms if k in m["pass^k"]), 6) for k in keys_hat},
        "mean_tokens": round(mean(m["efficiency"]["mean_tokens"] for m in ms), 1),
        "mean_truncation_rate": round(mean(m["efficiency"]["truncation_rate"] for m in ms), 4),
    }
