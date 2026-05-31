"""Privacy-Explanation Drift Index utilities."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def spearman_rank_stability(reference, candidate) -> float:
    """Compute Spearman rank correlation between explanation vectors."""

    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape:
        raise ValueError("Explanation vectors must have the same shape")
    if reference.size < 2:
        return float("nan")
    correlation = spearmanr(reference, candidate, nan_policy="omit").correlation
    return float(correlation)


def top_k_overlap(reference, candidate, k: int = 10) -> float:
    """Return top-k feature-overlap ratio."""

    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape:
        raise ValueError("Explanation vectors must have the same shape")
    k = min(int(k), reference.size)
    if k <= 0:
        return float("nan")
    ref_top = set(np.argsort(np.abs(reference))[::-1][:k])
    cand_top = set(np.argsort(np.abs(candidate))[::-1][:k])
    return float(len(ref_top & cand_top) / k)


def privacy_explanation_drift_index(reference, candidate) -> float:
    """Compute PEDI = 1 - Spearman(E_reference, E_private)."""

    stability = spearman_rank_stability(reference, candidate)
    return float(1.0 - stability)
