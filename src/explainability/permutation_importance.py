"""Permutation-importance helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance


def compute_permutation_importance(model, x, y, *, n_repeats: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Compute sklearn permutation importance as a tidy DataFrame."""

    result = permutation_importance(model, x, y, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
    return pd.DataFrame(
        {
            "feature": list(x.columns),
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
