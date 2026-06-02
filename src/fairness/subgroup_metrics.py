"""Subgroup metric utilities."""

from __future__ import annotations

import pandas as pd

from src.metrics import binary_classification_metrics


def subgroup_metrics(
    frame: pd.DataFrame,
    y_true,
    y_score,
    group_column: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute classification metrics for each subgroup."""

    working = frame[[group_column]].copy()
    working["_y_true"] = y_true
    working["_y_score"] = y_score

    rows = []
    for group_value, group_frame in working.groupby(group_column, dropna=False):
        metrics = binary_classification_metrics(
            group_frame["_y_true"],
            group_frame["_y_score"],
            threshold=threshold,
        )
        metrics[group_column] = group_value
        metrics["n"] = len(group_frame)
        rows.append(metrics)

    return pd.DataFrame(rows).sort_values("n", ascending=False)


def max_group_gap(metrics_frame: pd.DataFrame, metric: str) -> float:
    """Return max-min gap for one metric across groups."""

    values = metrics_frame[metric].dropna()
    if values.empty:
        return float("nan")
    return float(values.max() - values.min())
