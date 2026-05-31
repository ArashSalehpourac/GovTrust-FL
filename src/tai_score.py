"""TAI-Score utilities for multi-objective trustworthiness ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_WEIGHTS = {
    "predictive_utility": 0.25,
    "privacy_protection": 0.20,
    "explanation_stability": 0.20,
    "fairness": 0.15,
    "efficiency": 0.10,
    "calibration": 0.10,
}


def minmax_normalize(values, *, higher_is_better: bool = True) -> pd.Series:
    """Normalize values to 0-1, with graceful handling of constants."""

    series = pd.Series(values, dtype=float)
    if series.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    min_value = series.min(skipna=True)
    max_value = series.max(skipna=True)
    if np.isclose(max_value, min_value):
        normalized = pd.Series(1.0, index=series.index)
    else:
        normalized = (series - min_value) / (max_value - min_value)
    if not higher_is_better:
        normalized = 1.0 - normalized
    return normalized.clip(0.0, 1.0)


def compute_tai_score(frame: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Compute weighted TAI-Score from normalized component columns."""

    weights = weights or DEFAULT_WEIGHTS
    missing = [column for column in weights if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing TAI component columns: {missing}")
    output = frame.copy()
    output["tai_score"] = 0.0
    for column, weight in weights.items():
        output["tai_score"] += output[column].fillna(0.0) * float(weight)
    return output


def build_scorecard(
    model_frame: pd.DataFrame,
    *,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Normalize raw columns and compute a TAI-Score table."""

    work = model_frame.copy()
    components = pd.DataFrame({"model": work["model"]})
    components["predictive_utility"] = minmax_normalize(work["utility"], higher_is_better=True)
    components["privacy_protection"] = minmax_normalize(work["privacy_risk"], higher_is_better=False)
    components["explanation_stability"] = minmax_normalize(work["pedi"], higher_is_better=False)
    components["fairness"] = minmax_normalize(work["fairness_gap"], higher_is_better=False)
    components["efficiency"] = minmax_normalize(work["runtime_sec"], higher_is_better=False)
    components["calibration"] = minmax_normalize(work["calibration_error"], higher_is_better=False)
    return compute_tai_score(components, weights=weights).sort_values("tai_score", ascending=False)
