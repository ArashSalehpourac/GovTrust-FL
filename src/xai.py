"""Explainability helpers for SHAP and explanation stability."""

import numpy as np
import pandas as pd


def compute_shap_values(model, features, sample_size: int = 1_000, random_state: int = 42):
    """Compute SHAP values for a sample of rows."""

    import shap

    sample = features.sample(min(sample_size, len(features)), random_state=random_state)
    explainer = shap.Explainer(model, sample)
    values = explainer(sample)
    return sample, values


def mean_absolute_importance(shap_values, feature_names) -> pd.DataFrame:
    """Summarize mean absolute SHAP importance."""

    values = np.asarray(shap_values.values)
    if values.ndim == 3:
        values = values[:, :, -1]

    importance = np.abs(values).mean(axis=0)
    return pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance",
        ascending=False,
    )


def rank_stability(reference_importance: pd.DataFrame, candidate_importance: pd.DataFrame) -> float:
    """Compute Spearman rank stability between two feature-importance tables."""

    merged = reference_importance[["feature", "importance"]].merge(
        candidate_importance[["feature", "importance"]],
        on="feature",
        how="inner",
        suffixes=("_reference", "_candidate"),
    )
    if len(merged) < 2:
        return float("nan")
    return float(merged["importance_reference"].corr(merged["importance_candidate"], method="spearman"))

