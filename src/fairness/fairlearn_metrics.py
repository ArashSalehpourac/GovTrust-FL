"""Optional Fairlearn metric wrappers."""

from __future__ import annotations

import pandas as pd


def fairlearn_metric_frame(y_true, y_pred, sensitive_features) -> pd.DataFrame:
    """Return a Fairlearn MetricFrame as a DataFrame when Fairlearn is installed."""

    try:
        from fairlearn.metrics import MetricFrame, false_negative_rate, false_positive_rate, selection_rate
        from sklearn.metrics import precision_score, recall_score
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("Install fairlearn to use fairlearn_metric_frame") from exc

    metric_frame = MetricFrame(
        metrics={
            "precision": precision_score,
            "recall": recall_score,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "selection_rate": selection_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    return metric_frame.by_group.reset_index()
