"""Model evaluation metrics."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_classification_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    """Compute common binary classification metrics."""

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_score),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    return {key: float(value) for key, value in metrics.items()}


def predict_scores(model, features) -> np.ndarray:
    """Return positive-class scores for classifiers with common sklearn APIs."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "decision_function"):
        values = model.decision_function(features)
        return 1 / (1 + np.exp(-values))
    return model.predict(features)

