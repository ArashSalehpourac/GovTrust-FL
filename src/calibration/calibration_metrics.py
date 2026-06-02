"""Calibration metrics and wrappers."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    """Compute binary expected calibration error."""

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score, bins[1:-1], right=True)

    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        confidence = y_score[mask].mean()
        accuracy = y_true[mask].mean()
        ece += mask.mean() * abs(accuracy - confidence)
    return float(ece)


def calibration_summary(y_true, y_score, n_bins: int = 10) -> dict[str, object]:
    """Return ECE and calibration curve points."""

    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="uniform")
    return {
        "expected_calibration_error": expected_calibration_error(y_true, y_score, n_bins=n_bins),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }


def make_calibrated_classifier(estimator, method: str = "isotonic", cv: int = 3):
    """Wrap an estimator with sklearn probability calibration."""

    return CalibratedClassifierCV(estimator=estimator, method=method, cv=cv)
