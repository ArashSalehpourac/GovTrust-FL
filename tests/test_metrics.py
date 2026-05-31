import numpy as np

from src.calibration import expected_calibration_error
from src.metrics import binary_classification_metrics


def test_binary_metrics_include_paper_fields():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.7, 0.9])

    metrics = binary_classification_metrics(y_true, y_score)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_expected_calibration_error_bounds():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.0, 0.25, 0.75, 1.0])

    ece = expected_calibration_error(y_true, y_score, n_bins=2)

    assert 0.0 <= ece <= 1.0
