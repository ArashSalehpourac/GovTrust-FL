"""Unit tests for the reliability metrics."""

from __future__ import annotations

import numpy as np
import pytest

from src.redesign.reliability import (
    binary_reliability_metrics,
    pinball_loss,
    regression_metrics,
)


def test_pinball_loss_is_asymmetric() -> None:
    y_true = np.array([1.0, 1.0])
    over = pinball_loss(y_true, np.array([2.0, 2.0]), 0.9)
    under = pinball_loss(y_true, np.array([0.0, 0.0]), 0.9)
    assert under > over
    assert pinball_loss(y_true, y_true, 0.5) == pytest.approx(0.0)


def test_regression_metrics_recover_perfect_predictions() -> None:
    y_true = np.log1p(np.array([10.0, 40.0, 100.0, 250.0]))
    predictions = {0.1: y_true - 0.5, 0.5: y_true, 0.9: y_true + 0.5}
    metrics = regression_metrics(y_true, predictions)
    assert metrics["mae_log1p_hours"] == pytest.approx(0.0)
    assert metrics["rmse_log1p_hours"] == pytest.approx(0.0)
    assert metrics["mae_hours"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["interval_empirical_coverage"] == pytest.approx(1.0)
    assert metrics["interval_nominal_coverage"] == pytest.approx(0.8)


def test_quantile_coverage_tracks_nominal_level() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=20_000)
    predictions = {
        0.1: np.full(20_000, -1.2816),
        0.5: np.zeros(20_000),
        0.9: np.full(20_000, 1.2816),
    }
    metrics = regression_metrics(y_true, predictions)
    assert metrics["coverage_below_q10"] == pytest.approx(0.1, abs=0.02)
    assert metrics["coverage_below_q90"] == pytest.approx(0.9, abs=0.02)
    assert metrics["interval_coverage_gap"] == pytest.approx(0.0, abs=0.02)


def test_binary_reliability_metrics_reward_calibration() -> None:
    y_true = np.array([0, 0, 1, 1])
    calibrated = binary_reliability_metrics(y_true, np.array([0.0, 0.0, 1.0, 1.0]))
    miscalibrated = binary_reliability_metrics(y_true, np.array([0.9, 0.9, 0.1, 0.1]))
    assert calibrated["brier"] == pytest.approx(0.0)
    assert calibrated["ece"] == pytest.approx(0.0)
    assert miscalibrated["brier"] > calibrated["brier"]
    assert miscalibrated["ece"] > calibrated["ece"]
    assert calibrated["positive_rate"] == pytest.approx(0.5)


def test_binary_metrics_handle_single_class() -> None:
    metrics = binary_reliability_metrics(np.zeros(10), np.full(10, 0.2))
    assert np.isnan(metrics["auroc"])
    assert metrics["brier"] == pytest.approx(0.04)
