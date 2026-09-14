"""Tests for held-out-city isolation during training and for DP accounting."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.redesign.config import build_folds
from src.redesign.dp import ACCOUNTANT, plan_noise_multiplier, recompute_epsilon
from src.redesign.federated import FederatedConfig, run_federated
from src.redesign.folds import build_fold


def tiny_config(**overrides) -> FederatedConfig:
    defaults = dict(
        rounds=2,
        local_epochs=1,
        batch_size=64,
        hidden_sizes=(8,),
        seed=0,
        subgroup_min_rows=10,
    )
    defaults.update(overrides)
    return FederatedConfig(**defaults)


@pytest.fixture(scope="module")
def small_frames():
    from tests.redesign.conftest import make_city_frame
    from src.redesign.config import CITIES

    return {city: make_city_frame(city, n_rows=240, seed=i) for i, city in enumerate(CITIES)}


@pytest.fixture(scope="module")
def fold(small_frames):
    return build_fold(build_folds()[0], small_frames)


@pytest.mark.parametrize("algorithm", ["fedavg", "fedprox"])
def test_runs_report_internal_and_external_separately(fold, algorithm) -> None:
    result = run_federated(fold, tiny_config(algorithm=algorithm))
    payload = result.to_dict()

    assert set(payload["metrics_internal_source_cities"]["by_city_val"]) == set(
        fold.spec.source_cities
    )
    assert fold.spec.held_out_city not in payload["metrics_internal_source_cities"]["by_city_val"]
    assert payload["metrics_external_held_out_city"]["city"] == fold.spec.held_out_city

    for scope in (
        payload["metrics_internal_source_cities"]["pooled_internal_test"],
        payload["metrics_external_held_out_city"]["report"],
    ):
        assert set(scope) == {"primary", "secondary_sla"}
        assert math.isfinite(scope["primary"]["mae_log1p_hours"])
        assert math.isfinite(scope["primary"]["rmse_log1p_hours"])
        assert math.isfinite(scope["primary"]["mean_pinball"])
        assert 0.0 <= scope["secondary_sla"]["brier"] <= 1.0
        assert 0.0 <= scope["secondary_sla"]["ece"] <= 1.0


def test_training_and_selection_never_touch_the_held_out_city(fold, small_frames, poison_frame) -> None:
    """Poisoning the held-out city must not move a single trained weight."""

    spec = fold.spec
    poisoned_frames = dict(small_frames)
    poisoned_frames[spec.held_out_city] = poison_frame(small_frames[spec.held_out_city])
    poisoned_fold = build_fold(spec, poisoned_frames)

    clean = run_federated(fold, tiny_config())
    poisoned = run_federated(poisoned_fold, tiny_config())

    assert clean.selection == poisoned.selection
    assert (
        clean.internal["pooled_internal_test"]["primary"]["mae_log1p_hours"]
        == poisoned.internal["pooled_internal_test"]["primary"]["mae_log1p_hours"]
    )
    # Only the external evaluation, which is never fit on, may differ.
    assert (
        clean.external["report"]["primary"]["mae_log1p_hours"]
        != poisoned.external["report"]["primary"]["mae_log1p_hours"]
    )


def test_subgroup_reporting_is_external_and_operational(fold) -> None:
    result = run_federated(fold, tiny_config())
    assert result.subgroups
    groupings = {entry["grouping"] for entry in result.subgroups}
    assert groupings <= {"service_category", "area_group", "request_volume_stratum"}
    for entry in result.subgroups:
        assert entry["rows"] >= 10
        assert entry["city_scope"] == fold.spec.held_out_city


def test_dp_accounting_matches_the_training_mechanism(fold) -> None:
    config = tiny_config(target_epsilon=1.0, delta=1e-5)
    result = run_federated(fold, config)

    assert result.privacy["enabled"] is True
    assert set(result.privacy["by_city"]) == set(fold.spec.source_cities)
    for city, report in result.privacy["by_city"].items():
        assert report["mechanism"] == "example-level DP-SGD (Opacus PrivacyEngine)"
        assert report["accountant"] == ACCOUNTANT
        assert report["delta"] == 1e-5
        assert report["max_grad_norm"] == config.max_grad_norm
        assert report["noise_multiplier"] > 0
        assert report["epsilon_spent"] <= report["target_epsilon"] + 1e-6
        assert report["accounted_steps"] == report["steps_taken"]
        assert report["steps_taken"] <= report["planned_steps"]

        # Independently recompute epsilon from the reported mechanism parameters.
        recomputed = recompute_epsilon(
            noise_multiplier=report["noise_multiplier"],
            sample_rate=report["sample_rate"],
            steps=report["accounted_steps"],
            delta=report["delta"],
        )
        assert recomputed == pytest.approx(report["epsilon_spent"], rel=1e-6)


def test_no_dp_when_epsilon_is_infinite(fold) -> None:
    result = run_federated(fold, tiny_config(target_epsilon=float("inf")))
    assert result.privacy["enabled"] is False
    assert all(report is None for report in result.privacy["by_city"].values())


def test_noise_multiplier_increases_as_epsilon_tightens() -> None:
    loose = plan_noise_multiplier(
        target_epsilon=5.0, delta=1e-5, sample_rate=0.05, steps=100
    )
    tight = plan_noise_multiplier(
        target_epsilon=1.0, delta=1e-5, sample_rate=0.05, steps=100
    )
    assert tight > loose > 0
    assert plan_noise_multiplier(
        target_epsilon=float("inf"), delta=1e-5, sample_rate=0.05, steps=100
    ) == 0.0


def test_tighter_epsilon_costs_external_reliability_is_measurable(fold) -> None:
    """Both DP levels must produce complete, finite reports for every fold."""

    reports = {
        epsilon: run_federated(fold, tiny_config(target_epsilon=epsilon)).external["report"]
        for epsilon in (float("inf"), 1.0)
    }
    for report in reports.values():
        assert math.isfinite(report["primary"]["mae_log1p_hours"])
        assert math.isfinite(report["secondary_sla"]["brier"])


def test_quantiles_do_not_cross(fold) -> None:
    from src.redesign.model import QuantileDelayMLP, predict

    torch.manual_seed(0)
    model = QuantileDelayMLP(fold.preprocessor.output_dim, (8,), n_quantiles=3)
    features = torch.from_numpy(
        fold.preprocessor.transform(fold.external.head(64))
    )
    quantile_predictions, probabilities = predict(model, features, (0.1, 0.5, 0.9))
    assert np.all(quantile_predictions[0.1] <= quantile_predictions[0.5] + 1e-6)
    assert np.all(quantile_predictions[0.5] <= quantile_predictions[0.9] + 1e-6)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
