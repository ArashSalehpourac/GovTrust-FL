import importlib.util
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from pathlib import Path

from src.dp_training import DPTrainingConfig
from src.feature_engineering import fit_descriptor_vectorizer
from src.federated.strategies import fedprox_proximal_term
from src.governance import REQUIRED_TRANSPARENCY_FIELDS, transparency_record_payload, validate_transparency_record
from src.privacy import membership_inference_metrics
from src.secure_aggregation import simulated_secure_aggregate
from src.tai_score import build_scorecard


def test_fedprox_proximal_penalty_behavior():
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -1.0]]))
    global_parameters = [torch.zeros_like(model.weight)]

    penalty = fedprox_proximal_term(model, global_parameters)

    assert torch.isclose(penalty, torch.tensor(5.0))


def test_dp_configuration_validation():
    assert DPTrainingConfig(enabled=True, noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5).enabled
    with pytest.raises(ValueError):
        DPTrainingConfig(noise_multiplier=-0.1)
    with pytest.raises(ValueError):
        DPTrainingConfig(max_grad_norm=0.0)
    with pytest.raises(ValueError):
        DPTrainingConfig(delta=1.0)


def test_secure_aggregation_overhead_calculation():
    updates = [
        ([np.array([1.0, 3.0], dtype=np.float32)], 1),
        ([np.array([3.0, 5.0], dtype=np.float32)], 3),
    ]

    aggregated, info = simulated_secure_aggregate(updates, overhead_factor=0.5)

    np.testing.assert_allclose(aggregated[0], np.array([2.5, 4.5], dtype=np.float32))
    assert info["secure_aggregation_simulated"] is True
    assert info["secure_aggregation_overhead_bytes"] == aggregated[0].nbytes * len(updates) * 0.5


def test_membership_inference_output_schema():
    row = membership_inference_metrics([0.9, 0.8, 0.7], [0.55, 0.52, 0.51])

    assert {
        "attack_accuracy",
        "attack_auc",
        "attack_advantage",
        "member_rows",
        "nonmember_rows",
    } <= set(row)
    assert row["member_rows"] == 3
    assert row["nonmember_rows"] == 3
    assert 0 <= row["attack_advantage"] <= 1


def test_governance_record_required_fields_and_la_holdout():
    record = transparency_record_payload()

    assert REQUIRED_TRANSPARENCY_FIELDS <= set(record)
    validate_transparency_record(record)

    bad_record = {**record, "training_cities": ["nyc", "los_angeles"]}
    with pytest.raises(ValueError):
        validate_transparency_record(bad_record)


def test_los_angeles_not_used_to_fit_text_preprocessing(tmp_path):
    frames = {
        "nyc": pd.DataFrame({"descriptor": ["noise complaint", "road issue"]}),
        "chicago": pd.DataFrame({"descriptor": ["sanitation pickup", "noise issue"]}),
        "boston": pd.DataFrame({"descriptor": ["housing heat", "road repair"]}),
        "los_angeles": pd.DataFrame({"descriptor": ["exclusive_los_angeles_token"]}),
    }

    vectorizer = fit_descriptor_vectorizer(frames, tmp_path / "tfidf.joblib", max_features=20, min_df=1)

    vocabulary = set(vectorizer.get_feature_names_out())
    assert "exclusive_los_angeles_token" not in vocabulary


def test_los_angeles_excluded_from_tai_scorecard_normalization():
    module = load_trustworthiness_eval_module()
    performance = pd.DataFrame(
        {
            "model": ["internal_strong", "internal_strong", "internal_weak", "internal_weak"],
            "test_city": ["nyc", "los_angeles", "nyc", "los_angeles"],
            "macro_f1": [0.90, 0.05, 0.10, 1.00],
            "test_rows": [100, 10_000, 100, 10_000],
        }
    )
    calibration = pd.DataFrame(
        {
            "model": ["internal_strong", "internal_strong", "internal_weak", "internal_weak"],
            "city": ["nyc", "los_angeles", "nyc", "los_angeles"],
            "ece": [0.10, 0.99, 0.10, 0.01],
        }
    )
    fairness = pd.DataFrame(
        {
            "model": ["internal_strong", "internal_weak"],
            "macro_f1_gap": [0.10, 0.10],
        }
    )
    resources = pd.DataFrame(
        {
            "model": ["internal_strong", "internal_weak"],
            "runtime_sec": [1.0, 1.0],
        }
    )

    tai_input = module.build_tai_input(
        performance,
        calibration,
        fairness,
        resources,
        privacy_table=None,
        pedi_table=None,
        external_validation_city="los_angeles",
    )
    utility = tai_input.set_index("model")["utility"]
    calibration_error = tai_input.set_index("model")["calibration_error"]
    scorecard = build_scorecard(tai_input)

    assert utility["internal_strong"] == 0.90
    assert utility["internal_weak"] == 0.10
    assert calibration_error["internal_strong"] == 0.10
    assert calibration_error["internal_weak"] == 0.10
    assert scorecard.iloc[0]["model"] == "internal_strong"


def load_trustworthiness_eval_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_trustworthiness_eval.py"
    spec = importlib.util.spec_from_file_location("run_trustworthiness_eval_for_tests", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
