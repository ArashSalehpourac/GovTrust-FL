import pandas as pd

from src.tai_score import build_scorecard, compute_tai_score, minmax_normalize


def test_minmax_normalize_handles_constant_values():
    normalized = minmax_normalize([5, 5, 5])
    assert normalized.tolist() == [1.0, 1.0, 1.0]


def test_compute_tai_score_weighted_sum():
    frame = pd.DataFrame(
        {
            "predictive_utility": [1.0],
            "privacy_protection": [0.5],
            "explanation_stability": [0.5],
            "fairness": [1.0],
            "efficiency": [0.0],
            "calibration": [1.0],
        }
    )

    scored = compute_tai_score(frame)

    assert round(scored["tai_score"].iloc[0], 6) == 0.70


def test_build_scorecard_ranks_better_tradeoff_first():
    raw = pd.DataFrame(
        {
            "model": ["strong", "weak"],
            "utility": [0.9, 0.6],
            "privacy_risk": [0.1, 0.8],
            "pedi": [0.1, 0.7],
            "fairness_gap": [0.05, 0.5],
            "runtime_sec": [10.0, 100.0],
            "calibration_error": [0.03, 0.4],
        }
    )

    scorecard = build_scorecard(raw)

    assert scorecard["model"].iloc[0] == "strong"


def test_tai_score_monotonicity_for_uniform_component_improvement():
    raw = pd.DataFrame(
        {
            "model": ["better", "worse"],
            "utility": [0.8, 0.6],
            "privacy_risk": [0.2, 0.4],
            "pedi": [0.1, 0.3],
            "fairness_gap": [0.1, 0.2],
            "runtime_sec": [10.0, 20.0],
            "calibration_error": [0.05, 0.10],
        }
    )

    scorecard = build_scorecard(raw)

    assert scorecard.iloc[0]["model"] == "better"
    assert scorecard.iloc[0]["tai_score"] > scorecard.iloc[1]["tai_score"]
