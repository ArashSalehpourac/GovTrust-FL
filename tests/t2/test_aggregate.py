import pytest

from src.t2.aggregate import privacy_transfer_penalty


def test_ptp_definition_exact():
    private = {
        "external": {"mae_log1p_hours": 2.4},
        "source_macro_mae_log1p_hours": 1.3,
    }
    baseline = {
        "external": {"mae_log1p_hours": 2.0},
        "source_macro_mae_log1p_hours": 1.1,
    }
    out = privacy_transfer_penalty(private, baseline)
    assert out["external_privacy_cost"] == pytest.approx(0.4)
    assert out["internal_privacy_cost"] == pytest.approx(0.2)
    assert out["ptp"] == pytest.approx(0.2)


def test_ptp_rejects_mismatched_external_windows():
    private = {
        "external": {"mae_log1p_hours": 2.4},
        "source_macro_mae_log1p_hours": 1.3,
        "external_scope": "held-out city 2025 eligible completion/closure rows",
    }
    baseline = {
        "external": {"mae_log1p_hours": 2.0},
        "source_macro_mae_log1p_hours": 1.1,
        "external_scope": "held-out city 2021-2025 eligible completion/closure rows",
    }
    with pytest.raises(ValueError, match="same external evaluation scope"):
        privacy_transfer_penalty(private, baseline)
