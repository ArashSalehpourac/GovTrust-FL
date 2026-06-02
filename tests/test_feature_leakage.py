import pandas as pd

from src.features import infer_feature_columns


def test_leakage_columns_are_not_model_features():
    frame = pd.DataFrame(
        {
            "request_id": ["r1"],
            "created_date": [pd.Timestamp("2024-01-01")],
            "closed_date": [pd.Timestamp("2024-01-02")],
            "resolution_hours": [24.0],
            "delay_threshold_hours": [12.0],
            "status": ["Closed"],
            "category": ["noise"],
            "delayed": [1],
            "safe_feature": [3.0],
        }
    )

    features = infer_feature_columns(frame)

    assert "safe_feature" in features
    assert "closed_date" not in features
    assert "resolution_hours" not in features
    assert "delay_threshold_hours" not in features
    assert "status" not in features
