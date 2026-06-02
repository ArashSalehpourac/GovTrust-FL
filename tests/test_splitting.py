import pandas as pd

from src.splitting import time_split_frame


def test_temporal_split_order():
    frame = pd.DataFrame(
        {
            "request_id": [f"r{i}" for i in range(10)],
            "created_date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "delayed": [0, 1] * 5,
        }
    )

    splits = time_split_frame(frame)

    assert len(splits["train"]) == 7
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 2
    assert splits["train"]["created_date"].max() < splits["val"]["created_date"].min()
    assert splits["val"]["created_date"].max() < splits["test"]["created_date"].min()


def test_no_los_angeles_in_training_city_splits():
    frame = pd.DataFrame(
        {
            "request_id": [f"r{i}" for i in range(6)],
            "created_date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "city": ["nyc"] * 6,
            "delayed": [0, 1, 0, 1, 0, 1],
        }
    )

    splits = time_split_frame(frame)

    for split in splits.values():
        assert "los_angeles" not in set(split["city"])
