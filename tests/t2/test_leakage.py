import numpy as np
import pandas as pd

from src.t2.preprocessing import (
    CAT_COLS,
    FORBIDDEN_PRIMARY,
    NUM_COLS,
    TEXT_COL,
    build_fixed_preprocessor,
)


def _frame(prefix: str, n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "descriptor": [f"{prefix} token {i % 3}" for i in range(n)],
            "category": [f"cat{i % 2}" for i in range(n)],
            "hour": np.arange(n) % 24,
            "day_of_week": np.arange(n) % 7,
            "month": 1,
            "is_weekend": 0,
        }
    )


def test_preprocessor_is_data_independent_and_deterministic():
    first = build_fixed_preprocessor(text_hash_dim=32, category_hash_dim=8)
    second = build_fixed_preprocessor(text_hash_dim=32, category_hash_dim=8)
    frame = _frame("private-token")
    assert first.fingerprint == second.fingerprint
    assert first.output_dim == 47
    assert np.array_equal(first.transform(frame), second.transform(frame))


def test_preprocessor_constructor_accepts_no_training_data():
    pre = build_fixed_preprocessor(text_hash_dim=16, category_hash_dim=4)
    changed = _frame("completely-different-records")
    assert pre.transform(changed).shape == (len(changed), pre.output_dim)


def test_primary_features_exclude_city_shortcuts():
    selected = set(CAT_COLS + NUM_COLS + [TEXT_COL])
    assert selected.isdisjoint(FORBIDDEN_PRIMARY)
