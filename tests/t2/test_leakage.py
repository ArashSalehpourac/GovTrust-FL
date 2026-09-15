import numpy as np
import pandas as pd

from src.t2.preprocessing import (
    CAT_COLS,
    FORBIDDEN_PRIMARY,
    NUM_COLS,
    TEXT_COL,
    fit_source_train,
)


def _frame(prefix: str, n: int = 20) -> pd.DataFrame:
    return pd.DataFrame({
        "descriptor": [f"{prefix} token {i % 3}" for i in range(n)],
        "category": [f"cat{i % 2}" for i in range(n)],
        "hour": np.arange(n) % 24,
        "day_of_week": np.arange(n) % 7,
        "month": 1,
        "is_weekend": 0,
    })


def test_preprocessor_fits_train_only_not_val_or_test():
    base = {
        "a": {"train": _frame("train-a"), "val": _frame("val-a"), "internal_test": _frame("test-a")},
        "b": {"train": _frame("train-b"), "val": _frame("val-b"), "internal_test": _frame("test-b")},
        "c": {"train": _frame("train-c"), "val": _frame("val-c"), "internal_test": _frame("test-c")},
    }
    p1 = fit_source_train(base, max_text_features=50)
    changed = {city: {name: frame.copy() for name, frame in splits.items()} for city, splits in base.items()}
    for splits in changed.values():
        splits["val"]["descriptor"] = "TARGETLIKE SECRET FUTURE TOKEN"
        splits["val"]["category"] = "FUTURE_ONLY"
        splits["internal_test"]["descriptor"] = "ANOTHER FUTURE TOKEN"
        splits["internal_test"]["category"] = "FUTURE_ONLY_2"
    p2 = fit_source_train(changed, max_text_features=50)
    assert p1.fingerprint == p2.fingerprint


def test_primary_features_exclude_city_shortcuts():
    selected = set(CAT_COLS + NUM_COLS + [TEXT_COL])
    assert selected.isdisjoint(FORBIDDEN_PRIMARY)
