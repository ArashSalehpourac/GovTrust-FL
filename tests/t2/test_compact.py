import numpy as np
import pandas as pd

from src.t2.compact import CompactRegressionDataset
from src.t2.data import prepare_resolved_frame
from src.t2.preprocessing import build_fixed_preprocessor


def _frame() -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "request_id": ["a", "b", "c", "d"],
            "created_date": [
                "2025-01-01T01:00:00Z",
                "2025-01-02T02:00:00Z",
                "2025-01-03T03:00:00Z",
                "2025-01-04T04:00:00Z",
            ],
            "closed_date": [
                "2025-01-01T02:00:00Z",
                "2025-01-02T04:00:00Z",
                "2025-01-03T06:00:00Z",
                "2025-01-04T08:00:00Z",
            ],
            "status": ["closed"] * 4,
            "category": ["Tree", "Pothole", "Tree", "Noise"],
        }
    )
    return prepare_resolved_frame(raw)


def test_compact_dataset_is_feature_exact_without_dense_row_storage():
    frame = _frame()
    pre = build_fixed_preprocessor(text_hash_dim=32, category_hash_dim=8)
    expected = pre.transform(frame)

    dataset = CompactRegressionDataset.from_frame(frame, pre)
    observed = np.stack([dataset[i][0].numpy() for i in range(len(dataset))])

    assert observed.shape == expected.shape
    assert np.allclose(observed, expected)
    assert dataset.dense_rows_materialized == 0
    assert dataset.resident_bytes < expected.nbytes


def test_split_transform_helpers_match_full_transform():
    frame = _frame()
    pre = build_fixed_preprocessor(text_hash_dim=32, category_hash_dim=8)
    combined = np.concatenate(
        [
            pre.transform_category_values(frame["category"]),
            pre.transform_numeric(frame),
        ],
        axis=1,
    )
    assert np.allclose(combined, pre.transform(frame))
