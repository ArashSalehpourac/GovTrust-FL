import numpy as np
import pandas as pd

from src.t2.data import PRIMARY_TARGET, prepare_resolved_frame


def test_primary_target_is_direct_log1p_resolution_hours_and_open_rows_drop():
    frame = pd.DataFrame(
        {
            "request_id": ["a", "b", "c"],
            "created_date": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ],
            "closed_date": [
                "2026-01-01T01:00:00Z",
                None,
                "2026-01-03T00:00:00Z",
            ],
            "category": ["a", "a", "b"],
            "descriptor": ["x", "y", "z"],
            "status": ["closed", "open", "completed"],
        }
    )
    out = prepare_resolved_frame(frame)
    assert len(out) == 2
    assert np.allclose(out[PRIMARY_TARGET].to_numpy(), np.log1p([1.0, 48.0]))
    assert "resolution_hours" in out


def test_duplicate_request_ids_are_deduplicated_before_split_use():
    frame = pd.DataFrame(
        {
            "request_id": ["dup", "dup", "unique"],
            "created_date": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
                "2026-01-02T00:00:00Z",
            ],
            "closed_date": [
                "2026-01-01T01:00:00Z",
                "2026-01-01T05:00:00Z",
                "2026-01-02T02:00:00Z",
            ],
            "category": ["a", "a", "b"],
            "descriptor": ["x", "x", "y"],
            "status": ["closed", "closed", "closed"],
        }
    )
    out = prepare_resolved_frame(frame)
    assert list(out["request_id"].astype(str)) == ["dup", "unique"]
    assert np.isclose(out.loc[0, "resolution_hours"], 1.0)


def test_prepare_resolved_frame_preserves_mixed_timestamp_formats():
    frame = pd.DataFrame(
        {
            "request_id": ["a", "b"],
            "created_date": [
                "2021-01-01 00:06:37.397",
                "2021-01-02 01:44:00",
            ],
            "closed_date": [
                "2021-01-01 01:06:37.397",
                "2021-01-02 03:44:00",
            ],
            "category": ["a", "b"],
            "descriptor": ["x", "y"],
            "status": ["closed", "closed"],
        }
    )
    out = prepare_resolved_frame(frame)
    assert len(out) == 2
    assert np.allclose(out["resolution_hours"].to_numpy(), [1.0, 2.0])
