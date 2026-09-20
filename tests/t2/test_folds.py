import pandas as pd

from src.t2.folds import (
    INTERNAL_TEST_START,
    VALIDATION_START,
    chronological_split,
)


def _row(request_id: str, created: str, closed: str) -> dict[str, object]:
    return {
        "request_id": request_id,
        "created_date": created,
        "closed_date": closed,
        "status": "closed",
        "category": "c",
        "descriptor": "d",
    }


def test_frozen_year_split_purges_outcome_maturation_overlap():
    frame = pd.DataFrame(
        [
            _row("t21", "2021-01-01", "2021-01-02"),
            _row("t22", "2022-01-01", "2022-01-02"),
            _row("t23-good", "2023-06-01", "2023-06-02"),
            _row("t23-cross", "2023-12-31", "2024-01-02"),
            _row("v24-good", "2024-06-01", "2024-06-02"),
            _row("v24-cross", "2024-12-31", "2025-01-02"),
            _row("x25", "2025-06-01", "2025-06-02"),
        ]
    )

    splits = chronological_split(frame)

    assert set(splits) == {"train", "val", "internal_test"}
    assert list(splits["train"]["request_id"]) == ["t21", "t22", "t23-good"]
    assert list(splits["val"]["request_id"]) == ["v24-good"]
    assert list(splits["internal_test"]["request_id"]) == ["x25"]

    assert splits["train"]["closed_date"].max() < VALIDATION_START
    assert splits["val"]["closed_date"].max() < INTERNAL_TEST_START


def test_frozen_year_split_rejects_unexpected_years():
    frame = pd.DataFrame(
        [
            _row("t21", "2021-01-01", "2021-01-02"),
            _row("v24", "2024-01-01", "2024-01-02"),
            _row("x25", "2025-01-01", "2025-01-02"),
            _row("bad", "2026-01-01", "2026-01-02"),
        ]
    )

    try:
        chronological_split(frame)
    except ValueError as exc:
        assert "unexpected analysis years" in str(exc)
    else:
        raise AssertionError("2026 row should be rejected")
