from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from .config import CITIES

TRAIN_YEARS = (2021, 2022, 2023)
VALIDATION_YEAR = 2024
INTERNAL_TEST_YEAR = 2025
VALIDATION_START = pd.Timestamp("2024-01-01T00:00:00Z")
INTERNAL_TEST_START = pd.Timestamp("2025-01-01T00:00:00Z")


@dataclass(frozen=True)
class FoldSpec:
    held_out_city: str
    source_cities: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"holdout_{self.held_out_city}"


def loco_spec(held_out_city: str) -> FoldSpec:
    if held_out_city not in CITIES:
        raise ValueError(f"unknown city: {held_out_city}")
    return FoldSpec(held_out_city, tuple(city for city in CITIES if city != held_out_city))


def chronological_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the frozen purged source-city chronology split.

    Training uses 2021-2023 requests only when the outcome was known before
    validation starts. Validation uses 2024 requests only when the outcome was
    known before the 2025 internal-test period starts. Eligible 2025 requests
    form the internal test. This prevents outcome-maturation leakage across
    source-city chronological boundaries.
    """

    required = {"created_date", "closed_date", "request_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"chronological split missing columns: {sorted(missing)}")

    ordered = frame.copy()
    ordered["created_date"] = pd.to_datetime(
        ordered["created_date"], errors="coerce", utc=True, format="mixed"
    )
    ordered["closed_date"] = pd.to_datetime(
        ordered["closed_date"], errors="coerce", utc=True, format="mixed"
    )
    if ordered["created_date"].isna().any() or ordered["closed_date"].isna().any():
        raise ValueError("prepared analysis frame must have parseable created/closed dates")

    ordered = ordered.sort_values(["created_date", "request_id"], kind="stable").reset_index(
        drop=True
    )
    years = ordered["created_date"].dt.year
    unexpected = sorted(set(years.unique()) - {*TRAIN_YEARS, VALIDATION_YEAR, INTERNAL_TEST_YEAR})
    if unexpected:
        raise ValueError(f"unexpected analysis years: {unexpected}")

    train_candidate = years.isin(TRAIN_YEARS)
    validation_candidate = years.eq(VALIDATION_YEAR)
    test_candidate = years.eq(INTERNAL_TEST_YEAR)

    train = ordered[
        train_candidate & ordered["closed_date"].lt(VALIDATION_START)
    ].copy()
    val = ordered[
        validation_candidate & ordered["closed_date"].lt(INTERNAL_TEST_START)
    ].copy()
    internal_test = ordered[test_candidate].copy()

    if train.empty or val.empty or internal_test.empty:
        raise ValueError(
            "purged chronological split requires non-empty train, val, and internal-test partitions"
        )
    if train["closed_date"].max() >= VALIDATION_START:
        raise AssertionError("training outcome crosses validation boundary")
    if val["closed_date"].max() >= INTERNAL_TEST_START:
        raise AssertionError("validation outcome crosses internal-test boundary")
    if train["created_date"].max() >= VALIDATION_START:
        raise AssertionError("training creation time crosses validation boundary")
    if val["created_date"].min() < VALIDATION_START:
        raise AssertionError("validation begins before frozen validation boundary")
    if internal_test["created_date"].min() < INTERNAL_TEST_START:
        raise AssertionError("internal test begins before frozen test boundary")

    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "internal_test": internal_test.reset_index(drop=True),
    }


def split_source_cities(
    prepared: Mapping[str, pd.DataFrame], spec: FoldSpec
) -> dict[str, dict[str, pd.DataFrame]]:
    if spec.held_out_city in spec.source_cities:
        raise AssertionError("held-out city cannot be a source")
    missing = set(spec.source_cities) - set(prepared)
    if missing:
        raise ValueError(f"missing source frames: {sorted(missing)}")
    return {city: chronological_split(prepared[city]) for city in spec.source_cities}
