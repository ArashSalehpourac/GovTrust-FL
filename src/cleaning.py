"""Shared cleaning rules for harmonized city datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


HARMONIZED_SCHEMA = [
    "request_id",
    "created_date",
    "closed_date",
    "status",
    "category",
    "descriptor",
    "agency",
    "latitude",
    "longitude",
    "area",
    "city",
]

CLEANED_SCHEMA = [
    *HARMONIZED_SCHEMA,
    "resolution_hours",
]

FINAL_STATUSES = {"closed", "completed"}
BAD_STATUS_PATTERN = r"cancelled|canceled|invalid|incomplete"


@dataclass(frozen=True)
class CleaningSpec:
    """Input/output paths for one harmonized city dataset."""

    city: str
    input_path: Path
    output_path: Path


def clean_frame(frame: pd.DataFrame, min_category_samples: int = 100) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply common Step 3 cleaning rules and return row-removal counts."""

    working = frame.copy()
    start_rows = len(working)
    counts: dict[str, int] = {"input_rows": start_rows}

    working["created_date"] = pd.to_datetime(working["created_date"], errors="coerce")
    working["closed_date"] = pd.to_datetime(working["closed_date"], errors="coerce")
    working["latitude"] = pd.to_numeric(working["latitude"], errors="coerce")
    working["longitude"] = pd.to_numeric(working["longitude"], errors="coerce")

    working, counts["missing_created_date"] = _filter_and_count(
        working,
        working["created_date"].notna(),
    )
    working, counts["missing_closed_date"] = _filter_and_count(
        working,
        working["closed_date"].notna(),
    )
    working, counts["closed_before_created"] = _filter_and_count(
        working,
        working["closed_date"].ge(working["created_date"]),
    )

    before = len(working)
    working = working.sort_values(["request_id", "created_date"], kind="stable")
    working = working.drop_duplicates(subset=["request_id"], keep="first")
    counts["duplicate_request_id"] = before - len(working)

    status_normalized = working["status"].astype("string").str.strip().str.lower()
    has_final_status = status_normalized.isin(FINAL_STATUSES)
    has_bad_status = status_normalized.str.contains(BAD_STATUS_PATTERN, regex=True, na=False)
    working, counts["cancelled_invalid_incomplete_status"] = _filter_and_count(
        working,
        has_final_status & ~has_bad_status,
    )

    valid_coordinates = (
        working["latitude"].between(-90, 90)
        & working["longitude"].between(-180, 180)
        & ~(working["latitude"].eq(0) & working["longitude"].eq(0))
    )
    working, counts["impossible_coordinates"] = _filter_and_count(working, valid_coordinates)

    working["resolution_hours"] = (
        working["closed_date"] - working["created_date"]
    ).dt.total_seconds() / 3600

    outlier_thresholds = working.groupby("city", dropna=False, observed=True)["resolution_hours"].transform(
        lambda values: values.quantile(0.99) if len(values) >= 100 else values.max()
    )
    working, counts["resolution_hours_above_city_p99"] = _filter_and_count(
        working,
        working["resolution_hours"].le(outlier_thresholds),
    )

    category_counts = working["category"].value_counts(dropna=False)
    valid_categories = working["category"].map(category_counts).ge(min_category_samples)
    working, counts["category_below_min_samples"] = _filter_and_count(working, valid_categories)

    working = working.sort_values(["created_date", "request_id"], kind="stable")
    working = working.reset_index(drop=True)
    working["resolution_hours"] = working["resolution_hours"].astype("float64")

    counts["output_rows"] = len(working)
    counts["total_removed"] = start_rows - len(working)
    return working[CLEANED_SCHEMA], counts


def clean_file(spec: CleaningSpec, min_category_samples: int = 100) -> dict[str, object]:
    """Clean one harmonized Parquet file and write a cleaned Parquet file."""

    frame = pd.read_parquet(spec.input_path)
    cleaned, counts = clean_frame(frame, min_category_samples=min_category_samples)
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(spec.output_path, index=False)

    return {
        "city": spec.city,
        "input_path": str(spec.input_path),
        "output_path": str(spec.output_path),
        "counts": counts,
        "columns": list(cleaned.columns),
        "created_date_min": _iso_or_none(cleaned["created_date"].min()),
        "created_date_max": _iso_or_none(cleaned["created_date"].max()),
        "date_range": {
            "created_date_min": _iso_or_none(cleaned["created_date"].min()),
            "created_date_max": _iso_or_none(cleaned["created_date"].max()),
        },
        "resolution_hours_min": _float_or_none(cleaned["resolution_hours"].min()),
        "resolution_hours_max": _float_or_none(cleaned["resolution_hours"].max()),
        "categories": int(cleaned["category"].nunique(dropna=True)),
    }


def _filter_and_count(frame: pd.DataFrame, keep_mask: pd.Series) -> tuple[pd.DataFrame, int]:
    keep_mask = keep_mask.fillna(False)
    removed = int((~keep_mask).sum())
    return frame.loc[keep_mask].copy(), removed


def _iso_or_none(value) -> str | None:
    if pd.isna(value):
        return None
    return value.isoformat()


def _float_or_none(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
