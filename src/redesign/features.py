"""Creation-time feature construction for the redesigned pipeline.

All features here are row-local or backward-looking within a single city, so
they are computed by transformation only. Nothing in this module fits a
statistic across rows or across cities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RESOLUTION_HOURS_COLUMN

KEEP_COLUMNS = (
    "request_id",
    "created_date",
    "city",
    RESOLUTION_HOURS_COLUMN,
    *CATEGORICAL_FEATURES,
    *NUMERIC_FEATURES,
)


def build_features(frame: pd.DataFrame, grid_precision: int = 2) -> pd.DataFrame:
    """Return creation-time features for one city, ordered chronologically."""

    working = frame.copy()
    working["created_date"] = pd.to_datetime(working["created_date"], errors="coerce")
    working = (
        working.dropna(subset=["created_date"])
        .sort_values(["created_date", "request_id"], kind="stable")
        .reset_index(drop=True)
    )

    output = working[
        ["request_id", "created_date", "city", RESOLUTION_HOURS_COLUMN, *CATEGORICAL_FEATURES]
    ].copy()
    output["hour"] = working["created_date"].dt.hour.astype("int16")
    output["day_of_week"] = working["created_date"].dt.dayofweek.astype("int16")
    output["month"] = working["created_date"].dt.month.astype("int16")
    output["is_weekend"] = output["day_of_week"].isin((5, 6)).astype("int16")

    scale = 10**grid_precision
    output["latitude_grid"] = np.floor(
        pd.to_numeric(working["latitude"], errors="coerce") * scale
    ) / scale
    output["longitude_grid"] = np.floor(
        pd.to_numeric(working["longitude"], errors="coerce") * scale
    ) / scale

    output["area_request_count_7d"] = trailing_request_count(working, "area")
    output["category_request_count_7d"] = trailing_request_count(working, "category")
    output["agency_request_count_7d"] = trailing_request_count(working, "agency")

    return output[list(KEEP_COLUMNS)]


def trailing_request_count(frame: pd.DataFrame, group_column: str, window: str = "7D") -> pd.Series:
    """Count prior same-group requests in a trailing window, excluding the current row."""

    working = frame[["created_date", group_column]].copy()
    working["_position"] = np.arange(len(working))
    counts = np.zeros(len(working), dtype="int32")

    for _, group in working.groupby(group_column, dropna=False, sort=False, observed=True):
        group = group.sort_values(["created_date", "_position"], kind="stable")
        marker = pd.Series(1, index=pd.DatetimeIndex(group["created_date"]), dtype="int32")
        rolling = marker.rolling(window, closed="left").count().fillna(0).to_numpy()
        counts[group["_position"].to_numpy()] = rolling.astype("int32")

    return pd.Series(counts, index=frame.index, dtype="int32")
