"""Schema harmonization, cleaning, and delayed-resolution labels."""

import re

import numpy as np
import pandas as pd

from .config import (
    CLOSED_AT_COLUMN,
    DELAY_THRESHOLD_DAYS,
    OPENED_AT_COLUMN,
    RESOLUTION_DAYS_COLUMN,
    TARGET_COLUMN,
)


COLUMN_ALIASES = {
    "created_date": OPENED_AT_COLUMN,
    "creation_date": OPENED_AT_COLUMN,
    "open_date": OPENED_AT_COLUMN,
    "requested_datetime": OPENED_AT_COLUMN,
    "closed_date": CLOSED_AT_COLUMN,
    "closure_date": CLOSED_AT_COLUMN,
    "resolved_at": CLOSED_AT_COLUMN,
    "completion_date": CLOSED_AT_COLUMN,
    "service_request_type": "request_type",
    "complaint_type": "request_type",
    "type_of_service_request": "request_type",
    "zip": "zip_code",
    "zipcode": "zip_code",
    "postal_code": "zip_code",
}


def snake_case_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized snake_case columns."""

    renamed = {
        column: re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", str(column)).strip("_").lower())
        for column in frame.columns
    }
    return frame.rename(columns=renamed)


def harmonize_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Map common city-specific field names to canonical names."""

    normalized = snake_case_columns(frame)
    return normalized.rename(
        columns={column: COLUMN_ALIASES.get(column, column) for column in normalized.columns}
    )


def coerce_datetime_columns(
    frame: pd.DataFrame,
    columns: tuple[str, ...] = (OPENED_AT_COLUMN, CLOSED_AT_COLUMN),
) -> pd.DataFrame:
    """Convert known date columns to pandas datetimes when present."""

    output = frame.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_datetime(output[column], errors="coerce", utc=True)
    return output


def add_resolution_days(frame: pd.DataFrame) -> pd.DataFrame:
    """Add resolution duration in days from opened and closed timestamps."""

    if OPENED_AT_COLUMN not in frame.columns or CLOSED_AT_COLUMN not in frame.columns:
        missing = {OPENED_AT_COLUMN, CLOSED_AT_COLUMN}.difference(frame.columns)
        raise KeyError(f"Cannot compute resolution days; missing columns: {sorted(missing)}")

    output = frame.copy()
    duration = output[CLOSED_AT_COLUMN] - output[OPENED_AT_COLUMN]
    output[RESOLUTION_DAYS_COLUMN] = duration.dt.total_seconds() / 86_400
    output.loc[output[RESOLUTION_DAYS_COLUMN] < 0, RESOLUTION_DAYS_COLUMN] = np.nan
    return output


def make_delayed_resolution_label(
    frame: pd.DataFrame,
    threshold_days: int = DELAY_THRESHOLD_DAYS,
) -> pd.DataFrame:
    """Add a binary delayed-resolution label."""

    if RESOLUTION_DAYS_COLUMN not in frame.columns:
        frame = add_resolution_days(frame)

    output = frame.copy()
    valid_duration = output[RESOLUTION_DAYS_COLUMN].notna()
    output[TARGET_COLUMN] = pd.Series(pd.NA, index=output.index, dtype="Int8")
    output.loc[valid_duration, TARGET_COLUMN] = (
        output.loc[valid_duration, RESOLUTION_DAYS_COLUMN] > threshold_days
    ).astype("int8")
    return output


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add simple calendar features from the opened timestamp."""

    output = frame.copy()
    if OPENED_AT_COLUMN not in output.columns:
        return output

    opened = output[OPENED_AT_COLUMN]
    output["opened_year"] = opened.dt.year
    output["opened_month"] = opened.dt.month
    output["opened_dayofweek"] = opened.dt.dayofweek
    output["opened_hour"] = opened.dt.hour
    return output


def prepare_city_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Run the initial preprocessing pipeline for one city."""

    output = harmonize_schema(frame)
    output = coerce_datetime_columns(output)
    output = add_resolution_days(output)
    output = make_delayed_resolution_label(output)
    output = add_calendar_features(output)
    return output
