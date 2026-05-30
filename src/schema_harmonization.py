"""Schema harmonization for city 311 datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


TARGET_SCHEMA = [
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


@dataclass(frozen=True)
class HarmonizationSpec:
    """Column mapping from a city raw file to the target schema."""

    city: str
    raw_path: Path
    output_path: Path
    mapping: dict[str, str | Callable[[pd.DataFrame], pd.Series]]
    usecols: list[str]
    reader: Callable[["HarmonizationSpec"], pd.DataFrame] | None = None


def first_nonempty(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Return the first non-empty value across columns for each row."""

    result = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].astype("string").str.strip()
        values = values.mask(values.eq(""))
        result = result.fillna(values)
    return result


def harmonize_frame(raw: pd.DataFrame, spec: HarmonizationSpec) -> pd.DataFrame:
    """Convert a raw city frame into the shared target schema."""

    output = pd.DataFrame(index=raw.index)
    for target_column in TARGET_SCHEMA:
        if target_column == "city":
            output[target_column] = spec.city
            continue

        source = spec.mapping[target_column]
        if callable(source):
            output[target_column] = source(raw)
        else:
            output[target_column] = raw[source] if source in raw.columns else pd.NA

    for column in (
        "request_id",
        "status",
        "category",
        "descriptor",
        "agency",
        "area",
        "city",
    ):
        output[column] = output[column].astype("string").str.strip()
        output[column] = output[column].mask(output[column].eq(""))

    for column in ("created_date", "closed_date"):
        output[column] = pd.to_datetime(output[column], errors="coerce")

    for column in ("latitude", "longitude"):
        output[column] = pd.to_numeric(output[column], errors="coerce")

    return output[TARGET_SCHEMA]


def harmonize_file(spec: HarmonizationSpec) -> dict[str, object]:
    """Read one raw CSV, harmonize it, and write Parquet."""

    if spec.reader is None:
        raw = pd.read_csv(spec.raw_path, usecols=spec.usecols, dtype="string", low_memory=False)
    else:
        raw = spec.reader(spec)
    harmonized = harmonize_frame(raw, spec)
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    harmonized.to_parquet(spec.output_path, index=False)

    return {
        "city": spec.city,
        "input_path": str(spec.raw_path),
        "output_path": str(spec.output_path),
        "rows": int(len(harmonized)),
        "columns": list(harmonized.columns),
        "created_date_min": _iso_or_none(harmonized["created_date"].min()),
        "created_date_max": _iso_or_none(harmonized["created_date"].max()),
        "closed_date_missing": int(harmonized["closed_date"].isna().sum()),
        "latitude_missing": int(harmonized["latitude"].isna().sum()),
        "longitude_missing": int(harmonized["longitude"].isna().sum()),
    }


def _iso_or_none(value) -> str | None:
    if pd.isna(value):
        return None
    return value.isoformat()


def read_los_angeles_raw(spec: HarmonizationSpec) -> pd.DataFrame:
    """Read the combined LA raw file across yearly schema drift.

    The 2021 MyLA311 dataset lacks `createdbyuserorganization`, while 2022+
    include it after `requestsource`. Step 1 intentionally saved a single raw
    CSV without editing the bytes, so this reader aligns rows by the year-level
    raw schema before selecting the harmonized columns.
    """

    rows: list[dict[str, str | None]] = []
    with spec.raw_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.reader(input_file)
        base_header = next(reader)
        if "requestsource" not in base_header:
            raise ValueError("Unexpected LA raw schema: missing requestsource")

        insert_at = base_header.index("requestsource") + 1
        expanded_header = [
            *base_header[:insert_at],
            "createdbyuserorganization",
            *base_header[insert_at:],
        ]

        for row in reader:
            if not row:
                continue
            if len(row) == len(base_header):
                header = base_header
            elif len(row) == len(expanded_header):
                header = expanded_header
            else:
                raise ValueError(
                    "Unexpected LA row width: "
                    f"got {len(row)}, expected {len(base_header)} or {len(expanded_header)}"
                )

            values = dict(zip(header, row, strict=True))
            rows.append({column: values.get(column) for column in spec.usecols})

    return pd.DataFrame(rows, columns=spec.usecols, dtype="string")
