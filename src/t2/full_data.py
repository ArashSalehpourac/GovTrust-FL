from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .compact import CompactRegressionDataset, ShardedCompactRegressionDataset
from .data import frame_summary, prepare_resolved_frame
from .folds import INTERNAL_TEST_START, VALIDATION_START
from .preprocessing import FixedPreprocessor

YEARS = (2021, 2022, 2023, 2024, 2025)
CITY_FILENAME_PREFIX = {
    "nyc": "NYC",
    "chicago": "CHICAGO",
    "boston": "BOSTON",
    "los_angeles": "LOS_ANGELES",
}
READ_COLUMNS = [
    "request_id",
    "created_date",
    "closed_date",
    "status",
    "category",
    "city",
]


@dataclass(frozen=True)
class CompactCityBundle:
    splits: dict[str, CompactRegressionDataset | ShardedCompactRegressionDataset]
    summaries: dict[str, dict[str, object]]
    input_sha256_by_year: dict[str, str]
    coverage_notes: list[str]


def load_harmonized_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 20:
        raise RuntimeError("harmonized manifest must contain exactly 20 outputs")
    if manifest.get("row_filtering") != "none":
        raise RuntimeError("full-data contract requires row_filtering=none")
    return manifest


def _entries_by_key(
    manifest: dict[str, object],
) -> dict[tuple[str, int], dict[str, object]]:
    outputs = manifest["outputs"]
    if not isinstance(outputs, list):
        raise TypeError("harmonized outputs must be a list")
    out: dict[tuple[str, int], dict[str, object]] = {}
    for row in outputs:
        if not isinstance(row, dict):
            raise TypeError("harmonized output entry must be an object")
        key = (str(row["city"]), int(row["year"]))
        if key in out:
            raise RuntimeError(f"duplicate harmonized entry: {key}")
        out[key] = row
    return out


def harmonized_path(harmonized_dir: Path, city: str, year: int) -> Path:
    if city not in CITY_FILENAME_PREFIX:
        raise ValueError(f"unknown city: {city}")
    return harmonized_dir / (
        f"{CITY_FILENAME_PREFIX[city]}_{year}_harmonized.parquet"
    )


def _load_prepared_year(path: Path, city: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=READ_COLUMNS)
    observed = set(frame["city"].dropna().astype(str).str.strip().unique())
    if observed != {city}:
        raise RuntimeError(
            f"{path.name}: unexpected city values {sorted(observed)}"
        )
    return prepare_resolved_frame(frame)


def _compact(
    frame: pd.DataFrame,
    preprocessor: FixedPreprocessor,
) -> CompactRegressionDataset:
    return CompactRegressionDataset.from_frame(frame, preprocessor)


def load_source_city_bundle(
    *,
    harmonized_dir: Path,
    harmonized_manifest: dict[str, object],
    city: str,
    preprocessor: FixedPreprocessor,
) -> CompactCityBundle:
    entries = _entries_by_key(harmonized_manifest)
    train_shards: list[CompactRegressionDataset] = []
    summaries: dict[str, dict[str, object]] = {}
    sha_by_year: dict[str, str] = {}
    coverage_notes: list[str] = []
    train_summaries: list[dict[str, object]] = []
    train_rows = 0

    val: CompactRegressionDataset | None = None
    internal_test: CompactRegressionDataset | None = None

    for year in YEARS:
        key = (city, year)
        if key not in entries:
            raise RuntimeError(f"missing harmonized manifest entry: {key}")
        entry = entries[key]
        path = harmonized_path(harmonized_dir, city, year)
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.name != str(entry["harmonized_filename"]):
            raise RuntimeError(
                f"{city} {year}: filename differs from harmonized manifest"
            )
        if path.stat().st_size != int(entry["harmonized_size_bytes"]):
            raise RuntimeError(
                f"{city} {year}: size differs from harmonized manifest"
            )

        sha_by_year[str(year)] = str(entry["harmonized_sha256"])
        note = entry.get("coverage_note")
        if note:
            coverage_notes.append(str(note))

        prepared = _load_prepared_year(path, city)

        if year <= 2023:
            selected = prepared[
                prepared["closed_date"].lt(VALIDATION_START)
            ].copy()
            train_shards.append(_compact(selected, preprocessor))
            train_rows += len(selected)
            train_summaries.append(frame_summary(selected))
        elif year == 2024:
            selected = prepared[
                prepared["closed_date"].lt(INTERNAL_TEST_START)
            ].copy()
            summaries["val"] = frame_summary(selected)
            val = _compact(selected, preprocessor)
        else:
            selected = prepared
            summaries["internal_test"] = frame_summary(selected)
            internal_test = _compact(selected, preprocessor)

        del selected
        del prepared
        del frame

    if not train_shards:
        raise RuntimeError(f"{city}: no training shards")
    if val is None or internal_test is None:
        raise RuntimeError(f"{city}: incomplete compact source split")

    train = ShardedCompactRegressionDataset(train_shards)
    if len(train) != train_rows:
        raise AssertionError(f"{city}: compact train row count mismatch")

    summaries["train"] = {
        "rows": train_rows,
        "years": [2021, 2022, 2023],
        "year_summaries": train_summaries,
    }

    return CompactCityBundle(
        splits={
            "train": train,
            "val": val,
            "internal_test": internal_test,
        },
        summaries=summaries,
        input_sha256_by_year=sha_by_year,
        coverage_notes=coverage_notes,
    )


def load_external_2025(
    *,
    harmonized_dir: Path,
    harmonized_manifest: dict[str, object],
    city: str,
    preprocessor: FixedPreprocessor,
) -> tuple[
    CompactRegressionDataset,
    dict[str, object],
    dict[str, str],
    list[str],
]:
    entries = _entries_by_key(harmonized_manifest)
    entry = entries[(city, 2025)]
    path = harmonized_path(harmonized_dir, city, 2025)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.name != str(entry["harmonized_filename"]):
        raise RuntimeError(
            f"{city} 2025: filename differs from harmonized manifest"
        )
    if path.stat().st_size != int(entry["harmonized_size_bytes"]):
        raise RuntimeError(
            f"{city} 2025: size differs from harmonized manifest"
        )

    prepared = _load_prepared_year(path, city)
    summary = frame_summary(prepared)
    dataset = _compact(prepared, preprocessor)
    notes = (
        [str(entry["coverage_note"])]
        if entry.get("coverage_note")
        else []
    )
    return (
        dataset,
        summary,
        {"2025": str(entry["harmonized_sha256"])},
        notes,
    )
