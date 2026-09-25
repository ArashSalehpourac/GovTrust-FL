from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.t2.harmonize_frozen import (
    HARMONIZED_COLUMNS,
    _validate_header,
    harmonize_chunk,
    harmonize_snapshot,
)


def test_nyc_harmonization_is_row_preserving() -> None:
    raw = pd.DataFrame(
        {
            "unique_key": ["1", "2"],
            "created_date": ["2025-01-01 00:00:12", "2025-01-01 01:00:00"],
            "closed_date": ["2025-01-02", "2025-01-03"],
            "status": ["Closed", "Open"],
            "complaint_type": ["Noise", "Heat"],
            "descriptor": ["Loud", "Cold"],
            "agency": ["NYPD", "HPD"],
            "latitude": ["40.1", "40.2"],
            "longitude": ["-73.9", "-73.8"],
            "borough": ["MANHATTAN", "BRONX"],
        }
    )
    out = harmonize_chunk(raw, "nyc")
    assert list(out.columns) == HARMONIZED_COLUMNS
    assert len(out) == len(raw)
    assert out["request_id"].tolist() == ["1", "2"]
    assert out["area"].tolist() == ["MANHATTAN", "BRONX"]


def test_boston_rejects_datastore_only_id() -> None:
    columns = {
        "case_enquiry_id": ["1"],
        "open_dt": ["2025-01-01"],
        "closed_dt": ["2025-01-02"],
        "case_status": ["Closed"],
        "reason": ["Reason"],
        "type": ["Type"],
        "subject": ["Subject"],
        "latitude": ["42.3"],
        "longitude": ["-71.0"],
        "neighborhood": ["Back Bay"],
    }
    for i in range(20):
        columns[f"extra_{i}"] = ["x"]
    raw = pd.DataFrame(columns)
    raw["_id"] = ["1"]
    with pytest.raises(ValueError, match="datastore-only _id"):
        harmonize_chunk(raw, "boston")


def test_la_2025_display_schema_maps_case_number_without_filtering() -> None:
    raw = pd.DataFrame(
        {
            "CaseNumber": ["A", "B"],
            "CreatedDate": ["2025-03-28 06:10:07", "2025-12-31 12:59:54"],
            "ClosedDate": ["2025-03-29", pd.NA],
            "Status": ["Closed", "Open"],
            "RequestType": ["Bulky Items", "Graffiti"],
            "ActionTaken": ["Completed", "Assigned"],
            "Owner": ["LASAN", "LASAN"],
            "Latitude": ["34.0", "34.1"],
            "Longitude": ["-118.2", "-118.3"],
            "NCName": ["Downtown", "Hollywood"],
        }
    )
    out = harmonize_chunk(raw, "los_angeles")
    assert len(out) == 2
    assert out["request_id"].tolist() == ["A", "B"]
    assert out["category"].tolist() == ["Bulky Items", "Graffiti"]
    assert out["closed_date"].isna().sum() == 1


def test_chicago_owner_department_and_area_mapping() -> None:
    raw = pd.DataFrame(
        {
            "sr_number": ["SR1"],
            "created_date": ["2024-01-01 00:00:12"],
            "closed_date": ["2024-01-02"],
            "status": ["Completed"],
            "sr_type": ["Pothole"],
            "sr_short_code": ["PH"],
            "owner_department": ["CDOT"],
            "latitude": ["41.8"],
            "longitude": ["-87.6"],
            "community_area": ["32"],
            "ward": ["42"],
        }
    )
    out = harmonize_chunk(raw, "chicago")
    assert out.loc[0, "agency"] == "CDOT"
    assert out.loc[0, "area"] == "32"


def test_exact_frozen_header_rejects_unused_column_drift() -> None:
    entry = {
        "filename": "example.csv",
        "column_count": 3,
        "expected_columns": ["id", "created", "unused"],
    }
    _validate_header(["id", "created", "unused"], entry)
    with pytest.raises(RuntimeError, match="exact frozen header mismatch"):
        _validate_header(["id", "created", "different_unused"], entry)


def test_boston_snapshot_accepts_mixed_timestamp_formats(tmp_path) -> None:
    columns = {
        "case_enquiry_id": ["1", "2"],
        "open_dt": ["2021-01-01 00:06:37.397", "2021-12-31 23:44:48"],
        "closed_dt": ["2021-01-01 01:06:37.397", "2022-01-01 00:44:48"],
        "case_status": ["Closed", "Closed"],
        "reason": ["Reason A", "Reason B"],
        "type": ["Type A", "Type B"],
        "subject": ["Subject A", "Subject B"],
        "latitude": ["42.3", "42.4"],
        "longitude": ["-71.0", "-71.1"],
        "neighborhood": ["Back Bay", "Roxbury"],
    }
    for i in range(20):
        columns[f"extra_{i}"] = ["x", "y"]

    raw = pd.DataFrame(columns)
    raw_path = tmp_path / "BOSTON_2021_raw.csv"
    raw.to_csv(raw_path, index=False)
    raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    entry = {
        "city": "boston",
        "year": 2021,
        "filename": raw_path.name,
        "size_bytes": raw_path.stat().st_size,
        "sha256": raw_sha,
        "column_count": 30,
        "rows": 2,
        "min_created": "2021-01-01 00:06:37.397",
        "max_created": "2021-12-31 23:44:48",
        "null_ids": 0,
        "duplicate_ids": 0,
        "expected_columns": list(raw.columns),
    }
    output_path = tmp_path / "BOSTON_2021_harmonized.parquet"
    result = harmonize_snapshot(
        raw_path=raw_path,
        output_path=output_path,
        entry=entry,
        chunksize=100,
    )
    assert output_path.is_file()
    assert result["rows"] == 2
    assert result["created_date_min"] == "2021-01-01T00:06:37.397000+00:00"
    assert result["created_date_max"] == "2021-12-31T23:44:48+00:00"


def test_frozen_manifest_keeps_la_2021_schema_distinct() -> None:
    manifest_path = Path("configs/t2/frozen_raw_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    snapshots = {
        (row["city"], row["year"]): row
        for row in manifest["snapshots"]
    }

    la_2021 = snapshots[("los_angeles", 2021)]
    assert la_2021["column_count"] == 33
    assert la_2021["schema_key"] == "la_2021_v1"

    schema_2021 = manifest["schemas"]["la_2021_v1"]
    assert len(schema_2021) == 33
    assert "CreatedByUserOrganization" not in schema_2021

    legacy_schema = manifest["schemas"]["la_legacy_v1"]
    assert len(legacy_schema) == 34
    assert "CreatedByUserOrganization" in legacy_schema

    for year in (2022, 2023, 2024):
        row = snapshots[("los_angeles", year)]
        assert row["column_count"] == 34
        assert row["schema_key"] == "la_legacy_v1"
