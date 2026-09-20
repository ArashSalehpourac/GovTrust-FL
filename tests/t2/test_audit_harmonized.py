from __future__ import annotations

import hashlib

import pandas as pd

from src.t2.audit_harmonized import audit_one
from src.t2.harmonize_frozen import HARMONIZED_COLUMNS


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_audit_one_passes_row_preserving_valid_output(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "request_id": ["A", "B"],
            "created_date": ["2025-03-28 06:10:07", "2025-12-31 12:59:54"],
            "closed_date": ["2025-03-29 06:10:07", pd.NA],
            "status": ["Closed", "Open"],
            "category": ["Bulky Items", "Graffiti"],
            "descriptor": ["Completed", "Assigned"],
            "agency": ["LASAN", "LASAN"],
            "latitude": ["34.0", "34.1"],
            "longitude": ["-118.2", "-118.3"],
            "area": ["Downtown", "Hollywood"],
            "city": ["los_angeles", "los_angeles"],
        }
    )[HARMONIZED_COLUMNS]
    path = tmp_path / "LOS_ANGELES_2025_harmonized.parquet"
    frame.to_parquet(path, index=False)

    raw_entry = {
        "city": "los_angeles",
        "year": 2025,
        "rows": 2,
        "min_created": "2025-03-28 06:10:07",
        "max_created": "2025-12-31 12:59:54",
        "null_ids": 0,
        "duplicate_ids": 0,
    }
    output_entry = {
        "harmonized_size_bytes": path.stat().st_size,
        "harmonized_sha256": _sha256(path),
        "rows": 2,
    }
    report = audit_one(path=path, raw_entry=raw_entry, output_entry=output_entry)
    assert report["gate"] == "PASS"
    assert report["eligible_resolved_rows"] == 1
    assert report["missing_counts"]["closed_date"] == 1


def test_audit_one_blocks_created_year_violation(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "request_id": ["A"],
            "created_date": ["2024-12-31 23:59:59"],
            "closed_date": ["2025-01-01 01:00:00"],
            "status": ["Closed"],
            "category": ["Test"],
            "descriptor": ["Test"],
            "agency": ["Agency"],
            "latitude": ["1"],
            "longitude": ["2"],
            "area": ["3"],
            "city": ["nyc"],
        }
    )[HARMONIZED_COLUMNS]
    path = tmp_path / "NYC_2025_harmonized.parquet"
    frame.to_parquet(path, index=False)
    raw_entry = {
        "city": "nyc",
        "year": 2025,
        "rows": 1,
        "min_created": "2024-12-31 23:59:59",
        "max_created": "2024-12-31 23:59:59",
        "null_ids": 0,
        "duplicate_ids": 0,
    }
    output_entry = {
        "harmonized_size_bytes": path.stat().st_size,
        "harmonized_sha256": _sha256(path),
        "rows": 1,
    }
    report = audit_one(path=path, raw_entry=raw_entry, output_entry=output_entry)
    assert report["gate"] == "FAIL"
    assert any("outside snapshot year" in item for item in report["blockers"])
