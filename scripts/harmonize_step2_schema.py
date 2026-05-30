"""Run Step 2 schema harmonization for all raw city extracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.schema_harmonization import (  # noqa: E402
    TARGET_SCHEMA,
    HarmonizationSpec,
    first_nonempty,
    harmonize_file,
    read_los_angeles_raw,
)


RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "harmonization_manifest_step2.json"


def build_specs() -> list[HarmonizationSpec]:
    return [
        HarmonizationSpec(
            city="nyc",
            raw_path=RAW_DIR / "nyc" / "nyc_311_2021_2025.csv",
            output_path=PROCESSED_DIR / "nyc_harmonized.parquet",
            usecols=[
                "unique_key",
                "created_date",
                "closed_date",
                "status",
                "complaint_type",
                "descriptor",
                "agency",
                "latitude",
                "longitude",
                "community_board",
                "borough",
            ],
            mapping={
                "request_id": "unique_key",
                "created_date": "created_date",
                "closed_date": "closed_date",
                "status": "status",
                "category": "complaint_type",
                "descriptor": "descriptor",
                "agency": "agency",
                "latitude": "latitude",
                "longitude": "longitude",
                "area": lambda frame: first_nonempty(frame, ["community_board", "borough"]),
            },
        ),
        HarmonizationSpec(
            city="chicago",
            raw_path=RAW_DIR / "chicago" / "chicago_311_2021_2025.csv",
            output_path=PROCESSED_DIR / "chicago_harmonized.parquet",
            usecols=[
                "sr_number",
                "created_date",
                "closed_date",
                "status",
                "sr_type",
                "sr_short_code",
                "owner_department",
                "latitude",
                "longitude",
                "community_area",
                "ward",
            ],
            mapping={
                "request_id": "sr_number",
                "created_date": "created_date",
                "closed_date": "closed_date",
                "status": "status",
                "category": "sr_type",
                "descriptor": "sr_short_code",
                "agency": "owner_department",
                "latitude": "latitude",
                "longitude": "longitude",
                "area": lambda frame: first_nonempty(frame, ["community_area", "ward"]),
            },
        ),
        HarmonizationSpec(
            city="boston",
            raw_path=RAW_DIR / "boston" / "boston_311_2021_2025.csv",
            output_path=PROCESSED_DIR / "boston_harmonized.parquet",
            usecols=[
                "case_enquiry_id",
                "open_dt",
                "closed_dt",
                "case_status",
                "reason",
                "type",
                "subject",
                "department",
                "latitude",
                "longitude",
                "neighborhood",
                "ward",
            ],
            mapping={
                "request_id": "case_enquiry_id",
                "created_date": "open_dt",
                "closed_date": "closed_dt",
                "status": "case_status",
                "category": "reason",
                "descriptor": "type",
                "agency": lambda frame: first_nonempty(frame, ["subject", "department"]),
                "latitude": "latitude",
                "longitude": "longitude",
                "area": lambda frame: first_nonempty(frame, ["neighborhood", "ward"]),
            },
        ),
        HarmonizationSpec(
            city="los_angeles",
            raw_path=RAW_DIR / "los_angeles" / "la_311_2021_2025.csv",
            output_path=PROCESSED_DIR / "la_harmonized.parquet",
            usecols=[
                "srnumber",
                "createddate",
                "closeddate",
                "status",
                "requesttype",
                "actiontaken",
                "owner",
                "latitude",
                "longitude",
                "ncname",
                "nc",
                "cd",
            ],
            mapping={
                "request_id": "srnumber",
                "created_date": "createddate",
                "closed_date": "closeddate",
                "status": "status",
                "category": "requesttype",
                "descriptor": "actiontaken",
                "agency": "owner",
                "latitude": "latitude",
                "longitude": "longitude",
                "area": lambda frame: first_nonempty(frame, ["ncname", "nc", "cd"]),
            },
            reader=read_los_angeles_raw,
        ),
    ]


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in build_specs():
        print(f"Harmonizing {spec.city}: {spec.raw_path}", flush=True)
        result = harmonize_file(spec)
        results.append(result)
        print(
            f"  wrote {result['rows']:,} rows to {result['output_path']}",
            flush=True,
        )

    manifest = {
        "target_schema": TARGET_SCHEMA,
        "outputs": results,
        "mapping_notes": {
            "nyc": {
                "request_id": "unique_key",
                "category": "complaint_type",
                "descriptor": "descriptor",
                "agency": "agency",
                "area": "community_board, fallback borough",
            },
            "chicago": {
                "request_id": "sr_number",
                "category": "sr_type",
                "descriptor": "sr_short_code",
                "agency": "owner_department",
                "area": "community_area, fallback ward",
            },
            "boston": {
                "request_id": "case_enquiry_id",
                "created_date": "open_dt",
                "closed_date": "closed_dt",
                "category": "reason",
                "descriptor": "type",
                "agency": "subject, fallback department",
                "area": "neighborhood, fallback ward",
            },
            "los_angeles": {
                "request_id": "srnumber",
                "created_date": "createddate",
                "closed_date": "closeddate",
                "category": "requesttype",
                "descriptor": "actiontaken",
                "agency": "owner",
                "area": "ncname, fallback nc, fallback cd",
                "reader_note": "Aligns 2021 LA rows with 2022+ rows that include createdbyuserorganization before selecting target columns.",
            },
        },
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {MANIFEST_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
