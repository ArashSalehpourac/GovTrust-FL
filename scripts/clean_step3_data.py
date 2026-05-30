"""Run Step 3 cleaning for all harmonized city datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cleaning import CLEANED_SCHEMA, CleaningSpec, clean_file  # noqa: E402


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "cleaning_manifest_step3.json"


def build_specs() -> list[CleaningSpec]:
    return [
        CleaningSpec(
            city="nyc",
            input_path=PROCESSED_DIR / "nyc_harmonized.parquet",
            output_path=PROCESSED_DIR / "nyc_cleaned.parquet",
        ),
        CleaningSpec(
            city="chicago",
            input_path=PROCESSED_DIR / "chicago_harmonized.parquet",
            output_path=PROCESSED_DIR / "chicago_cleaned.parquet",
        ),
        CleaningSpec(
            city="boston",
            input_path=PROCESSED_DIR / "boston_harmonized.parquet",
            output_path=PROCESSED_DIR / "boston_cleaned.parquet",
        ),
        CleaningSpec(
            city="los_angeles",
            input_path=PROCESSED_DIR / "la_harmonized.parquet",
            output_path=PROCESSED_DIR / "la_cleaned.parquet",
        ),
    ]


def main() -> int:
    results = []
    for spec in build_specs():
        print(f"Cleaning {spec.city}: {spec.input_path}", flush=True)
        result = clean_file(spec, min_category_samples=100)
        results.append(result)
        counts = result["counts"]
        print(
            f"  wrote {counts['output_rows']:,} rows "
            f"({counts['total_removed']:,} removed) to {result['output_path']}",
            flush=True,
        )

    manifest = {
        "cleaned_schema": CLEANED_SCHEMA,
        "rules": [
            "Convert created_date and closed_date to datetime.",
            "Remove records without created_date.",
            "Remove records without closed_date.",
            "Remove records where closed_date < created_date.",
            "Remove duplicate request_id, keeping the earliest created_date.",
            "Keep only final statuses: Closed or Completed; remove cancelled, invalid, incomplete, and non-final statuses.",
            "Remove impossible coordinates: missing latitude/longitude, latitude outside [-90, 90], longitude outside [-180, 180], or coordinate (0, 0).",
            "Compute resolution_hours as closed_date - created_date in hours.",
            "Remove categories with fewer than 100 samples within each city after the preceding cleaning rules.",
        ],
        "outputs": results,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {MANIFEST_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

