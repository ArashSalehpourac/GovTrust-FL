"""Run Step 4 target-label creation for all cleaned city datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.labeling import LABELED_SCHEMA, LabelingSpec, label_file  # noqa: E402


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
THRESHOLDS_PATH = PROCESSED_DIR / "delay_thresholds_q3_by_city_category.parquet"
THRESHOLDS_CSV_PATH = RESULTS_TABLES_DIR / "delay_thresholds_by_city_category.csv"
MANIFEST_PATH = PROCESSED_DIR / "labeling_manifest_step4.json"


def build_specs() -> list[LabelingSpec]:
    return [
        LabelingSpec(
            city="nyc",
            input_path=PROCESSED_DIR / "nyc_cleaned.parquet",
            output_path=PROCESSED_DIR / "nyc_labeled.parquet",
        ),
        LabelingSpec(
            city="chicago",
            input_path=PROCESSED_DIR / "chicago_cleaned.parquet",
            output_path=PROCESSED_DIR / "chicago_labeled.parquet",
        ),
        LabelingSpec(
            city="boston",
            input_path=PROCESSED_DIR / "boston_cleaned.parquet",
            output_path=PROCESSED_DIR / "boston_labeled.parquet",
        ),
        LabelingSpec(
            city="los_angeles",
            input_path=PROCESSED_DIR / "la_cleaned.parquet",
            output_path=PROCESSED_DIR / "la_labeled.parquet",
        ),
    ]


def main() -> int:
    results = []
    threshold_frames = []

    for spec in build_specs():
        print(f"Labeling {spec.city}: {spec.input_path}", flush=True)
        result, thresholds = label_file(spec)
        thresholds = thresholds.copy()
        thresholds["source_city"] = spec.city
        threshold_frames.append(thresholds)
        results.append(result)
        print(
            f"  wrote {result['rows']:,} rows; "
            f"delayed rate={result['delayed_rate']:.4f}; "
            f"threshold groups={result['threshold_groups']}",
            flush=True,
        )

    all_thresholds = pd.concat(threshold_frames, ignore_index=True)
    all_thresholds = all_thresholds[["city", "category", "delay_threshold_hours"]]
    all_thresholds.to_parquet(THRESHOLDS_PATH, index=False)
    THRESHOLDS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_thresholds.to_csv(THRESHOLDS_CSV_PATH, index=False)

    manifest = {
        "labeled_schema": LABELED_SCHEMA,
        "main_target": {
            "name": "delayed",
            "rule": "For each city and category, threshold = Q3(resolution_hours | city, category). delayed = 1 if resolution_hours > threshold else 0.",
            "threshold_path": str(THRESHOLDS_PATH),
            "threshold_csv_path": str(THRESHOLDS_CSV_PATH),
        },
        "secondary_target": {
            "name": "service_routing_target",
            "rule": "Use agency as the routing target, with category as fallback only when agency is missing.",
        },
        "outputs": results,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote thresholds: {THRESHOLDS_PATH}", flush=True)
    print(f"Wrote threshold CSV: {THRESHOLDS_CSV_PATH}", flush=True)
    print(f"Wrote manifest: {MANIFEST_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
