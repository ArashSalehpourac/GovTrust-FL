"""Create Step 6 experimental train/validation/test splits."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.splitting import (  # noqa: E402
    EXTERNAL_CITY,
    FEDERATED_CITIES,
    SplitSpec,
    write_city_splits,
    write_combined_split,
    write_external_test,
    write_manifest,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
MANIFEST_PATH = SPLITS_DIR / "split_manifest_step6.json"


FEATURE_FILES = {
    "nyc": PROCESSED_DIR / "nyc_features.parquet",
    "chicago": PROCESSED_DIR / "chicago_features.parquet",
    "boston": PROCESSED_DIR / "boston_features.parquet",
    "los_angeles": PROCESSED_DIR / "la_features.parquet",
}


def main() -> int:
    city_results = []
    for city in FEDERATED_CITIES:
        print(f"Creating time splits for {city}", flush=True)
        city_results.append(
            write_city_splits(
                SplitSpec(
                    city=city,
                    input_path=FEATURE_FILES[city],
                    output_dir=SPLITS_DIR / city,
                )
            )
        )

    print("Writing Los Angeles external test split", flush=True)
    external_result = write_external_test(
        EXTERNAL_CITY,
        FEATURE_FILES[EXTERNAL_CITY],
        SPLITS_DIR / "external" / "los_angeles_external_test.parquet",
    )

    combined_results = []
    for split_name in ("train", "val", "test"):
        input_paths = [
            SPLITS_DIR / city / f"{city}_{split_name}.parquet"
            for city in FEDERATED_CITIES
        ]
        output_path = SPLITS_DIR / "centralized" / f"centralized_{split_name}.parquet"
        print(f"Writing centralized {split_name}", flush=True)
        combined_results.append(write_combined_split(input_paths, output_path, split_name))

    manifest = {
        "split_design": {
            "centralized": "NYC + Chicago + Boston pooled together.",
            "local_only": "Each of NYC, Chicago, and Boston keeps its own train/val/test split.",
            "federated_by_city": "Client 1=NYC, Client 2=Chicago, Client 3=Boston.",
            "federated_by_district": "Deferred until city-level FL version works.",
            "external_validation": "Los Angeles is external-only and not used in train/validation.",
        },
        "split_method": "Time-based within each federated city: earliest 70% train, next 15% validation, latest 15% test.",
        "city_splits": city_results,
        "external_test": external_result,
        "centralized_splits": combined_results,
    }
    write_manifest(MANIFEST_PATH, manifest)
    print(f"Wrote manifest: {MANIFEST_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

