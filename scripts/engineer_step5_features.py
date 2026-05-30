"""Run Step 5 creation-time feature engineering."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import (  # noqa: E402
    BASE_FEATURE_COLUMNS,
    FEATURE_OUTPUT_COLUMNS,
    FeatureSpec,
    engineer_features,
    fit_descriptor_vectorizer,
    tfidf_feature_names,
    write_feature_manifest,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_MODELS_DIR = PROJECT_ROOT / "results" / "models"
VECTORIZER_PATH = RESULTS_MODELS_DIR / "descriptor_tfidf_step5.joblib"
MANIFEST_PATH = PROCESSED_DIR / "feature_manifest_step5.json"
GRID_PRECISION = 2
TFIDF_MAX_FEATURES = 100


def build_specs() -> list[FeatureSpec]:
    return [
        FeatureSpec(
            city="nyc",
            input_path=PROCESSED_DIR / "nyc_labeled.parquet",
            output_path=PROCESSED_DIR / "nyc_features.parquet",
        ),
        FeatureSpec(
            city="chicago",
            input_path=PROCESSED_DIR / "chicago_labeled.parquet",
            output_path=PROCESSED_DIR / "chicago_features.parquet",
        ),
        FeatureSpec(
            city="boston",
            input_path=PROCESSED_DIR / "boston_labeled.parquet",
            output_path=PROCESSED_DIR / "boston_features.parquet",
        ),
        FeatureSpec(
            city="los_angeles",
            input_path=PROCESSED_DIR / "la_labeled.parquet",
            output_path=PROCESSED_DIR / "la_features.parquet",
        ),
    ]


def main() -> int:
    specs = build_specs()
    frames = {
        spec.city: pd.read_parquet(spec.input_path)
        for spec in specs
    }

    print("Fitting descriptor TF-IDF on nyc, chicago, and boston only", flush=True)
    vectorizer = fit_descriptor_vectorizer(
        frames,
        VECTORIZER_PATH,
        max_features=TFIDF_MAX_FEATURES,
    )
    print(f"  vocabulary size={len(vectorizer.get_feature_names_out())}", flush=True)

    results = []
    for spec in specs:
        print(f"Engineering {spec.city}: {spec.input_path}", flush=True)
        features = engineer_features(
            frames[spec.city],
            vectorizer,
            grid_precision=GRID_PRECISION,
        )
        features.to_parquet(spec.output_path, index=False)
        tfidf_columns = tfidf_feature_names(vectorizer)
        result = {
            "city": spec.city,
            "input_path": str(spec.input_path),
            "output_path": str(spec.output_path),
            "rows": int(len(features)),
            "columns": int(len(features.columns)),
            "base_feature_columns": BASE_FEATURE_COLUMNS,
            "tfidf_feature_count": len(tfidf_columns),
            "delayed_rate": float(features["delayed"].mean()),
            "service_routing_target_classes": int(features["service_routing_target"].nunique(dropna=True)),
            "max_area_request_count_7d": int(features["area_request_count_7d"].max()),
            "max_category_request_count_7d": int(features["category_request_count_7d"].max()),
            "max_agency_request_count_7d": int(features["agency_request_count_7d"].max()),
        }
        results.append(result)
        print(
            f"  wrote {result['rows']:,} rows x {result['columns']} columns "
            f"to {spec.output_path}",
            flush=True,
        )

    write_feature_manifest(
        MANIFEST_PATH,
        results,
        VECTORIZER_PATH,
        vectorizer,
        GRID_PRECISION,
    )
    print(f"Wrote manifest: {MANIFEST_PATH}", flush=True)
    print(f"Wrote vectorizer: {VECTORIZER_PATH}", flush=True)
    print(
        "Feature columns exclude closed_date, resolution_hours, delay_threshold_hours, and post-cleaning status.",
        flush=True,
    )
    print(f"Core output prefix: {FEATURE_OUTPUT_COLUMNS}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

