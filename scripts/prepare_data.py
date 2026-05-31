"""Prepare canonical city datasets and temporal splits for GovTrust-FL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.cleaning import clean_frame  # noqa: E402
from src.config import CITY_CONFIGS, EXTERNAL_VALIDATION_CITY, FEDERATED_CLIENTS, ensure_project_dirs  # noqa: E402
from src.config_loader import load_config, resolve_path  # noqa: E402
from src.feature_engineering import engineer_features, fit_descriptor_vectorizer  # noqa: E402
from src.labeling import add_delayed_resolution_label  # noqa: E402
from src.schema_harmonization import HarmonizationSpec, TARGET_SCHEMA, harmonize_frame  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--tfidf-max-features", type=int, default=50)
    parser.add_argument("--tfidf-min-df", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_dirs()

    raw_dir = resolve_path(config, "data.raw_dir")
    processed_dir = resolve_path(config, "data.processed_dir")
    splits_dir = resolve_path(config, "data.splits_dir")
    models_dir = resolve_path(config, "outputs.models_dir")
    logs_dir = resolve_path(config, "outputs.logs_dir")
    min_category_samples = int(config["data"].get("min_category_samples", 1))

    labeled_frames: dict[str, pd.DataFrame] = {}
    manifest: dict[str, object] = {"cities": {}}
    for city in (*FEDERATED_CLIENTS, EXTERNAL_VALIDATION_CITY):
        raw = load_raw_city(raw_dir / city, city)
        harmonized = harmonize_canonical(raw, city)
        cleaned, counts = clean_frame(harmonized, min_category_samples=min_category_samples)
        labeled, thresholds = add_delayed_resolution_label(cleaned)
        labeled_frames[city] = labeled
        manifest["cities"][city] = {
            "raw_rows": len(raw),
            "cleaned_rows": len(cleaned),
            "labeled_rows": len(labeled),
            "threshold_groups": len(thresholds),
            "cleaning_counts": counts,
        }

    vectorizer_path = models_dir / "descriptor_tfidf_prepare_data.joblib"
    vectorizer = fit_descriptor_vectorizer(
        labeled_frames,
        vectorizer_path,
        max_features=args.tfidf_max_features,
        min_df=args.tfidf_min_df,
    )

    processed_frames = {}
    for city, labeled in labeled_frames.items():
        features = engineer_features(labeled, vectorizer)
        output_path = processed_dir / f"{city}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(output_path, index=False)
        processed_frames[city] = features
        manifest["cities"][city]["processed_path"] = str(output_path)
        manifest["cities"][city]["processed_rows"] = len(features)

    create_splits(
        processed_frames,
        splits_dir,
        validation_size=float(config["data"]["validation_size"]),
        test_size=float(config["data"]["test_size"]),
    )
    joblib.dump(vectorizer, vectorizer_path)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "prepare_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared data for {len(processed_frames)} cities. Manifest: {logs_dir / 'prepare_data_manifest.json'}")
    return 0


def load_raw_city(city_dir: Path, city: str) -> pd.DataFrame:
    files = sorted([*city_dir.glob("*.csv"), *city_dir.glob("*.parquet")])
    if not files:
        raise FileNotFoundError(f"No raw files found for {city}: {city_dir}")
    frames = []
    for path in files:
        if path.suffix.lower() == ".csv":
            frames.append(pd.read_csv(path))
        elif path.suffix.lower() == ".parquet":
            frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def harmonize_canonical(raw: pd.DataFrame, city: str) -> pd.DataFrame:
    missing = [column for column in TARGET_SCHEMA if column not in raw.columns and column != "city"]
    if missing:
        raise ValueError(
            f"prepare_data.py expects canonical columns for {city}; missing {missing}. "
            "Use the city-specific download/harmonization scripts for original municipal exports."
        )
    frame = raw.copy()
    frame["city"] = city
    identity_mapping = {column: column for column in TARGET_SCHEMA if column != "city"}
    spec = HarmonizationSpec(
        city=city,
        raw_path=Path(),
        output_path=Path(),
        mapping=identity_mapping,
        usecols=list(identity_mapping),
    )
    return harmonize_frame(frame, spec)


def create_splits(
    frames: dict[str, pd.DataFrame],
    splits_dir: Path,
    *,
    validation_size: float,
    test_size: float,
) -> None:
    for city in FEDERATED_CLIENTS:
        frame = frames[city].sort_values(["created_date", "request_id"], kind="stable").reset_index(drop=True)
        train_end = int(len(frame) * (1.0 - validation_size - test_size))
        val_end = int(len(frame) * (1.0 - test_size))
        split_map = {
            "train": frame.iloc[:train_end].copy(),
            "val": frame.iloc[train_end:val_end].copy(),
            "test": frame.iloc[val_end:].copy(),
        }
        city_dir = splits_dir / city
        city_dir.mkdir(parents=True, exist_ok=True)
        for split_name, split_frame in split_map.items():
            split_frame.to_parquet(city_dir / f"{city}_{split_name}.parquet", index=False)

    external_dir = splits_dir / "external"
    external_dir.mkdir(parents=True, exist_ok=True)
    frames[EXTERNAL_VALIDATION_CITY].sort_values(
        ["created_date", "request_id"], kind="stable"
    ).to_parquet(external_dir / "los_angeles_external_test.parquet", index=False)


if __name__ == "__main__":
    raise SystemExit(main())
