"""Leave-one-city-out fold construction with auditable leakage controls.

Order of operations (this order is the leakage control):

1. cap rows per city,
2. build creation-time features (transformation only),
3. split each *source* city chronologically into train / val / internal test,
4. fit the target policy on pooled source-city training rows,
5. fit the preprocessing artifacts on the same pooled source-city training rows,
6. transform everything else, including the held-out city, with those artifacts.

The held-out city never participates in steps 4 and 5.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import (
    FOLDS_DIR,
    PRIMARY_TARGET_COLUMN,
    RESOLUTION_HOURS_COLUMN,
    SECONDARY_TARGET_COLUMN,
    FoldSpec,
    SplitConfig,
    TargetConfig,
)
from .features import build_features
from .preprocess import FoldPreprocessor
from .targets import TargetPolicy, apply_target_policy, fit_target_policy

SPLIT_NAMES = ("train", "val", "internal_test")


@dataclass
class FoldData:
    """Materialized fold: per-city source splits plus the external city."""

    spec: FoldSpec
    source_splits: dict[str, dict[str, pd.DataFrame]]
    external: pd.DataFrame
    preprocessor: FoldPreprocessor
    policy: TargetPolicy
    manifest: dict[str, object]

    def pooled(self, split: str) -> pd.DataFrame:
        return pd.concat(
            [splits[split] for splits in self.source_splits.values()], ignore_index=True
        )


def cap_rows(frame: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    """Keep the most recent ``max_rows`` rows, preserving chronological order."""

    ordered = frame.sort_values(["created_date", "request_id"], kind="stable")
    if max_rows is not None and len(ordered) > max_rows:
        ordered = ordered.iloc[-max_rows:]
    return ordered.reset_index(drop=True)


def chronological_split(frame: pd.DataFrame, config: SplitConfig) -> dict[str, pd.DataFrame]:
    """Split one city chronologically; no shuffling, no target-aware logic."""

    ordered = frame.sort_values(["created_date", "request_id"], kind="stable").reset_index(drop=True)
    n_rows = len(ordered)
    train_end = int(n_rows * config.train_frac)
    val_end = int(n_rows * (config.train_frac + config.val_frac))
    return {
        "train": ordered.iloc[:train_end].reset_index(drop=True),
        "val": ordered.iloc[train_end:val_end].reset_index(drop=True),
        "internal_test": ordered.iloc[val_end:].reset_index(drop=True),
    }


def build_fold(
    spec: FoldSpec,
    city_frames: dict[str, pd.DataFrame],
    *,
    split_config: SplitConfig | None = None,
    target_config: TargetConfig | None = None,
    max_rows_per_city: int | None = None,
    tfidf_max_features: int = 100,
) -> FoldData:
    """Build one leave-one-city-out fold in memory."""

    split_config = split_config or SplitConfig()
    target_config = target_config or TargetConfig()

    missing = [city for city in (*spec.source_cities, spec.held_out_city) if city not in city_frames]
    if missing:
        raise KeyError(f"Missing cleaned frames for cities: {missing}")

    source_splits: dict[str, dict[str, pd.DataFrame]] = {}
    for city in spec.source_cities:
        featurized = build_features(cap_rows(city_frames[city], max_rows_per_city))
        source_splits[city] = chronological_split(featurized, split_config)

    external = build_features(cap_rows(city_frames[spec.held_out_city], max_rows_per_city))

    pooled_train = pd.concat(
        [splits["train"] for splits in source_splits.values()], ignore_index=True
    ).sort_values(["created_date", "city", "request_id"], kind="stable").reset_index(drop=True)

    if spec.held_out_city in set(pooled_train["city"].astype(str)):
        raise AssertionError("Held-out city rows reached the fitting frame.")

    policy = fit_target_policy(pooled_train, target_config)
    preprocessor = FoldPreprocessor.fit(
        pooled_train,
        tfidf_max_features=tfidf_max_features,
        fit_scope={
            "fold": spec.name,
            "held_out_city": spec.held_out_city,
            "fitted_on_cities": sorted(spec.source_cities),
            "fitted_on_split": "train",
            "fitted_on_rows": int(len(pooled_train)),
            "created_date_max": _iso_or_none(pooled_train["created_date"].max()),
        },
    )

    source_splits = {
        city: {name: apply_target_policy(frame, policy) for name, frame in splits.items()}
        for city, splits in source_splits.items()
    }
    external = apply_target_policy(external, policy)

    manifest = _build_manifest(spec, source_splits, external, preprocessor, policy, split_config)
    return FoldData(
        spec=spec,
        source_splits=source_splits,
        external=external,
        preprocessor=preprocessor,
        policy=policy,
        manifest=manifest,
    )


def write_fold(fold: FoldData, folds_dir: Path = FOLDS_DIR) -> Path:
    """Persist fold splits, the fitted preprocessor, the policy, and the manifest."""

    fold_dir = folds_dir / fold.spec.name
    fold_dir.mkdir(parents=True, exist_ok=True)

    for city, splits in fold.source_splits.items():
        for split_name, frame in splits.items():
            frame.to_parquet(fold_dir / f"source_{city}_{split_name}.parquet", index=False)
    fold.external.to_parquet(
        fold_dir / f"external_{fold.spec.held_out_city}.parquet", index=False
    )

    preprocessor_path = fold_dir / "preprocessor.joblib"
    fold.preprocessor.save(preprocessor_path)
    (fold_dir / "target_policy.json").write_text(
        json.dumps(fold.policy.to_dict(), indent=2), encoding="utf-8"
    )
    manifest = dict(fold.manifest)
    manifest["artifacts"] = {
        "preprocessor": str(preprocessor_path),
        "target_policy": str(fold_dir / "target_policy.json"),
    }
    (fold_dir / "fold_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return fold_dir


def load_fold(fold_name: str, folds_dir: Path = FOLDS_DIR) -> FoldData:
    """Load a previously written fold from disk."""

    fold_dir = folds_dir / fold_name
    manifest = json.loads((fold_dir / "fold_manifest.json").read_text(encoding="utf-8"))
    spec = FoldSpec(
        held_out_city=manifest["held_out_city"],
        source_cities=tuple(manifest["source_cities"]),
    )
    source_splits = {
        city: {
            split: pd.read_parquet(fold_dir / f"source_{city}_{split}.parquet")
            for split in SPLIT_NAMES
        }
        for city in spec.source_cities
    }
    external = pd.read_parquet(fold_dir / f"external_{spec.held_out_city}.parquet")
    preprocessor = FoldPreprocessor.load(fold_dir / "preprocessor.joblib")
    policy_payload = json.loads((fold_dir / "target_policy.json").read_text(encoding="utf-8"))
    policy = TargetPolicy(
        winsor_quantile=policy_payload["winsor_quantile"],
        winsor_hours=policy_payload["winsor_hours"],
        sla_quantile=policy_payload["sla_quantile"],
        global_sla_threshold_hours=policy_payload["global_sla_threshold_hours"],
        category_sla_threshold_hours=policy_payload["category_sla_threshold_hours"],
        fitted_on_rows=policy_payload["fit_scope"]["rows"],
        fitted_on_cities=tuple(policy_payload["fit_scope"]["cities"]),
        min_category_rows_for_threshold=policy_payload["min_category_rows_for_threshold"],
    )
    return FoldData(
        spec=spec,
        source_splits=source_splits,
        external=external,
        preprocessor=preprocessor,
        policy=policy,
        manifest=manifest,
    )


def _build_manifest(
    spec: FoldSpec,
    source_splits: dict[str, dict[str, pd.DataFrame]],
    external: pd.DataFrame,
    preprocessor: FoldPreprocessor,
    policy: TargetPolicy,
    split_config: SplitConfig,
) -> dict[str, object]:
    external_ids = set(external["request_id"].astype(str))
    source_ids: set[str] = set()
    for splits in source_splits.values():
        for frame in splits.values():
            source_ids |= set(frame["request_id"].astype(str))

    return {
        "fold": spec.name,
        "held_out_city": spec.held_out_city,
        "source_cities": list(spec.source_cities),
        "protocol": (
            "Four-fold leave-one-city-out. Source cities are split chronologically before any "
            "fitting. TF-IDF, encoders, imputers, scalers, target winsorization, SLA thresholds, "
            "and model selection all use source-city training rows only."
        ),
        "split_fractions": {
            "train": split_config.train_frac,
            "val": split_config.val_frac,
            "internal_test": round(1.0 - split_config.train_frac - split_config.val_frac, 6),
            "method": "chronological by created_date within each source city",
        },
        "targets": policy.to_dict(),
        "preprocessing": preprocessor.describe(),
        "splits": {
            city: {
                split: _split_summary(frame) for split, frame in splits.items()
            }
            for city, splits in source_splits.items()
        },
        "external": _split_summary(external),
        "isolation_checks": {
            "held_out_city_in_fitting_frames": False,
            "request_id_overlap_source_vs_external": len(source_ids & external_ids),
            "cities_in_source_splits": sorted(
                {
                    str(city)
                    for splits in source_splits.values()
                    for frame in splits.values()
                    for city in frame["city"].unique()
                }
            ),
            "cities_in_external": sorted({str(city) for city in external["city"].unique()}),
            "preprocessor_fingerprint": preprocessor.fingerprint(),
            "source_train_id_digest": _digest(source_splits, "train"),
        },
    }


def _split_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "created_date_min": _iso_or_none(frame["created_date"].min()),
        "created_date_max": _iso_or_none(frame["created_date"].max()),
        "resolution_hours_median": _float_or_none(frame[RESOLUTION_HOURS_COLUMN].median()),
        f"{PRIMARY_TARGET_COLUMN}_mean": _float_or_none(frame[PRIMARY_TARGET_COLUMN].mean())
        if PRIMARY_TARGET_COLUMN in frame.columns
        else None,
        f"{SECONDARY_TARGET_COLUMN}_rate": _float_or_none(frame[SECONDARY_TARGET_COLUMN].mean())
        if SECONDARY_TARGET_COLUMN in frame.columns
        else None,
    }


def _digest(source_splits: dict[str, dict[str, pd.DataFrame]], split: str) -> str:
    ids = sorted(
        str(value)
        for splits in source_splits.values()
        for value in splits[split]["request_id"].tolist()
    )
    return hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()


def _iso_or_none(value) -> str | None:
    if pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def _float_or_none(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
