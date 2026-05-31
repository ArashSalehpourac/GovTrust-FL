"""Creation-time feature engineering for delayed-resolution prediction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


FEDERATED_TRAINING_CITIES = {"nyc", "chicago", "boston"}

BASE_FEATURE_COLUMNS = [
    "city",
    "category",
    "descriptor",
    "agency",
    "area",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "latitude_grid",
    "longitude_grid",
    "area_request_count_7d",
    "category_request_count_7d",
    "agency_request_count_7d",
]

TARGET_COLUMNS = [
    "delayed",
    "service_routing_target",
]

FEATURE_OUTPUT_COLUMNS = [
    "request_id",
    "created_date",
    *BASE_FEATURE_COLUMNS,
    *TARGET_COLUMNS,
]


@dataclass(frozen=True)
class FeatureSpec:
    """Input/output paths for one labeled city dataset."""

    city: str
    input_path: Path
    output_path: Path


def make_descriptor_vectorizer(max_features: int = 100, min_df: int = 10) -> TfidfVectorizer:
    """Create the shared descriptor TF-IDF vectorizer."""

    return TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        ngram_range=(1, 2),
        strip_accents="unicode",
        lowercase=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_/-]+\b",
        dtype=np.float32,
    )


def fit_descriptor_vectorizer(
    frames: dict[str, pd.DataFrame],
    output_path: Path,
    max_features: int = 100,
    min_df: int = 10,
) -> TfidfVectorizer:
    """Fit TF-IDF on federated clients only and persist the vectorizer."""

    texts = []
    for city, frame in frames.items():
        if city in FEDERATED_TRAINING_CITIES:
            texts.append(_descriptor_text(frame))
    if not texts:
        raise ValueError("No federated training-city frames supplied for TF-IDF fitting.")

    vectorizer = make_descriptor_vectorizer(max_features=max_features, min_df=min_df)
    vectorizer.fit(pd.concat(texts, ignore_index=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, output_path)
    return vectorizer


def engineer_features(
    frame: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    grid_precision: int = 2,
) -> pd.DataFrame:
    """Create non-leaky features available at request creation time."""

    working = frame.copy()
    working["created_date"] = pd.to_datetime(working["created_date"], errors="coerce")
    working = working.sort_values(["created_date", "request_id"], kind="stable").reset_index(drop=True)

    output = working[
        [
            "request_id",
            "created_date",
            "city",
            "category",
            "descriptor",
            "agency",
            "area",
            "delayed",
            "service_routing_target",
        ]
    ].copy()

    output["hour"] = working["created_date"].dt.hour.astype("int8")
    output["day_of_week"] = working["created_date"].dt.dayofweek.astype("int8")
    output["month"] = working["created_date"].dt.month.astype("int8")
    output["is_weekend"] = output["day_of_week"].isin([5, 6]).astype("int8")

    scale = 10**grid_precision
    output["latitude_grid"] = np.floor(working["latitude"].astype(float) * scale) / scale
    output["longitude_grid"] = np.floor(working["longitude"].astype(float) * scale) / scale

    output["area_request_count_7d"] = historical_request_count(
        working,
        ["city", "area"],
        "7D",
    )
    output["category_request_count_7d"] = historical_request_count(
        working,
        ["city", "category"],
        "7D",
    )
    output["agency_request_count_7d"] = historical_request_count(
        working,
        ["city", "agency"],
        "7D",
    )

    tfidf = vectorizer.transform(_descriptor_text(working))
    tfidf_names = tfidf_feature_names(vectorizer)
    tfidf_frame = pd.DataFrame(
        tfidf.toarray(),
        columns=tfidf_names,
        index=output.index,
    )

    output = pd.concat([output[FEATURE_OUTPUT_COLUMNS], tfidf_frame], axis=1)
    return output


def historical_request_count(
    frame: pd.DataFrame,
    group_columns: list[str],
    window: str,
) -> pd.Series:
    """Count prior requests in a trailing time window, excluding current rows."""

    working = frame[["created_date", *group_columns]].copy()
    working["_original_index"] = np.arange(len(working))

    counts = pd.Series(0, index=working.index, dtype="int32")
    grouped = working.groupby(group_columns, dropna=False, sort=False)
    for _, group in grouped:
        group = group.sort_values(["created_date", "_original_index"], kind="stable")
        marker = pd.Series(
            1,
            index=pd.DatetimeIndex(group["created_date"]),
            dtype="int8",
        )
        rolling_counts = marker.rolling(window, closed="left").count().fillna(0).astype("int32")
        counts.iloc[group["_original_index"].to_numpy()] = rolling_counts.to_numpy()

    return counts


def tfidf_feature_names(vectorizer: TfidfVectorizer) -> list[str]:
    """Return safe TF-IDF feature column names."""

    return [f"tfidf_descriptor__{_safe_name(name)}" for name in vectorizer.get_feature_names_out()]


def write_feature_manifest(
    path: Path,
    results: list[dict[str, object]],
    vectorizer_path: Path,
    vectorizer: TfidfVectorizer,
    grid_precision: int,
) -> None:
    """Write an audit manifest for Step 5 outputs."""

    manifest = {
        "creation_time_only": True,
        "base_features": BASE_FEATURE_COLUMNS,
        "tfidf_features": tfidf_feature_names(vectorizer),
        "target_columns": TARGET_COLUMNS,
        "non_feature_columns": ["request_id", "created_date"],
        "tfidf_fit_scope": "Federated clients only: nyc, chicago, boston. Los Angeles is transformed using the training vocabulary.",
        "tfidf_vectorizer_path": str(vectorizer_path),
        "geo_grid": {
            "method": "floor(latitude/longitude to decimal grid)",
            "decimal_precision": grid_precision,
        },
        "workload_windows": {
            "area_request_count_7d": "Prior requests in same city and area during previous 7 days, excluding current row.",
            "category_request_count_7d": "Prior requests in same city and category during previous 7 days, excluding current row.",
            "agency_request_count_7d": "Prior requests in same city and agency during previous 7 days, excluding current row.",
        },
        "excluded_as_leakage": [
            "closed_date",
            "resolution_hours",
            "delay_threshold_hours",
            "status after cleaning",
        ],
        "outputs": results,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _descriptor_text(frame: pd.DataFrame) -> pd.Series:
    return frame["descriptor"].fillna("").astype("string")


def _safe_name(name: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")
    return safe or "empty"
