from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import HashingVectorizer

TEXT_COL = "category"
CAT_COLS = ["category"]
NUM_COLS = ["hour", "day_of_week", "month", "is_weekend"]
FORBIDDEN_PRIMARY = {
    "city",
    "agency",
    "latitude",
    "longitude",
    "latitude_grid",
    "longitude_grid",
    "area",
    "zip_code",
    "descriptor",
}
TEXT_HASH_DIM = 512
CATEGORY_HASH_DIM = 64
NUMERIC_OUTPUT_FEATURES = (
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
)


@dataclass(frozen=True)
class FixedPreprocessor:
    """Data-independent primary representation.

    No vocabulary, category level, imputer, scaler, or statistic is learned from
    any municipal record. This is required so the released DP pipeline does not
    leak through an unprotected preprocessing fit.
    """

    text_hasher: HashingVectorizer
    category_hasher: FeatureHasher
    text_hash_dim: int
    category_hash_dim: int
    fingerprint: str

    @property
    def output_dim(self) -> int:
        return self.text_hash_dim + self.category_hash_dim + len(NUMERIC_OUTPUT_FEATURES)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        required = {TEXT_COL, *CAT_COLS, *NUM_COLS}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"missing primary feature columns: {sorted(missing)}")

        descriptor = frame[TEXT_COL].fillna("").astype(str)
        text = self.text_hasher.transform(descriptor)

        category_tokens = [
            [f"category={value}"]
            for value in frame[CAT_COLS[0]].fillna("UNK").astype(str)
        ]
        category = self.category_hasher.transform(category_tokens)

        hour = pd.to_numeric(frame["hour"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        day = pd.to_numeric(frame["day_of_week"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        month = pd.to_numeric(frame["month"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        weekend = pd.to_numeric(frame["is_weekend"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        numeric = np.column_stack(
            [
                np.sin(2.0 * np.pi * hour / 24.0),
                np.cos(2.0 * np.pi * hour / 24.0),
                np.sin(2.0 * np.pi * day / 7.0),
                np.cos(2.0 * np.pi * day / 7.0),
                np.sin(2.0 * np.pi * (month - 1.0) / 12.0),
                np.cos(2.0 * np.pi * (month - 1.0) / 12.0),
                weekend,
            ]
        ).astype(np.float32)

        if hasattr(text, "toarray"):
            text = text.toarray()
        if hasattr(category, "toarray"):
            category = category.toarray()
        return np.concatenate(
            [
                np.asarray(text, dtype=np.float32),
                np.asarray(category, dtype=np.float32),
                numeric,
            ],
            axis=1,
        )


def build_fixed_preprocessor(
    *,
    text_hash_dim: int = TEXT_HASH_DIM,
    category_hash_dim: int = CATEGORY_HASH_DIM,
) -> FixedPreprocessor:
    """Construct the fixed representation without accepting or inspecting data."""

    if text_hash_dim < 1 or category_hash_dim < 1:
        raise ValueError("hash dimensions must be positive")

    text_hasher = HashingVectorizer(
        n_features=text_hash_dim,
        alternate_sign=False,
        ngram_range=(1, 2),
        norm="l2",
        lowercase=True,
    )
    category_hasher = FeatureHasher(
        n_features=category_hash_dim,
        input_type="string",
        alternate_sign=False,
    )
    canonical = {
        "representation": "data-independent lexical and identity hashing of intake category plus deterministic cyclical time encoding",
        "text_column": TEXT_COL,
        "text_hash_dim": text_hash_dim,
        "text_ngram_range": [1, 2],
        "text_alternate_sign": False,
        "text_norm": "l2",
        "category_column": CAT_COLS[0],
        "category_hash_dim": category_hash_dim,
        "category_alternate_sign": False,
        "numeric_inputs": NUM_COLS,
        "numeric_outputs": list(NUMERIC_OUTPUT_FEATURES),
    }
    fingerprint = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return FixedPreprocessor(
        text_hasher=text_hasher,
        category_hasher=category_hasher,
        text_hash_dim=text_hash_dim,
        category_hash_dim=category_hash_dim,
        fingerprint=fingerprint,
    )


def feature_manifest(preprocessor: FixedPreprocessor) -> dict[str, object]:
    return {
        "representation": "data-independent fixed hashing of intake category plus deterministic time encoding",
        "text": TEXT_COL,
        "text_hash_dim": preprocessor.text_hash_dim,
        "categorical": CAT_COLS,
        "category_hash_dim": preprocessor.category_hash_dim,
        "numeric_inputs": NUM_COLS,
        "numeric_outputs": list(NUMERIC_OUTPUT_FEATURES),
        "fit_scope": "none; no data-dependent preprocessing fit",
        "explicitly_excluded_primary": sorted(FORBIDDEN_PRIMARY),
        "descriptor_policy": "excluded from primary predictors because Los Angeles descriptor is sourced from ActionTaken, a post-service field",
    }
