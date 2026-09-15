from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TEXT_COL = "descriptor"
CAT_COLS = ["category"]
NUM_COLS = ["hour", "day_of_week", "month", "is_weekend"]
FORBIDDEN_PRIMARY = {"city", "agency", "latitude", "longitude", "latitude_grid", "longitude_grid", "area", "zip_code"}


@dataclass
class SourceTrainPreprocessor:
    transformer: ColumnTransformer
    fitted_rows: int
    fitted_cities: tuple[str, ...]
    fingerprint: str

    @property
    def output_dim(self) -> int:
        return int(len(self.transformer.get_feature_names_out()))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.transformer.transform(frame)
        if hasattr(x, "toarray"):
            x = x.toarray()
        return np.asarray(x, dtype=np.float32)


def fit_source_train(source_splits: Mapping[str, Mapping[str, pd.DataFrame]], max_text_features: int = 1000) -> SourceTrainPreprocessor:
    cities = tuple(sorted(source_splits))
    train = pd.concat([source_splits[c]["train"] for c in cities], ignore_index=True)
    if any(col in FORBIDDEN_PRIMARY for col in CAT_COLS + NUM_COLS + [TEXT_COL]):
        raise AssertionError("forbidden city shortcut in primary feature set")

    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_text_features, min_df=2, ngram_range=(1, 2))),
    ])
    transformer = ColumnTransformer(
        transformers=[
            ("text", text_pipe, TEXT_COL),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CAT_COLS),
            ("num", StandardScaler(), NUM_COLS),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    transformer.fit(train)

    tfidf = transformer.named_transformers_["text"].named_steps["tfidf"]
    encoder = transformer.named_transformers_["cat"]
    scaler = transformer.named_transformers_["num"]
    canonical = {
        "fit_scope": "source-city train partitions only",
        "cities": list(cities),
        "rows": int(len(train)),
        "text_vocabulary": sorted(tfidf.vocabulary_.items(), key=lambda kv: kv[1]),
        "category_levels": [list(map(str, levels)) for levels in encoder.categories_],
        "numeric_mean": np.asarray(scaler.mean_, dtype=float).round(12).tolist(),
        "numeric_scale": np.asarray(scaler.scale_, dtype=float).round(12).tolist(),
        "features": {"text": TEXT_COL, "categorical": CAT_COLS, "numeric": NUM_COLS},
    }
    fingerprint = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return SourceTrainPreprocessor(transformer, int(len(train)), cities, fingerprint)


def category_oov_rate(preprocessor: SourceTrainPreprocessor, frame: pd.DataFrame) -> float:
    encoder = preprocessor.transformer.named_transformers_["cat"]
    known = set(map(str, encoder.categories_[0]))
    values = frame["category"].astype(str)
    return float((~values.isin(known)).mean()) if len(values) else 0.0
