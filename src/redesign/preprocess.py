"""Fold-specific preprocessing fitted on source-city training rows only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TEXT_FEATURE


@dataclass
class FoldPreprocessor:
    """TF-IDF + one-hot + impute/scale, all fitted on one fold's training rows."""

    tfidf: TfidfVectorizer
    column_transformer: ColumnTransformer
    feature_names: list[str]
    fit_scope: dict[str, object]

    @classmethod
    def fit(
        cls,
        train_frame: pd.DataFrame,
        *,
        tfidf_max_features: int = 100,
        fit_scope: dict[str, object] | None = None,
    ) -> "FoldPreprocessor":
        """Fit every stateful transform on ``train_frame`` and nothing else."""

        tfidf = TfidfVectorizer(
            max_features=tfidf_max_features,
            min_df=5,
            ngram_range=(1, 2),
            strip_accents="unicode",
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_/-]+\b",
            dtype=np.float32,
        )
        tfidf.fit(_text(train_frame))

        column_transformer = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                            (
                                "onehot",
                                OneHotEncoder(
                                    handle_unknown="infrequent_if_exist",
                                    min_frequency=10,
                                    sparse_output=False,
                                    dtype=np.float32,
                                ),
                            ),
                        ]
                    ),
                    list(CATEGORICAL_FEATURES),
                ),
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    list(NUMERIC_FEATURES),
                ),
            ],
            sparse_threshold=0.0,
        )
        column_transformer.fit(_tabular(train_frame))

        feature_names = [
            *(str(name) for name in column_transformer.get_feature_names_out()),
            *(f"tfidf__{name}" for name in tfidf.get_feature_names_out()),
        ]
        return cls(
            tfidf=tfidf,
            column_transformer=column_transformer,
            feature_names=feature_names,
            fit_scope=dict(fit_scope or {}),
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Transform any split with the already-fitted artifacts."""

        tabular = self.column_transformer.transform(_tabular(frame))
        text = self.tfidf.transform(_text(frame)).toarray()
        return np.hstack([tabular, text]).astype(np.float32)

    @property
    def output_dim(self) -> int:
        return len(self.feature_names)

    def fingerprint(self) -> str:
        """Stable hash of every fitted parameter, used by the leakage tests."""

        parts: list[str] = [
            json.dumps(sorted((term, int(index)) for term, index in self.tfidf.vocabulary_.items()))
        ]
        parts.append(np.array2string(self.tfidf.idf_, precision=10))
        categorical = self.column_transformer.named_transformers_["categorical"]
        numeric = self.column_transformer.named_transformers_["numeric"]
        parts.append(
            json.dumps([list(map(str, values)) for values in categorical["onehot"].categories_])
        )
        parts.append(np.array2string(numeric["imputer"].statistics_, precision=10))
        parts.append(np.array2string(numeric["scaler"].mean_, precision=10))
        parts.append(np.array2string(numeric["scaler"].scale_, precision=10))
        parts.append(json.dumps(self.feature_names))
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    def describe(self) -> dict[str, object]:
        numeric = self.column_transformer.named_transformers_["numeric"]
        categorical = self.column_transformer.named_transformers_["categorical"]
        return {
            "fit_scope": self.fit_scope,
            "fingerprint": self.fingerprint(),
            "output_dim": self.output_dim,
            "tfidf": {
                "column": TEXT_FEATURE,
                "vocabulary_size": len(self.tfidf.vocabulary_),
                "max_features": self.tfidf.max_features,
                "min_df": self.tfidf.min_df,
            },
            "categorical_features": list(CATEGORICAL_FEATURES),
            "categorical_levels": {
                column: int(len(levels))
                for column, levels in zip(
                    CATEGORICAL_FEATURES, categorical["onehot"].categories_, strict=True
                )
            },
            "numeric_features": list(NUMERIC_FEATURES),
            "numeric_median": numeric["imputer"].statistics_.tolist(),
            "numeric_mean": numeric["scaler"].mean_.tolist(),
            "numeric_scale": numeric["scaler"].scale_.tolist(),
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> "FoldPreprocessor":
        return joblib.load(path)


def _text(frame: pd.DataFrame) -> pd.Series:
    return frame[TEXT_FEATURE].astype("string").fillna("").astype(str)


def _tabular(frame: pd.DataFrame) -> pd.DataFrame:
    tabular = frame.loc[:, [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]].copy()
    for column in CATEGORICAL_FEATURES:
        tabular[column] = tabular[column].astype("string").fillna("missing").astype(str)
    for column in NUMERIC_FEATURES:
        tabular[column] = pd.to_numeric(tabular[column], errors="coerce")
    return tabular
