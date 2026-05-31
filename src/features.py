"""Feature matrix construction for tabular models."""

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CLOSED_AT_COLUMN, OPENED_AT_COLUMN, RESOLUTION_DAYS_COLUMN, TARGET_COLUMN


DEFAULT_DROP_COLUMNS = {
    TARGET_COLUMN,
    "request_id",
    "created_date",
    OPENED_AT_COLUMN,
    CLOSED_AT_COLUMN,
    RESOLUTION_DAYS_COLUMN,
    "delay_threshold_hours",
    "service_routing_target",
    "status",
    "source_file",
}


def infer_feature_columns(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    drop_columns: Sequence[str] | None = None,
) -> list[str]:
    """Infer model features by removing target/leakage columns."""

    excluded = set(DEFAULT_DROP_COLUMNS)
    excluded.add(target_column)
    if drop_columns:
        excluded.update(drop_columns)
    return [column for column in frame.columns if column not in excluded]


def build_xy(
    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    drop_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a processed frame into feature and target objects."""

    working = frame.dropna(subset=[target_column]).copy()
    feature_columns = infer_feature_columns(working, target_column=target_column, drop_columns=drop_columns)
    return working[feature_columns], working[target_column].astype(int)


def make_tabular_preprocessor() -> ColumnTransformer:
    """Create a numeric/categorical preprocessing transformer."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, make_column_selector(dtype_include="number")),
            ("cat", categorical_pipeline, make_column_selector(dtype_exclude="number")),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
