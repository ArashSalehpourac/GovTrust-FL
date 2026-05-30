"""Model factories and evaluation utilities for baseline experiments."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NON_FEATURE_COLUMNS = {
    "request_id",
    "created_date",
    "delayed",
    "service_routing_target",
}

CATEGORICAL_FEATURES = ["city", "category", "descriptor", "agency", "area"]
NUMERIC_BASE_FEATURES = [
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


@dataclass(frozen=True)
class ModelRunResult:
    """Timing, memory, and metric result for one fitted model."""

    model_name: str
    train_rows: int
    test_rows: int
    train_seconds: float
    inference_seconds: float
    peak_memory_mb: float
    metrics: dict[str, float]


def infer_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return non-leaky feature columns from Step 5 feature files."""

    return [column for column in frame.columns if column not in NON_FEATURE_COLUMNS]


def split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split Step 5 frame into X and delayed y."""

    features = frame[infer_feature_columns(frame)].copy()
    for column in CATEGORICAL_FEATURES:
        if column in features.columns:
            features[column] = features[column].astype("object").where(features[column].notna(), np.nan)
    for column in features.columns:
        if column not in CATEGORICAL_FEATURES:
            features[column] = pd.to_numeric(features[column], errors="coerce")
    return features, frame["delayed"].astype("int8")


def categorical_and_numeric_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_columns = infer_feature_columns(frame)
    categorical = [column for column in CATEGORICAL_FEATURES if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    return categorical, numeric


def make_preprocessor(
    frame: pd.DataFrame,
    *,
    scale_numeric: bool,
    dense_output: bool = False,
) -> ColumnTransformer:
    """Build a categorical/numeric preprocessor."""

    categorical_columns, numeric_columns = categorical_and_numeric_columns(frame)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=not dense_output,
                ),
            ),
        ]
    )

    if scale_numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=dense_output)),
            ]
        )
    else:
        numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_columns),
            ("num", numeric_pipeline, numeric_columns),
        ],
        remainder="drop",
        sparse_threshold=0.0 if dense_output else 0.3,
    )


def make_model_pipeline(model_name: str, train_frame: pd.DataFrame) -> Pipeline:
    """Create one of the planned baseline model pipelines."""

    name = model_name.lower()
    if name == "logistic_regression":
        model = LogisticRegression(
            max_iter=100,
            tol=1e-2,
            class_weight="balanced",
            solver="saga",
            random_state=42,
        )
        return Pipeline([("preprocess", make_preprocessor(train_frame, scale_numeric=True)), ("model", model)])

    if name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=60,
            max_depth=18,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        )
        return Pipeline([("preprocess", make_preprocessor(train_frame, scale_numeric=False)), ("model", model)])

    if name == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=160,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
        )
        return Pipeline([("preprocess", make_preprocessor(train_frame, scale_numeric=False)), ("model", model)])

    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=240,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            verbosity=-1,
        )
        return Pipeline([("preprocess", make_preprocessor(train_frame, scale_numeric=False)), ("model", model)])

    if name == "catboost":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=240,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        return Pipeline([("preprocess", make_preprocessor(train_frame, scale_numeric=False)), ("model", model)])

    if name == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            alpha=1e-4,
            batch_size=2048,
            learning_rate_init=1e-3,
            max_iter=20,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=42,
        )
        return Pipeline(
            [
                ("preprocess", make_preprocessor(train_frame, scale_numeric=True, dense_output=False)),
                ("model", model),
            ]
        )

    raise ValueError(f"Unknown model name: {model_name}")


def fit_and_evaluate(
    model_name: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[Pipeline, ModelRunResult]:
    """Train a model and evaluate it on a test frame."""

    pipeline = make_model_pipeline(model_name, train_frame)
    x_train, y_train = split_xy(train_frame)
    x_test, y_test = split_xy(test_frame)

    process = psutil.Process()
    monitor = PeakMemoryMonitor(process)
    monitor.start()
    train_started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    train_seconds = time.perf_counter() - train_started

    inference_started = time.perf_counter()
    scores = predict_scores(pipeline, x_test)
    inference_seconds = time.perf_counter() - inference_started
    peak_memory_mb = monitor.stop()

    result = ModelRunResult(
        model_name=model_name,
        train_rows=len(train_frame),
        test_rows=len(test_frame),
        train_seconds=train_seconds,
        inference_seconds=inference_seconds,
        peak_memory_mb=peak_memory_mb,
        metrics=binary_metrics(y_test, scores),
    )
    return pipeline, result


def evaluate_fitted_model(
    pipeline: Pipeline,
    model_name: str,
    train_rows: int,
    test_frame: pd.DataFrame,
) -> ModelRunResult:
    """Evaluate an already-fitted model with timing and memory stats."""

    x_test, y_test = split_xy(test_frame)
    process = psutil.Process()
    monitor = PeakMemoryMonitor(process)
    monitor.start()
    inference_started = time.perf_counter()
    scores = predict_scores(pipeline, x_test)
    inference_seconds = time.perf_counter() - inference_started
    peak_memory_mb = monitor.stop()

    return ModelRunResult(
        model_name=model_name,
        train_rows=train_rows,
        test_rows=len(test_frame),
        train_seconds=0.0,
        inference_seconds=inference_seconds,
        peak_memory_mb=peak_memory_mb,
        metrics=binary_metrics(y_test, scores),
    )


def predict_scores(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities or sigmoid decision scores."""

    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(features)[:, 1]
    elif hasattr(model, "decision_function"):
        raw_scores = model.decision_function(features)
        scores = 1 / (1 + np.exp(-raw_scores))
    else:
        scores = model.predict(features)
    return np.asarray(scores, dtype=float)


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    """Compute all required binary classification metrics."""

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "ece": float(expected_calibration_error(y_true, y_score)),
    }


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    """Compute binary expected calibration error."""

    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(y_score, bin_edges[1:-1], right=True)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_score[mask].mean())
    return float(ece)


def result_to_row(result: ModelRunResult, **extra: Any) -> dict[str, Any]:
    """Flatten a run result into one table row."""

    row = {
        **extra,
        "model": result.model_name,
        "train_rows": result.train_rows,
        "test_rows": result.test_rows,
        "training_time_sec": result.train_seconds,
        "inference_time_sec": result.inference_seconds,
        "memory_peak_mb": result.peak_memory_mb,
    }
    row.update(result.metrics)
    return row


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class PeakMemoryMonitor:
    """Sample process RSS during a model run."""

    def __init__(self, process: psutil.Process, interval_seconds: float = 0.1) -> None:
        self.process = process
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_bytes = process.memory_info().rss

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
        return self.peak_bytes / (1024 * 1024)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
            except psutil.Error:
                break
            time.sleep(self.interval_seconds)


def ensure_dense_if_needed(matrix):
    """Convert sparse matrices to dense arrays for estimators that need dense input."""

    if sparse.issparse(matrix):
        return matrix.toarray()
    return matrix
