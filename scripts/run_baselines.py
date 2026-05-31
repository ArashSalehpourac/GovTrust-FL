"""Run centralized, local-only, and external-validation baseline experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXTERNAL_VALIDATION_CITY, FEDERATED_CLIENTS, ensure_project_dirs  # noqa: E402
from src.config_loader import load_config, resolve_path  # noqa: E402
from src.features import build_xy  # noqa: E402
from src.metrics import binary_classification_metrics, predict_scores  # noqa: E402
from src.models import make_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", default=None, help="Override model.type from config")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_dirs()

    splits_dir = resolve_path(config, "data.splits_dir")
    tables_dir = resolve_path(config, "outputs.tables_dir")
    models_dir = resolve_path(config, "outputs.models_dir")
    logs_dir = resolve_path(config, "outputs.logs_dir")
    model_name = args.model or config["model"]["type"]

    train_frames = {city: pd.read_parquet(split_path(splits_dir, city, "train")) for city in FEDERATED_CLIENTS}
    test_frames = {city: pd.read_parquet(split_path(splits_dir, city, "test")) for city in FEDERATED_CLIENTS}
    external_frame = pd.read_parquet(splits_dir / "external" / "los_angeles_external_test.parquet")

    rows = []
    run_log = {"model_type": model_name, "runs": []}

    for city in FEDERATED_CLIENTS:
        model, elapsed = fit_model(model_name, train_frames[city])
        model_path = models_dir / f"local_only_{city}_{model_name}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        rows.append(evaluate_row(model, "local_only", city, city, train_frames[city], test_frames[city], elapsed))
        rows.append(
            evaluate_row(
                model,
                "local_only_external",
                city,
                EXTERNAL_VALIDATION_CITY,
                train_frames[city],
                external_frame,
                elapsed,
            )
        )
        run_log["runs"].append({"setting": "local_only", "train_city": city, "model_path": str(model_path)})

    pooled_train = pd.concat([train_frames[city] for city in FEDERATED_CLIENTS], ignore_index=True)
    pooled_test = pd.concat([test_frames[city] for city in FEDERATED_CLIENTS], ignore_index=True)
    centralized, elapsed = fit_model(model_name, pooled_train)
    centralized_path = models_dir / f"centralized_{model_name}.joblib"
    joblib.dump(centralized, centralized_path)
    rows.append(evaluate_row(centralized, "centralized", "pooled", "pooled_training_cities", pooled_train, pooled_test, elapsed))
    rows.append(
        evaluate_row(
            centralized,
            "centralized_external",
            "pooled",
            EXTERNAL_VALIDATION_CITY,
            pooled_train,
            external_frame,
            elapsed,
        )
    )
    run_log["runs"].append({"setting": "centralized", "train_city": "pooled", "model_path": str(centralized_path)})

    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tables_dir / "baseline_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    (logs_dir / "baseline_run.json").write_text(json.dumps(run_log, indent=2), encoding="utf-8")
    print(f"Wrote baseline metrics: {metrics_path}")
    return 0


def fit_model(model_name: str, train_frame: pd.DataFrame):
    x_train, y_train = build_xy(train_frame)
    model = make_pipeline(model_name)
    started = time.perf_counter()
    model.fit(x_train, y_train)
    return model, time.perf_counter() - started


def evaluate_row(
    model,
    setting: str,
    train_city: str,
    test_city: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    training_time_sec: float,
) -> dict[str, object]:
    x_test, y_test = build_xy(test_frame)
    started = time.perf_counter()
    scores = predict_scores(model, x_test)
    inference_time_sec = time.perf_counter() - started
    metrics = binary_classification_metrics(y_test, scores)
    return {
        "setting": setting,
        "train_city": train_city,
        "test_city": test_city,
        "train_rows": len(train_frame),
        "test_rows": len(test_frame),
        "training_time_sec": training_time_sec,
        "inference_time_sec": inference_time_sec,
        **metrics,
    }


def split_path(splits_dir: Path, city: str, split_name: str) -> Path:
    return splits_dir / city / f"{city}_{split_name}.parquet"


if __name__ == "__main__":
    raise SystemExit(main())
