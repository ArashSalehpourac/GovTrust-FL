"""Train Step 7 centralized baseline models."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_models import fit_and_evaluate, result_to_row, write_json  # noqa: E402


SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_MODELS_DIR = PROJECT_ROOT / "results" / "models"

TRAIN_PATH = SPLITS_DIR / "centralized" / "centralized_train.parquet"
TEST_PATH = SPLITS_DIR / "centralized" / "centralized_test.parquet"
EXTERNAL_TEST_PATH = SPLITS_DIR / "external" / "los_angeles_external_test.parquet"

RESULTS_PATH = RESULTS_TABLES_DIR / "step7_centralized_baselines.csv"
EXTERNAL_RESULTS_PATH = RESULTS_TABLES_DIR / "step7_centralized_external_la.csv"
MANIFEST_PATH = RESULTS_TABLES_DIR / "step7_centralized_baselines_manifest.json"

MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
    "mlp",
]


def main() -> int:
    print(f"Loading train: {TRAIN_PATH}", flush=True)
    train_frame = pd.read_parquet(TRAIN_PATH)
    test_frame = pd.read_parquet(TEST_PATH)
    external_frame = pd.read_parquet(EXTERNAL_TEST_PATH)

    rows = []
    external_rows = []
    for model_name in MODELS:
        print(f"Training centralized {model_name}", flush=True)
        model, result = fit_and_evaluate(model_name, train_frame, test_frame)
        rows.append(result_to_row(result, setting="centralized", test_city="pooled_test"))

        model_path = RESULTS_MODELS_DIR / f"step7_centralized_{model_name}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        external_result = fit_external_eval(model, model_name, result.train_rows, external_frame)
        external_rows.append(
            result_to_row(external_result, setting="centralized_external", test_city="los_angeles")
        )
        print(
            f"  pooled AUROC={result.metrics['auroc']:.4f}; "
            f"LA AUROC={external_result.metrics['auroc']:.4f}; "
            f"train_sec={result.train_seconds:.1f}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    external_results = pd.DataFrame(external_rows)
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    external_results.to_csv(EXTERNAL_RESULTS_PATH, index=False)

    write_json(
        MANIFEST_PATH,
        {
            "train_path": str(TRAIN_PATH),
            "test_path": str(TEST_PATH),
            "external_test_path": str(EXTERNAL_TEST_PATH),
            "models": MODELS,
            "results_path": str(RESULTS_PATH),
            "external_results_path": str(EXTERNAL_RESULTS_PATH),
            "notes": "Centralized models train on pooled NYC+Chicago+Boston train splits, evaluate on pooled held-out test and LA external test.",
        },
    )
    print(f"Wrote results: {RESULTS_PATH}", flush=True)
    print(f"Wrote external results: {EXTERNAL_RESULTS_PATH}", flush=True)
    return 0


def fit_external_eval(model, model_name: str, train_rows: int, external_frame: pd.DataFrame):
    from src.experiment_models import evaluate_fitted_model

    return evaluate_fitted_model(model, model_name, train_rows, external_frame)


if __name__ == "__main__":
    raise SystemExit(main())

