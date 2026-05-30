"""Train Step 8 local-only LightGBM models and cross-city evaluations."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_models import (  # noqa: E402
    evaluate_fitted_model,
    fit_and_evaluate,
    result_to_row,
    write_json,
)


SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_MODELS_DIR = PROJECT_ROOT / "results" / "models"

FEDERATED_CITIES = ("nyc", "chicago", "boston")
MODEL_NAME = "lightgbm"

LONG_RESULTS_PATH = RESULTS_TABLES_DIR / "step8_local_only_long_metrics.csv"
AUROC_MATRIX_PATH = RESULTS_TABLES_DIR / "step8_local_only_auroc_matrix.csv"
MACRO_F1_MATRIX_PATH = RESULTS_TABLES_DIR / "step8_local_only_macro_f1_matrix.csv"
MANIFEST_PATH = RESULTS_TABLES_DIR / "step8_local_only_manifest.json"


def split_path(city: str, split_name: str) -> Path:
    return SPLITS_DIR / city / f"{city}_{split_name}.parquet"


def main() -> int:
    train_frames = {
        city: pd.read_parquet(split_path(city, "train"))
        for city in FEDERATED_CITIES
    }
    test_frames = {
        city: pd.read_parquet(split_path(city, "test"))
        for city in FEDERATED_CITIES
    }
    test_frames["los_angeles"] = pd.read_parquet(
        SPLITS_DIR / "external" / "los_angeles_external_test.parquet"
    )

    long_rows = []
    fitted_models = {}
    for train_city, train_frame in train_frames.items():
        print(f"Training {train_city}-only {MODEL_NAME}", flush=True)
        model, own_result = fit_and_evaluate(MODEL_NAME, train_frame, test_frames[train_city])
        fitted_models[train_city] = model
        long_rows.append(
            result_to_row(
                own_result,
                setting="local_only",
                train_city=train_city,
                test_city=train_city,
            )
        )

        model_path = RESULTS_MODELS_DIR / f"step8_{train_city}_only_{MODEL_NAME}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        for test_city, test_frame in test_frames.items():
            if test_city == train_city:
                continue
            result = evaluate_fitted_model(
                model,
                MODEL_NAME,
                len(train_frame),
                test_frame,
            )
            long_rows.append(
                result_to_row(
                    result,
                    setting="local_only",
                    train_city=train_city,
                    test_city=test_city,
                )
            )
            print(
                f"  {train_city} -> {test_city}: AUROC={result.metrics['auroc']:.4f}, "
                f"macro-F1={result.metrics['macro_f1']:.4f}",
                flush=True,
            )

    centralized_model_path = RESULTS_MODELS_DIR / "step7_centralized_lightgbm.joblib"
    if centralized_model_path.exists():
        centralized_model = joblib.load(centralized_model_path)
        centralized_train = pd.read_parquet(SPLITS_DIR / "centralized" / "centralized_train.parquet")
        for test_city, test_frame in test_frames.items():
            result = evaluate_fitted_model(
                centralized_model,
                MODEL_NAME,
                len(centralized_train),
                test_frame,
            )
            long_rows.append(
                result_to_row(
                    result,
                    setting="centralized",
                    train_city="centralized",
                    test_city=test_city,
                )
            )

    long_results = pd.DataFrame(long_rows)
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    long_results.to_csv(LONG_RESULTS_PATH, index=False)

    write_matrix(long_results, "auroc", AUROC_MATRIX_PATH)
    write_matrix(long_results, "macro_f1", MACRO_F1_MATRIX_PATH)

    write_json(
        MANIFEST_PATH,
        {
            "model": MODEL_NAME,
            "long_results_path": str(LONG_RESULTS_PATH),
            "auroc_matrix_path": str(AUROC_MATRIX_PATH),
            "macro_f1_matrix_path": str(MACRO_F1_MATRIX_PATH),
            "notes": "Local-only city models are LightGBM models trained separately on each city train split and evaluated on all city test splits plus Los Angeles external test.",
        },
    )
    print(f"Wrote long results: {LONG_RESULTS_PATH}", flush=True)
    print(f"Wrote AUROC matrix: {AUROC_MATRIX_PATH}", flush=True)
    return 0


def write_matrix(results: pd.DataFrame, metric: str, output_path: Path) -> None:
    matrix = (
        results.pivot_table(
            index="train_city",
            columns="test_city",
            values=metric,
            aggfunc="first",
        )
        .reindex(index=["nyc", "chicago", "boston", "centralized"])
        .rename_axis(index="Train city", columns=None)
    )
    preferred_columns = ["nyc", "chicago", "boston", "los_angeles"]
    matrix = matrix[[column for column in preferred_columns if column in matrix.columns]]
    matrix.to_csv(output_path)


if __name__ == "__main__":
    raise SystemExit(main())

