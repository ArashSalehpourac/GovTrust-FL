"""Train Step 9/10 Flower-compatible PyTorch MLP FL models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fl_pytorch import (  # noqa: E402
    TorchFlowerClient,
    build_dense_preprocessor,
    clean_feature_frame,
    communication_cost,
    evaluate_model,
    prepare_client_data,
    run_fl_experiment,
    sample_frame,
    save_preprocessor,
    save_torch_model,
    transform_to_tensors,
    write_json,
)


FEDERATED_CITIES = ("nyc", "chicago", "boston")
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_MODELS_DIR = PROJECT_ROOT / "results" / "models"


def split_path(city: str, split_name: str) -> Path:
    return SPLITS_DIR / city / f"{city}_{split_name}.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--proximal-mu", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--max-train-per-client", type=int, default=50_000)
    parser.add_argument("--max-val-per-client", type=int, default=20_000)
    parser.add_argument("--max-test-per-city", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    city_frames = load_city_frames(args)
    train_for_preprocessor = pd.concat(
        [city_frames[city]["train"] for city in FEDERATED_CITIES],
        ignore_index=True,
    )
    print(
        f"Fitting FL preprocessor on {len(train_for_preprocessor):,} sampled federated train rows",
        flush=True,
    )
    preprocessor = build_dense_preprocessor(train_for_preprocessor)
    preprocessor.fit(clean_feature_frame(train_for_preprocessor))

    client_data = prepare_client_data(city_frames, preprocessor)
    input_dim = client_data["nyc"].x_train.shape[1]
    print(f"Prepared FL tensors with input_dim={input_dim}", flush=True)

    fl_clients = {
        city: TorchFlowerClient(
            client_data=data,
            input_dim=input_dim,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
        )
        for city, data in client_data.items()
    }

    external_frame = pd.read_parquet(SPLITS_DIR / "external" / "los_angeles_external_test.parquet")
    external_frame = sample_frame(external_frame, args.max_test_per_city, args.seed)
    x_la, y_la = transform_to_tensors(external_frame, preprocessor)

    all_rows = []
    all_histories = []
    summaries = []

    for algorithm in ("fedavg", "fedprox"):
        print(f"Training {algorithm.upper()} MLP", flush=True)
        model, history, summary = run_fl_experiment(
            algorithm=algorithm,
            clients=fl_clients,
            input_dim=input_dim,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            proximal_mu=args.proximal_mu,
            patience=args.patience,
            min_delta=args.min_delta,
            device=device,
        )
        summaries.append(summary)
        for row in history:
            all_histories.append(row)

        model_path = RESULTS_MODELS_DIR / f"step9_10_{algorithm}_mlp.pt"
        save_torch_model(model, model_path, metadata=summary)

        city_metrics = {}
        for city, data in client_data.items():
            metrics = evaluate_model(model, data.x_test, data.y_test, device)
            city_metrics[city] = metrics
            all_rows.append(
                {
                    "model": algorithm,
                    "test_city": city,
                    "rounds_to_converge": summary["best_round"],
                    "rounds_completed": summary["rounds_completed"],
                    "train_rows": sum(len(client_data[source_city].y_train) for source_city in FEDERATED_CITIES),
                    "test_rows": len(data.y_test),
                    "training_time_sec": summary["training_time_sec"],
                    "memory_peak_mb": summary["memory_peak_mb"],
                    "model_size_bytes": summary["model_size_bytes"],
                    "total_communication_bytes": summary["total_bytes"],
                    **metrics,
                }
            )

        la_metrics = evaluate_model(model, x_la, y_la, device)
        all_rows.append(
            {
                "model": algorithm,
                "test_city": "los_angeles",
                "rounds_to_converge": summary["best_round"],
                "rounds_completed": summary["rounds_completed"],
                "train_rows": sum(len(client_data[source_city].y_train) for source_city in FEDERATED_CITIES),
                "test_rows": len(y_la),
                "training_time_sec": summary["training_time_sec"],
                "memory_peak_mb": summary["memory_peak_mb"],
                "model_size_bytes": summary["model_size_bytes"],
                "total_communication_bytes": summary["total_bytes"],
                **la_metrics,
            }
        )

        global_macro_f1 = float(
            sum(city_metrics[city]["macro_f1"] * len(client_data[city].y_test) for city in FEDERATED_CITIES)
            / sum(len(client_data[city].y_test) for city in FEDERATED_CITIES)
        )
        print(
            f"  {algorithm}: global macro-F1={global_macro_f1:.4f}, "
            f"NYC={city_metrics['nyc']['macro_f1']:.4f}, "
            f"Chicago={city_metrics['chicago']['macro_f1']:.4f}, "
            f"Boston={city_metrics['boston']['macro_f1']:.4f}, "
            f"LA={la_metrics['macro_f1']:.4f}, "
            f"rounds={summary['rounds_completed']}",
            flush=True,
        )

    preprocessor_path = RESULTS_MODELS_DIR / "step9_10_fl_preprocessor.joblib"
    save_preprocessor(preprocessor, preprocessor_path)

    long_results = pd.DataFrame(all_rows)
    history = pd.DataFrame(all_histories)
    long_path = RESULTS_TABLES_DIR / "step9_10_fl_mlp_long_metrics.csv"
    history_path = RESULTS_TABLES_DIR / "step9_10_fl_mlp_round_history.csv"
    comparison_path = RESULTS_TABLES_DIR / "step9_10_fedavg_vs_fedprox_macro_f1.csv"
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    long_results.to_csv(long_path, index=False)
    history.to_csv(history_path, index=False)
    write_comparison_table(long_results, comparison_path)

    manifest_path = RESULTS_TABLES_DIR / "step9_10_fl_mlp_manifest.json"
    write_json(
        manifest_path,
        {
            "framework": "Flower-compatible NumPyClient local simulation with PyTorch MLP.",
            "clients": list(FEDERATED_CITIES),
            "external_test": "los_angeles",
            "rounds_requested": args.rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "early_stopping": "Mean validation macro-F1 across NYC, Chicago, and Boston.",
            "max_train_per_client": args.max_train_per_client,
            "max_val_per_client": args.max_val_per_client,
            "max_test_per_city": args.max_test_per_city,
            "proximal_mu": args.proximal_mu,
            "preprocessor_path": str(preprocessor_path),
            "results_path": str(long_path),
            "history_path": str(history_path),
            "comparison_path": str(comparison_path),
            "summaries": summaries,
        },
    )
    print(f"Wrote FL metrics: {long_path}", flush=True)
    print(f"Wrote FL comparison table: {comparison_path}", flush=True)
    print(f"Wrote FL manifest: {manifest_path}", flush=True)
    # Keep communication_cost imported as an explicit reminder that the summary
    # uses the same model-size formula as later privacy-overhead steps.
    _ = communication_cost
    return 0


def load_city_frames(args: argparse.Namespace) -> dict[str, dict[str, pd.DataFrame]]:
    city_frames = {}
    for city in FEDERATED_CITIES:
        city_frames[city] = {
            "train": sample_frame(pd.read_parquet(split_path(city, "train")), args.max_train_per_client, args.seed),
            "val": sample_frame(pd.read_parquet(split_path(city, "val")), args.max_val_per_client, args.seed),
            "test": sample_frame(pd.read_parquet(split_path(city, "test")), args.max_test_per_city, args.seed),
        }
        print(
            f"{city}: train={len(city_frames[city]['train']):,}, "
            f"val={len(city_frames[city]['val']):,}, "
            f"test={len(city_frames[city]['test']):,}",
            flush=True,
        )
    return city_frames


def write_comparison_table(long_results: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for model_name, model_frame in long_results.groupby("model", sort=False):
        by_city = model_frame.set_index("test_city")
        federated = model_frame[model_frame["test_city"].isin(FEDERATED_CITIES)]
        global_macro_f1 = float(np.average(federated["macro_f1"], weights=federated["test_rows"]))
        rows.append(
            {
                "Model": model_name,
                "Global Macro-F1": global_macro_f1,
                "NYC Macro-F1": by_city.loc["nyc", "macro_f1"],
                "Chicago Macro-F1": by_city.loc["chicago", "macro_f1"],
                "Boston Macro-F1": by_city.loc["boston", "macro_f1"],
                "LA Macro-F1": by_city.loc["los_angeles", "macro_f1"],
                "Rounds to converge": int(by_city.iloc[0]["rounds_to_converge"]),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
