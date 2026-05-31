"""Run Flower-compatible FedAvg/FedProx experiments on prepared city splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXTERNAL_VALIDATION_CITY, FEDERATED_CLIENTS, ensure_project_dirs  # noqa: E402
from src.config_loader import load_config, resolve_path  # noqa: E402
from src.dp_training import DPTrainingConfig  # noqa: E402
from src.fl_model import get_parameters  # noqa: E402
from src.fl_training import (  # noqa: E402
    ClientTensors,
    FlowerClient,
    evaluate_model,
    fit_preprocessor,
    frame_to_tensors,
    run_federated_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--algorithm", choices=["fedavg", "fedprox"], default="fedavg")
    parser.add_argument("--dp", action="store_true", help="Enable Opacus local DP training")
    parser.add_argument("--secure-aggregation", action="store_true", help="Enable secure aggregation simulation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_dirs()
    torch.manual_seed(int(config["project"]["random_seed"]))

    splits_dir = resolve_path(config, "data.splits_dir")
    tables_dir = resolve_path(config, "outputs.tables_dir")
    models_dir = resolve_path(config, "outputs.models_dir")
    logs_dir = resolve_path(config, "outputs.logs_dir")

    train_frames = {city: pd.read_parquet(split_path(splits_dir, city, "train")) for city in FEDERATED_CLIENTS}
    val_frames = {city: pd.read_parquet(split_path(splits_dir, city, "val")) for city in FEDERATED_CLIENTS}
    test_frames = {city: pd.read_parquet(split_path(splits_dir, city, "test")) for city in FEDERATED_CLIENTS}
    external_frame = pd.read_parquet(splits_dir / "external" / "los_angeles_external_test.parquet")

    preprocessor = fit_preprocessor(list(train_frames.values()))
    client_tensors = {}
    for city in FEDERATED_CLIENTS:
        x_train, y_train = frame_to_tensors(train_frames[city], preprocessor)
        x_val, y_val = frame_to_tensors(val_frames[city], preprocessor)
        x_test, y_test = frame_to_tensors(test_frames[city], preprocessor)
        client_tensors[city] = ClientTensors(city, x_train, y_train, x_val, y_val, x_test, y_test)

    input_dim = next(iter(client_tensors.values())).x_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fl_cfg = config["fl"]
    dp_cfg = config["dp"]
    secagg_cfg = config["secure_aggregation"]
    dp_config = DPTrainingConfig(
        enabled=bool(args.dp or dp_cfg["enabled"]),
        noise_multiplier=float(dp_cfg["noise_multiplier"]),
        max_grad_norm=float(dp_cfg["max_grad_norm"]),
        delta=float(dp_cfg["delta"]),
    )
    clients = {
        city: FlowerClient(
            tensors=tensors,
            input_dim=input_dim,
            hidden_dim=int(fl_cfg["hidden_dim"]),
            batch_size=int(fl_cfg["batch_size"]),
            learning_rate=float(fl_cfg["learning_rate"]),
            device=device,
        )
        for city, tensors in client_tensors.items()
    }

    model, history, summary = run_federated_experiment(
        clients,
        input_dim=input_dim,
        hidden_dim=int(fl_cfg["hidden_dim"]),
        rounds=int(fl_cfg["rounds"]),
        local_epochs=int(fl_cfg["local_epochs"]),
        algorithm=args.algorithm,
        fedprox_mu=float(fl_cfg["fedprox_mu"]),
        dp_config=dp_config,
        secure_aggregation=bool(args.secure_aggregation or secagg_cfg["enabled"]),
        secure_overhead_factor=float(secagg_cfg["overhead_factor"]),
        device=device,
    )

    x_external, y_external = frame_to_tensors(external_frame, preprocessor)
    external_metrics = evaluate_model(model, x_external, y_external, device)
    model_label = federated_model_label(args.algorithm, dp_config.enabled, bool(args.secure_aggregation or secagg_cfg["enabled"]))
    per_city_rows = []
    for city, tensors in client_tensors.items():
        per_city_rows.append(
            {
                "model": model_label,
                "algorithm": args.algorithm,
                "test_city": city,
                **evaluate_model(model, tensors.x_test, tensors.y_test, device),
            }
        )
    per_city_rows.append(
        {
            "model": model_label,
            "algorithm": args.algorithm,
            "test_city": EXTERNAL_VALIDATION_CITY,
            **external_metrics,
        }
    )
    macro_f1_table = make_macro_f1_table(model_label, per_city_rows, summary)

    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    history_frame = pd.DataFrame(history)
    if not history_frame.empty:
        history_frame.insert(0, "model", model_label)
    history_frame.to_csv(tables_dir / "federated_round_metrics.csv", index=False)
    pd.DataFrame(per_city_rows).to_csv(tables_dir / "federated_metrics.csv", index=False)
    macro_f1_table.to_csv(tables_dir / "step9_10_fedavg_vs_fedprox_macro_f1.csv", index=False)
    macro_f1_table.to_csv(tables_dir / "step11_12_fl_privacy_macro_f1.csv", index=False)
    torch.save(
        {"state_dict": model.state_dict(), "summary": summary, "parameters": get_parameters(model)},
        models_dir / f"{args.algorithm}_global_mlp.pt",
    )
    joblib.dump(preprocessor, models_dir / "federated_preprocessor.joblib")
    (logs_dir / "federated_run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if dp_config.enabled:
        (logs_dir / "dp_accounting.json").write_text(
            json.dumps({"epsilon": summary["epsilon"], "delta": dp_config.delta, "noise_multiplier": dp_config.noise_multiplier}, indent=2),
            encoding="utf-8",
        )
    print(f"Wrote federated metrics: {tables_dir / 'federated_metrics.csv'}")
    return 0


def split_path(splits_dir: Path, city: str, split_name: str) -> Path:
    return splits_dir / city / f"{city}_{split_name}.parquet"


def federated_model_label(algorithm: str, dp_enabled: bool, secure_aggregation: bool) -> str:
    base = "FedProx" if algorithm.lower() == "fedprox" else "FedAvg"
    if secure_aggregation:
        base += " + SecAgg"
    if dp_enabled:
        base += " + DP"
    return base


def make_macro_f1_table(model_label: str, rows: list[dict[str, object]], summary: dict[str, object]) -> pd.DataFrame:
    by_city = {row["test_city"]: row for row in rows}
    training_city_rows = [by_city[city] for city in FEDERATED_CLIENTS if city in by_city]
    global_macro_f1 = sum(float(row["macro_f1"]) for row in training_city_rows) / max(len(training_city_rows), 1)
    return pd.DataFrame(
        [
            {
                "Model": model_label,
                "Global Macro-F1": global_macro_f1,
                "NYC Macro-F1": float(by_city["nyc"]["macro_f1"]),
                "Chicago Macro-F1": float(by_city["chicago"]["macro_f1"]),
                "Boston Macro-F1": float(by_city["boston"]["macro_f1"]),
                "LA Macro-F1": float(by_city[EXTERNAL_VALIDATION_CITY]["macro_f1"]),
                "Rounds to converge": int(summary["rounds"]),
            }
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
