"""Run first privacy, attack, fairness, calibration, PEDI, and TAI tables."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from opacus.accountants.utils import get_noise_multiplier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment_models import predict_scores as predict_sklearn_scores  # noqa: E402
from src.experiment_models import split_xy  # noqa: E402
from src.fl_pytorch import (  # noqa: E402
    FEDERATED_CITIES,
    TorchFlowerClient,
    binary_metrics,
    build_dense_preprocessor,
    clean_feature_frame,
    evaluate_model,
    expected_calibration_error,
    prepare_client_data,
    run_fl_experiment,
    sample_frame,
    save_preprocessor,
    save_torch_model,
    transform_to_tensors,
)
from src.privacy import attack_advantage, membership_inference_auc  # noqa: E402


SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_MODELS_DIR = PROJECT_ROOT / "results" / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--proximal-mu", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--max-train-per-client", type=int, default=10_000)
    parser.add_argument("--max-val-per-client", type=int, default=5_000)
    parser.add_argument("--max-test-per-city", type=int, default=10_000)
    parser.add_argument("--mia-sample", type=int, default=8_000)
    parser.add_argument("--xai-sample", type=int, default=2_000)
    parser.add_argument("--xai-top-k", type=int, default=30)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--dp-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_path(city: str, split_name: str) -> Path:
    return SPLITS_DIR / city / f"{city}_{split_name}.parquet"


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    city_frames = load_city_frames(args)
    train_for_preprocessor = pd.concat([city_frames[city]["train"] for city in FEDERATED_CITIES], ignore_index=True)
    preprocessor = build_dense_preprocessor(train_for_preprocessor)
    preprocessor.fit(clean_feature_frame(train_for_preprocessor))
    save_preprocessor(preprocessor, RESULTS_MODELS_DIR / "step11_19_fl_preprocessor.joblib")

    client_data = prepare_client_data(city_frames, preprocessor)
    input_dim = client_data["nyc"].x_train.shape[1]
    external_frame = sample_frame(
        pd.read_parquet(SPLITS_DIR / "external" / "los_angeles_external_test.parquet"),
        args.max_test_per_city,
        args.seed,
    )
    x_la, y_la = transform_to_tensors(external_frame, preprocessor)

    sample_rate = min(1.0, args.batch_size / max(args.max_train_per_client, 1))
    steps = args.rounds * args.local_epochs * math.ceil(args.max_train_per_client / args.batch_size)
    dp_levels = {
        "Weak DP": 8.0,
        "Medium DP": 3.0,
        "Strong DP": 1.0,
    }
    dp_noise = {
        label: float(
            get_noise_multiplier(
                target_epsilon=epsilon,
                target_delta=args.dp_delta,
                sample_rate=sample_rate,
                steps=steps,
            )
        )
        for label, epsilon in dp_levels.items()
    }

    variants = [
        {
            "model_label": "FedAvg",
            "algorithm": "fedavg",
            "privacy_level": "None",
            "target_epsilon": np.nan,
            "noise_multiplier": 0.0,
            "secure_aggregation": False,
            "seed_offset": 0,
        },
        {
            "model_label": "FedProx",
            "algorithm": "fedprox",
            "privacy_level": "None",
            "target_epsilon": np.nan,
            "noise_multiplier": 0.0,
            "secure_aggregation": False,
            "seed_offset": 100,
        },
        {
            "model_label": "FedAvg + SecAgg",
            "algorithm": "fedavg",
            "privacy_level": "Secure aggregation",
            "target_epsilon": np.nan,
            "noise_multiplier": 0.0,
            "secure_aggregation": True,
            "seed_offset": 0,
        },
    ]
    for label, epsilon in dp_levels.items():
        variants.append(
            {
                "model_label": f"FedAvg + DP eps={int(epsilon)}",
                "algorithm": "fedavg",
                "privacy_level": label,
                "target_epsilon": epsilon,
                "noise_multiplier": dp_noise[label],
                "secure_aggregation": False,
                "seed_offset": int(epsilon * 10),
            }
        )
    variants.append(
        {
            "model_label": "FedProx + DP eps=3",
            "algorithm": "fedprox",
            "privacy_level": "Medium DP",
            "target_epsilon": 3.0,
            "noise_multiplier": dp_noise["Medium DP"],
            "secure_aggregation": False,
            "seed_offset": 130,
        }
    )

    all_metric_rows = []
    all_history_rows = []
    summaries = []
    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    trained_models = {}

    for index, variant in enumerate(variants, start=1):
        print(f"Training {variant['model_label']} ({index}/{len(variants)})", flush=True)
        variant_seed = args.seed + variant["seed_offset"]
        torch.manual_seed(variant_seed)
        np.random.seed(variant_seed)
        clients = build_clients(client_data, input_dim, args, device)
        model, history, summary = run_fl_experiment(
            algorithm=variant["algorithm"],
            clients=clients,
            input_dim=input_dim,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            proximal_mu=args.proximal_mu,
            patience=args.patience,
            min_delta=args.min_delta,
            device=device,
            dp_noise_multiplier=variant["noise_multiplier"],
            dp_clip_norm=args.dp_clip_norm,
            secure_aggregation=variant["secure_aggregation"],
            seed=variant_seed,
        )
        summary.update(variant)
        summaries.append(summary)
        trained_models[variant["model_label"]] = model
        save_torch_model(
            model,
            RESULTS_MODELS_DIR / f"step11_19_{slug(variant['model_label'])}.pt",
            metadata=summary,
        )
        for row in history:
            row.update({"model": variant["model_label"], "privacy_level": variant["privacy_level"]})
            all_history_rows.append(row)

        predictions[variant["model_label"]] = {}
        city_metrics = {}
        for city, data in client_data.items():
            metrics = evaluate_model(model, data.x_test, data.y_test, device)
            scores = predict_torch_scores(model, data.x_test, device)
            predictions[variant["model_label"]][city] = {
                "y_true": data.y_test.numpy().astype(int),
                "score": scores,
            }
            city_metrics[city] = metrics
            all_metric_rows.append(
                metric_row(variant, summary, city, len(data.y_test), sum(len(d.y_train) for d in client_data.values()), metrics)
            )

        la_metrics = evaluate_model(model, x_la, y_la, device)
        predictions[variant["model_label"]]["los_angeles"] = {
            "y_true": y_la.numpy().astype(int),
            "score": predict_torch_scores(model, x_la, device),
        }
        all_metric_rows.append(
            metric_row(variant, summary, "los_angeles", len(y_la), sum(len(d.y_train) for d in client_data.values()), la_metrics)
        )

        global_macro_f1 = weighted_global_macro_f1(city_metrics, client_data)
        print(
            f"  {variant['model_label']}: global macro-F1={global_macro_f1:.4f}, "
            f"LA={la_metrics['macro_f1']:.4f}, rounds={summary['rounds_completed']}",
            flush=True,
        )

    long_metrics = pd.DataFrame(all_metric_rows)
    history_frame = pd.DataFrame(all_history_rows)
    macro_table = write_macro_f1_table(long_metrics, RESULTS_TABLES_DIR / "step11_12_fl_privacy_macro_f1.csv")

    calibration = calibration_tables(predictions)
    calibration["by_city"].to_csv(RESULTS_TABLES_DIR / "step17_calibration_by_city.csv", index=False)
    calibration["reliability"].to_csv(RESULTS_TABLES_DIR / "step17_reliability_bins.csv", index=False)

    fairness = fairness_summary(predictions, city_frames, external_frame)
    fairness.to_csv(RESULTS_TABLES_DIR / "step16_fairness_summary.csv", index=False)

    xai_tables = explanation_tables(trained_models, client_data, preprocessor, device, args)
    xai_tables["importance"].to_csv(RESULTS_TABLES_DIR / "step15_gradient_importance.csv", index=False)
    xai_tables["pedi"].to_csv(RESULTS_TABLES_DIR / "step15_pedi_gradient_drift.csv", index=False)

    mia = membership_inference_table(predictions, trained_models, client_data, city_frames, args, device)
    mia.to_csv(RESULTS_TABLES_DIR / "step13_membership_inference.csv", index=False)

    tradeoff = privacy_tradeoff_table(long_metrics, macro_table, summaries, calibration["by_city"], fairness, xai_tables["pedi"])
    tradeoff.to_csv(RESULTS_TABLES_DIR / "step11_dp_privacy_tradeoffs.csv", index=False)
    secagg = secure_aggregation_table(summaries, macro_table)
    secagg.to_csv(RESULTS_TABLES_DIR / "step12_secure_aggregation_overhead.csv", index=False)
    resources = resource_table(summaries)
    resources.to_csv(RESULTS_TABLES_DIR / "step18_resource_communication.csv", index=False)
    tai = tai_scorecard(macro_table, summaries, calibration["by_city"], fairness, xai_tables["pedi"], mia)
    tai.to_csv(RESULTS_TABLES_DIR / "step19_tai_scorecard.csv", index=False)

    long_metrics.to_csv(RESULTS_TABLES_DIR / "step11_19_fl_privacy_long_metrics.csv", index=False)
    history_frame.to_csv(RESULTS_TABLES_DIR / "step11_19_fl_privacy_round_history.csv", index=False)
    write_json(
        RESULTS_TABLES_DIR / "step11_19_privacy_scorecard_manifest.json",
        {
            "framework": "Flower-compatible PyTorch MLP with update-level DP simulation calibrated by Opacus.",
            "clients": list(FEDERATED_CITIES),
            "external_test": "los_angeles",
            "rounds_requested": args.rounds,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "max_train_per_client": args.max_train_per_client,
            "max_val_per_client": args.max_val_per_client,
            "max_test_per_city": args.max_test_per_city,
            "dp_delta": args.dp_delta,
            "dp_clip_norm": args.dp_clip_norm,
            "dp_noise_multipliers": dp_noise,
            "outputs": {
                "macro_f1": str(RESULTS_TABLES_DIR / "step11_12_fl_privacy_macro_f1.csv"),
                "dp_tradeoffs": str(RESULTS_TABLES_DIR / "step11_dp_privacy_tradeoffs.csv"),
                "secure_aggregation": str(RESULTS_TABLES_DIR / "step12_secure_aggregation_overhead.csv"),
                "membership_inference": str(RESULTS_TABLES_DIR / "step13_membership_inference.csv"),
                "fairness": str(RESULTS_TABLES_DIR / "step16_fairness_summary.csv"),
                "calibration": str(RESULTS_TABLES_DIR / "step17_calibration_by_city.csv"),
                "resources": str(RESULTS_TABLES_DIR / "step18_resource_communication.csv"),
                "tai_score": str(RESULTS_TABLES_DIR / "step19_tai_scorecard.csv"),
            },
            "summaries": summaries,
        },
    )
    print(f"Wrote macro-F1 table: {RESULTS_TABLES_DIR / 'step11_12_fl_privacy_macro_f1.csv'}", flush=True)
    print(f"Wrote TAI scorecard: {RESULTS_TABLES_DIR / 'step19_tai_scorecard.csv'}", flush=True)
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
            f"val={len(city_frames[city]['val']):,}, test={len(city_frames[city]['test']):,}",
            flush=True,
        )
    return city_frames


def build_clients(client_data, input_dim: int, args: argparse.Namespace, device: torch.device):
    return {
        city: TorchFlowerClient(
            client_data=data,
            input_dim=input_dim,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
        )
        for city, data in client_data.items()
    }


def metric_row(variant, summary, test_city: str, test_rows: int, train_rows: int, metrics: dict[str, float]) -> dict:
    return {
        "model": variant["model_label"],
        "algorithm": variant["algorithm"],
        "privacy_level": variant["privacy_level"],
        "target_epsilon": variant["target_epsilon"],
        "noise_multiplier": variant["noise_multiplier"],
        "secure_aggregation": variant["secure_aggregation"],
        "test_city": test_city,
        "rounds_to_converge": summary["best_round"],
        "rounds_completed": summary["rounds_completed"],
        "train_rows": train_rows,
        "test_rows": test_rows,
        "training_time_sec": summary["training_time_sec"],
        "memory_peak_mb": summary["memory_peak_mb"],
        "model_size_bytes": summary["model_size_bytes"],
        "total_communication_bytes": summary["total_bytes_with_overheads"],
        **metrics,
    }


def weighted_global_macro_f1(city_metrics, client_data) -> float:
    return float(
        sum(city_metrics[city]["macro_f1"] * len(client_data[city].y_test) for city in FEDERATED_CITIES)
        / sum(len(client_data[city].y_test) for city in FEDERATED_CITIES)
    )


def write_macro_f1_table(long_metrics: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    rows = []
    for model, frame in long_metrics.groupby("model", sort=False):
        by_city = frame.set_index("test_city")
        federated = frame[frame["test_city"].isin(FEDERATED_CITIES)]
        rows.append(
            {
                "Model": model,
                "Global Macro-F1": float(np.average(federated["macro_f1"], weights=federated["test_rows"])),
                "NYC Macro-F1": float(by_city.loc["nyc", "macro_f1"]),
                "Chicago Macro-F1": float(by_city.loc["chicago", "macro_f1"]),
                "Boston Macro-F1": float(by_city.loc["boston", "macro_f1"]),
                "LA Macro-F1": float(by_city.loc["los_angeles", "macro_f1"]),
                "Rounds to converge": int(by_city.iloc[0]["rounds_to_converge"]),
            }
        )
    table = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


@torch.no_grad()
def predict_torch_scores(model, features: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    scores = []
    for start in range(0, len(features), 4096):
        batch = features[start : start + 4096].to(device)
        scores.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(scores)


def calibration_tables(predictions):
    by_city_rows = []
    reliability_rows = []
    for model, city_predictions in predictions.items():
        for city, values in city_predictions.items():
            y_true = values["y_true"]
            scores = values["score"]
            by_city_rows.append(
                {
                    "model": model,
                    "city": city,
                    "brier": float(np.mean((scores - y_true) ** 2)),
                    "ece": expected_calibration_error(y_true, scores),
                    "n": len(y_true),
                }
            )
            bins = np.linspace(0.0, 1.0, 11)
            bin_ids = np.digitize(scores, bins[1:-1], right=True)
            for bin_id in range(10):
                mask = bin_ids == bin_id
                if not np.any(mask):
                    continue
                reliability_rows.append(
                    {
                        "model": model,
                        "city": city,
                        "bin": bin_id,
                        "mean_predicted_probability": float(scores[mask].mean()),
                        "observed_delay_rate": float(y_true[mask].mean()),
                        "n": int(mask.sum()),
                    }
                )
    return {"by_city": pd.DataFrame(by_city_rows), "reliability": pd.DataFrame(reliability_rows)}


def fairness_summary(predictions, city_frames, external_frame) -> pd.DataFrame:
    frame_by_city = {city: city_frames[city]["test"].reset_index(drop=True) for city in FEDERATED_CITIES}
    frame_by_city["los_angeles"] = external_frame.reset_index(drop=True)
    rows = []
    for model, city_predictions in predictions.items():
        combined_frames = []
        combined_true = []
        combined_scores = []
        for city, values in city_predictions.items():
            combined_frames.append(frame_by_city[city])
            combined_true.append(values["y_true"])
            combined_scores.append(values["score"])
        frame = pd.concat(combined_frames, ignore_index=True)
        y_true = np.concatenate(combined_true)
        scores = np.concatenate(combined_scores)
        for group_column in ("city", "category", "area"):
            rows.append(group_gap_row(model, frame, y_true, scores, group_column))
    return pd.DataFrame(rows)


def group_gap_row(model: str, frame: pd.DataFrame, y_true: np.ndarray, scores: np.ndarray, group_column: str) -> dict:
    working = frame[[group_column]].copy()
    working["_y_true"] = y_true
    working["_score"] = scores
    rows = []
    for group, group_frame in working.groupby(group_column, dropna=False):
        if len(group_frame) < 100:
            continue
        group_y = group_frame["_y_true"].to_numpy(dtype=int)
        group_score = group_frame["_score"].to_numpy(dtype=float)
        group_pred = (group_score >= 0.5).astype(int)
        tp = int(((group_y == 1) & (group_pred == 1)).sum())
        tn = int(((group_y == 0) & (group_pred == 0)).sum())
        fp = int(((group_y == 0) & (group_pred == 1)).sum())
        fn = int(((group_y == 1) & (group_pred == 0)).sum())
        metrics = binary_metrics(group_y, group_score)
        rows.append(
            {
                "group": group,
                "n": len(group_frame),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "macro_f1": metrics["macro_f1"],
                "fpr": fp / max(fp + tn, 1),
                "fnr": fn / max(fn + tp, 1),
                "ece": metrics["ece"],
            }
        )
    group_metrics = pd.DataFrame(rows)
    if group_metrics.empty:
        return {
            "model": model,
            "group_column": group_column,
            "groups": 0,
            "precision_gap": np.nan,
            "recall_gap": np.nan,
            "macro_f1_gap": np.nan,
            "false_positive_rate_gap": np.nan,
            "false_negative_rate_gap": np.nan,
            "equal_opportunity_difference": np.nan,
            "calibration_ece_gap": np.nan,
        }
    return {
        "model": model,
        "group_column": group_column,
        "groups": len(group_metrics),
        "precision_gap": gap(group_metrics["precision"]),
        "recall_gap": gap(group_metrics["recall"]),
        "macro_f1_gap": gap(group_metrics["macro_f1"]),
        "false_positive_rate_gap": gap(group_metrics["fpr"]),
        "false_negative_rate_gap": gap(group_metrics["fnr"]),
        "equal_opportunity_difference": gap(group_metrics["recall"]),
        "calibration_ece_gap": gap(group_metrics["ece"]),
    }


def explanation_tables(trained_models, client_data, preprocessor, device, args):
    feature_names = list(preprocessor.get_feature_names_out())
    x_parts = [client_data[city].x_test for city in FEDERATED_CITIES]
    x_all = torch.cat(x_parts, dim=0)
    if len(x_all) > args.xai_sample:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(len(x_all), size=args.xai_sample, replace=False)
        x_all = x_all[indices]

    importance_rows = []
    vectors = {}
    for model_name, model in trained_models.items():
        vector = gradient_importance(model, x_all, device)
        vectors[model_name] = vector
        top_indices = np.argsort(vector)[::-1][: args.xai_top_k]
        for rank, index in enumerate(top_indices, start=1):
            importance_rows.append(
                {
                    "model": model_name,
                    "rank": rank,
                    "feature": feature_names[index],
                    "importance": float(vector[index]),
                }
            )

    reference = vectors["FedAvg"]
    reference_top = set(np.argsort(reference)[::-1][: args.xai_top_k])
    pedi_rows = []
    for model_name, vector in vectors.items():
        spearman = pd.Series(reference).corr(pd.Series(vector), method="spearman")
        candidate_top = set(np.argsort(vector)[::-1][: args.xai_top_k])
        pedi_rows.append(
            {
                "reference_model": "FedAvg",
                "model": model_name,
                "spearman_rank_correlation": float(spearman),
                "top_k_feature_overlap": len(reference_top & candidate_top) / args.xai_top_k,
                "explanation_stability": max(float(spearman), 0.0),
                "pedi": 1.0 - float(spearman),
                "method": "input-gradient importance; fast PEDI proxy before full SHAP/DiCE run",
            }
        )
    return {"importance": pd.DataFrame(importance_rows), "pedi": pd.DataFrame(pedi_rows)}


def gradient_importance(model, x_sample: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    sample = x_sample.clone().to(device).requires_grad_(True)
    logits = model(sample)
    objective = torch.sigmoid(logits).mean()
    model.zero_grad(set_to_none=True)
    objective.backward()
    values = (sample.grad.detach().abs() * sample.detach().abs()).mean(dim=0).cpu().numpy()
    return values / max(float(values.sum()), 1e-12)


def membership_inference_table(predictions, trained_models, client_data, city_frames, args, device) -> pd.DataFrame:
    rows = []
    base_advantage = None
    for model_name, model in trained_models.items():
        train_scores = []
        nonmember_scores = []
        for city, data in client_data.items():
            train_scores.append(predict_torch_scores(model, data.x_train[: args.mia_sample], device))
            nonmember_scores.append(predictions[model_name][city]["score"])
        member_confidence = confidence(np.concatenate(train_scores))
        nonmember_confidence = confidence(np.concatenate(nonmember_scores))
        row = attack_row(model_name, member_confidence, nonmember_confidence)
        if model_name == "FedAvg":
            base_advantage = row["attack_advantage"]
        rows.append(row)

    central_model_path = RESULTS_MODELS_DIR / "step7_centralized_mlp.joblib"
    if central_model_path.exists():
        central_model = joblib.load(central_model_path)
        train_frame = sample_frame(pd.read_parquet(SPLITS_DIR / "centralized" / "centralized_train.parquet"), args.mia_sample, args.seed)
        test_frame = sample_frame(pd.read_parquet(SPLITS_DIR / "centralized" / "centralized_test.parquet"), args.mia_sample, args.seed)
        x_train, _ = split_xy(train_frame)
        x_test, _ = split_xy(test_frame)
        row = attack_row(
            "Centralized MLP",
            confidence(predict_sklearn_scores(central_model, x_train)),
            confidence(predict_sklearn_scores(central_model, x_test)),
        )
        rows.insert(0, row)

    table = pd.DataFrame(rows)
    if base_advantage is None:
        base_advantage = float(table["attack_advantage"].dropna().max())
    table["privacy_risk_reduction_pct"] = table["attack_advantage"].apply(
        lambda value: 100.0 * max(base_advantage - value, 0.0) / max(base_advantage, 1e-12)
    )
    return table


def attack_row(model_name: str, member_confidence: np.ndarray, nonmember_confidence: np.ndarray) -> dict:
    auc = membership_inference_auc(member_confidence, nonmember_confidence)
    labels = np.concatenate([np.ones(len(member_confidence)), np.zeros(len(nonmember_confidence))])
    scores = np.concatenate([member_confidence, nonmember_confidence])
    thresholds = np.quantile(scores, np.linspace(0.05, 0.95, 19))
    accuracy = max(float(((scores >= threshold).astype(int) == labels).mean()) for threshold in thresholds)
    return {
        "model": model_name,
        "attack_accuracy": accuracy,
        "attack_auc": auc,
        "attack_advantage": attack_advantage(auc),
    }


def confidence(scores: np.ndarray) -> np.ndarray:
    return np.maximum(scores, 1.0 - scores)


def privacy_tradeoff_table(long_metrics, macro_table, summaries, calibration_by_city, fairness, pedi):
    resources = resource_table(summaries).set_index("model")
    calibration = calibration_by_city.groupby("model", as_index=False).agg({"brier": "mean", "ece": "mean"})
    fairness_city = fairness[fairness["group_column"] == "city"][["model", "macro_f1_gap"]]
    rows = []
    for _, row in macro_table.iterrows():
        model = row["Model"]
        summary = next(item for item in summaries if item["model_label"] == model)
        cal = calibration[calibration["model"] == model].iloc[0]
        fair = fairness_city[fairness_city["model"] == model].iloc[0]
        drift = pedi[pedi["model"] == model].iloc[0]
        baseline_model = "FedProx" if model.startswith("FedProx") else "FedAvg"
        runtime_overhead = pct_overhead(resources.loc[model, "training_time_sec"], resources.loc[baseline_model, "training_time_sec"])
        communication_overhead = pct_overhead(
            resources.loc[model, "total_communication_mb"],
            resources.loc[baseline_model, "total_communication_mb"],
        )
        rows.append(
            {
                "model": model,
                "privacy_level": summary["privacy_level"],
                "target_epsilon": summary["target_epsilon"],
                "noise_multiplier": summary["noise_multiplier"],
                "global_macro_f1": row["Global Macro-F1"],
                "la_macro_f1": row["LA Macro-F1"],
                "brier": cal["brier"],
                "ece": cal["ece"],
                "explanation_drift_pedi": drift["pedi"],
                "fairness_macro_f1_gap": fair["macro_f1_gap"],
                "runtime_overhead_pct": runtime_overhead,
                "communication_overhead_pct": communication_overhead,
            }
        )
    return pd.DataFrame(rows)


def secure_aggregation_table(summaries, macro_table):
    by_model = {summary["model_label"]: summary for summary in summaries}
    base = by_model["FedAvg"]
    secagg = by_model["FedAvg + SecAgg"]
    base_perf = float(macro_table[macro_table["Model"] == "FedAvg"]["Global Macro-F1"].iloc[0])
    secagg_perf = float(macro_table[macro_table["Model"] == "FedAvg + SecAgg"]["Global Macro-F1"].iloc[0])
    return pd.DataFrame(
        [
            {
                "model": "FedAvg + SecAgg",
                "extra_runtime_pct": pct_overhead(secagg["training_time_sec"], base["training_time_sec"]),
                "extra_communication_mb": secagg["secure_aggregation_extra_bytes"] / (1024 * 1024),
                "communication_overhead_pct": pct_overhead(
                    secagg["total_bytes_with_overheads"], base["total_bytes_with_overheads"]
                ),
                "rounds_change": secagg["rounds_completed"] - base["rounds_completed"],
                "global_macro_f1_change": secagg_perf - base_perf,
            }
        ]
    )


def resource_table(summaries):
    rows = []
    for summary in summaries:
        rows.append(
            {
                "model": summary["model_label"],
                "training_time_sec": summary["training_time_sec"],
                "memory_peak_mb": summary["memory_peak_mb"],
                "model_size_mb": summary["model_size_bytes"] / (1024 * 1024),
                "upload_mb": summary["upload_bytes"] / (1024 * 1024),
                "download_mb": summary["download_bytes"] / (1024 * 1024),
                "secure_aggregation_extra_mb": summary["secure_aggregation_extra_bytes"] / (1024 * 1024),
                "total_communication_mb": summary["total_bytes_with_overheads"] / (1024 * 1024),
                "rounds_completed": summary["rounds_completed"],
            }
        )
    return pd.DataFrame(rows)


def tai_scorecard(macro_table, summaries, calibration_by_city, fairness, pedi, mia):
    resource = resource_table(summaries)
    calibration = calibration_by_city.groupby("model", as_index=False).agg({"ece": "mean"})
    fairness_city = fairness[fairness["group_column"] == "city"][["model", "macro_f1_gap"]]
    rows = []
    for _, row in macro_table.iterrows():
        model = row["Model"]
        summary = next(item for item in summaries if item["model_label"] == model)
        perf = float(row["Global Macro-F1"])
        privacy = privacy_component(summary, mia)
        xai = float(pedi[pedi["model"] == model]["explanation_stability"].iloc[0])
        fair = 1.0 - min(float(fairness_city[fairness_city["model"] == model]["macro_f1_gap"].iloc[0]), 1.0)
        model_resource = resource[resource["model"] == model].iloc[0]
        efficiency = 1.0 / (1.0 + model_resource["training_time_sec"] / 60.0 + model_resource["total_communication_mb"] / 100.0)
        calib = 1.0 - min(float(calibration[calibration["model"] == model]["ece"].iloc[0]), 1.0)
        tai = 0.25 * perf + 0.20 * privacy + 0.20 * xai + 0.15 * fair + 0.10 * efficiency + 0.10 * calib
        rows.append(
            {
                "Model": model,
                "Performance": perf,
                "Privacy": privacy,
                "XAI stability": xai,
                "Fairness": fair,
                "Efficiency": efficiency,
                "Calibration": calib,
                "TAI-Score": tai,
            }
        )
    table = pd.DataFrame(rows).sort_values("TAI-Score", ascending=False)
    return table


def privacy_component(summary, mia):
    model = summary["model_label"]
    base = {
        "None": 0.20,
        "Secure aggregation": 0.55,
        "Weak DP": 0.70,
        "Medium DP": 0.85,
        "Strong DP": 1.00,
    }[summary["privacy_level"]]
    if model in set(mia["model"]):
        advantage = float(mia[mia["model"] == model]["attack_advantage"].iloc[0])
        empirical = 1.0 - min(advantage, 1.0)
        return float(0.65 * base + 0.35 * empirical)
    return base


def pct_overhead(value, baseline) -> float:
    return float(100.0 * (value - baseline) / max(abs(baseline), 1e-12))


def gap(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return float("nan")
    return float(values.max() - values.min())


def slug(value: str) -> str:
    return (
        value.lower()
        .replace(" + ", "_")
        .replace(" ", "_")
        .replace("=", "")
        .replace("-", "_")
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
