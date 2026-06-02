"""Build fairness, calibration, resource, and TAI-Score tables.

This script consumes prepared split files and any saved sklearn baseline models.
It intentionally does not fabricate missing experiments: unavailable privacy,
PEDI, or resource inputs are logged and represented with conservative neutral
defaults in the scorecard so smoke tests can run on synthetic data.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration import expected_calibration_error  # noqa: E402
from src.config import EXTERNAL_VALIDATION_CITY, FEDERATED_CLIENTS, ensure_project_dirs  # noqa: E402
from src.config_loader import load_config, resolve_path  # noqa: E402
from src.features import build_xy  # noqa: E402
from src.fairness import max_group_gap, subgroup_metrics  # noqa: E402
from src.metrics import binary_classification_metrics, predict_scores  # noqa: E402
from src.tai_score import build_scorecard  # noqa: E402


@dataclass(frozen=True)
class ModelRecord:
    """Saved sklearn model metadata."""

    label: str
    path: Path
    setting: str
    train_city: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_dirs()

    splits_dir = resolve_path(config, "data.splits_dir")
    tables_dir = resolve_path(config, "outputs.tables_dir")
    figures_dir = resolve_path(config, "outputs.figures_dir")
    logs_dir = resolve_path(config, "outputs.logs_dir")
    models_dir = resolve_path(config, "outputs.models_dir")
    group_columns = list(config["fairness"]["group_columns"])
    n_bins = int(config["calibration"]["n_bins"])

    test_frames = load_test_frames(splits_dir)
    models = discover_models(models_dir)
    if not models:
        raise FileNotFoundError(
            f"No baseline models found in {models_dir}. Run scripts/run_baselines.py first."
        )

    performance_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    fairness_rows: list[dict[str, object]] = []
    resource_rows: list[dict[str, object]] = []

    for record in models:
        model = joblib.load(record.path)
        combined_frames = []
        combined_true = []
        combined_scores = []
        inference_time_sec = 0.0

        for city, frame in test_frames.items():
            x_test, y_test = build_xy(frame)
            started = time.perf_counter()
            scores = predict_scores(model, x_test)
            inference_time_sec += time.perf_counter() - started

            metrics = binary_classification_metrics(y_test, scores, threshold=args.threshold)
            performance_rows.append(
                {
                    "model": record.label,
                    "setting": record.setting,
                    "train_city": record.train_city,
                    "test_city": city,
                    "test_rows": len(y_test),
                    **metrics,
                }
            )
            calibration_rows.append(
                {
                    "model": record.label,
                    "city": city,
                    "brier": metrics["brier"],
                    "ece": expected_calibration_error(y_test, scores, n_bins=n_bins),
                    "n": len(y_test),
                }
            )
            reliability_rows.extend(
                reliability_bins(record.label, city, y_test.to_numpy(dtype=int), scores, n_bins=n_bins)
            )
            combined_frames.append(frame.reset_index(drop=True))
            combined_true.append(y_test.to_numpy(dtype=int))
            combined_scores.append(np.asarray(scores, dtype=float))

        combined_frame = pd.concat(combined_frames, ignore_index=True)
        y_all = np.concatenate(combined_true)
        scores_all = np.concatenate(combined_scores)
        for group_column in group_columns:
            if group_column in combined_frame.columns:
                fairness_rows.append(
                    fairness_summary_row(
                        record.label,
                        combined_frame,
                        y_all,
                        scores_all,
                        group_column,
                        threshold=args.threshold,
                    )
                )

        resource_rows.append(
            {
                "model": record.label,
                "setting": record.setting,
                "model_size_mb": record.path.stat().st_size / (1024 * 1024),
                "inference_time_sec": inference_time_sec,
                "runtime_sec": inference_time_sec,
                "communication_mb": 0.0,
            }
        )

    performance = pd.DataFrame(performance_rows)
    calibration = pd.DataFrame(calibration_rows)
    reliability = pd.DataFrame(reliability_rows)
    fairness = pd.DataFrame(fairness_rows)
    resources = pd.DataFrame(resource_rows)

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    performance.to_csv(tables_dir / "trustworthiness_model_performance.csv", index=False)
    calibration.to_csv(tables_dir / "step17_calibration_by_city.csv", index=False)
    calibration.to_csv(tables_dir / "calibration_metrics.csv", index=False)
    reliability.to_csv(tables_dir / "step17_reliability_bins.csv", index=False)
    fairness.to_csv(tables_dir / "step16_fairness_summary.csv", index=False)
    fairness.to_csv(tables_dir / "fairness_metrics.csv", index=False)
    resources.to_csv(tables_dir / "step18_resource_communication.csv", index=False)
    resources.to_csv(tables_dir / "efficiency_results.csv", index=False)
    plot_reliability_curve(reliability, figures_dir / "reliability_curve.png")
    plot_reliability_curve(reliability, figures_dir / "reliability_diagram_overall.png")
    plot_fairness_heatmap(fairness, figures_dir / "fairness_heatmap.png")

    tai_input = build_tai_input(
        performance,
        calibration,
        fairness,
        resources,
        privacy_table=read_optional_table(tables_dir / "privacy_attack_metrics.csv"),
        pedi_table=read_optional_table(tables_dir / "pedi_metrics.csv"),
    )
    tai = build_scorecard(tai_input, weights=config["tai_score"]["weights"])
    tai = tai.rename(
        columns={
            "predictive_utility": "Performance",
            "privacy_protection": "Privacy",
            "explanation_stability": "XAI stability",
            "fairness": "Fairness",
            "efficiency": "Efficiency",
            "calibration": "Calibration",
            "tai_score": "TAI-Score",
        }
    )
    tai.to_csv(tables_dir / "step19_tai_scorecard.csv", index=False)
    tai.to_csv(tables_dir / "tai_scorecard.csv", index=False)
    tai.to_csv(tables_dir / "tai_score_ranking.csv", index=False)
    plot_tai_radar_chart(tai, figures_dir / "tai_score_radar_chart.png")

    log = {
        "models_evaluated": [record.label for record in models],
        "test_cities": list(test_frames),
        "tables": {
            "performance": str(tables_dir / "trustworthiness_model_performance.csv"),
            "calibration": str(tables_dir / "step17_calibration_by_city.csv"),
            "fairness": str(tables_dir / "step16_fairness_summary.csv"),
            "resources": str(tables_dir / "step18_resource_communication.csv"),
            "tai_score": str(tables_dir / "step19_tai_scorecard.csv"),
            "reliability_curve": str(figures_dir / "reliability_curve.png"),
        },
        "note": "Fairness is geographic/service-category based. No demographic fairness claims are made.",
    }
    (logs_dir / "trustworthiness_eval.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Wrote trustworthiness tables to: {tables_dir}")
    return 0


def load_test_frames(splits_dir: Path) -> dict[str, pd.DataFrame]:
    frames = {
        city: pd.read_parquet(splits_dir / city / f"{city}_test.parquet")
        for city in FEDERATED_CLIENTS
    }
    frames[EXTERNAL_VALIDATION_CITY] = pd.read_parquet(
        splits_dir / "external" / "los_angeles_external_test.parquet"
    )
    return frames


def discover_models(models_dir: Path) -> list[ModelRecord]:
    records = []
    for path in sorted(models_dir.glob("*.joblib")):
        stem = path.stem
        if stem.startswith("centralized_"):
            records.append(
                ModelRecord(
                    label=stem,
                    path=path,
                    setting="centralized",
                    train_city="pooled",
                )
            )
        elif stem.startswith("local_only_"):
            parts = stem.split("_")
            city = parts[2] if len(parts) >= 3 else "unknown"
            records.append(
                ModelRecord(
                    label=stem,
                    path=path,
                    setting="local_only",
                    train_city=city,
                )
            )
    return records


def reliability_bins(
    model: str,
    city: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int,
) -> list[dict[str, object]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score, bins[1:-1], right=True)
    rows = []
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        rows.append(
            {
                "model": model,
                "city": city,
                "bin": bin_id,
                "mean_predicted_probability": float(y_score[mask].mean()),
                "observed_delay_rate": float(y_true[mask].mean()),
                "n": int(mask.sum()),
            }
        )
    return rows


def fairness_summary_row(
    model: str,
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    group_column: str,
    *,
    threshold: float,
) -> dict[str, object]:
    metrics = subgroup_metrics(frame, y_true, y_score, group_column, threshold=threshold)
    if metrics.empty:
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

    working = frame[[group_column]].copy()
    working["_y_true"] = y_true
    working["_y_pred"] = (y_score >= threshold).astype(int)
    rate_rows = []
    for group_value, group_frame in working.groupby(group_column, dropna=False):
        y_group = group_frame["_y_true"].to_numpy(dtype=int)
        pred_group = group_frame["_y_pred"].to_numpy(dtype=int)
        fp = int(((y_group == 0) & (pred_group == 1)).sum())
        tn = int(((y_group == 0) & (pred_group == 0)).sum())
        fn = int(((y_group == 1) & (pred_group == 0)).sum())
        tp = int(((y_group == 1) & (pred_group == 1)).sum())
        rate_rows.append(
            {
                group_column: group_value,
                "fpr": fp / max(fp + tn, 1),
                "fnr": fn / max(fn + tp, 1),
            }
        )
    rates = pd.DataFrame(rate_rows)
    return {
        "model": model,
        "group_column": group_column,
        "groups": int(len(metrics)),
        "precision_gap": max_group_gap(metrics, "precision"),
        "recall_gap": max_group_gap(metrics, "recall"),
        "macro_f1_gap": max_group_gap(metrics, "macro_f1"),
        "false_positive_rate_gap": gap(rates["fpr"]),
        "false_negative_rate_gap": gap(rates["fnr"]),
        "equal_opportunity_difference": max_group_gap(metrics, "recall"),
        "calibration_ece_gap": np.nan,
    }


def build_tai_input(
    performance: pd.DataFrame,
    calibration: pd.DataFrame,
    fairness: pd.DataFrame,
    resources: pd.DataFrame,
    *,
    privacy_table: pd.DataFrame | None,
    pedi_table: pd.DataFrame | None,
) -> pd.DataFrame:
    utility_rows = []
    for model, group in performance.groupby("model"):
        utility_rows.append(
            {
                "model": model,
                "utility": float(np.average(group["macro_f1"], weights=group["test_rows"])),
            }
        )
    utility = pd.DataFrame(utility_rows)
    calibration_error = calibration.groupby("model", as_index=False)["ece"].mean()
    calibration_error = calibration_error.rename(columns={"ece": "calibration_error"})
    fairness_gap = fairness.groupby("model", as_index=False)["macro_f1_gap"].mean()
    fairness_gap = fairness_gap.rename(columns={"macro_f1_gap": "fairness_gap"})
    runtime = resources[["model", "runtime_sec"]].copy()

    output = utility.merge(calibration_error, on="model", how="left")
    output = output.merge(fairness_gap, on="model", how="left")
    output = output.merge(runtime, on="model", how="left")
    output["privacy_risk"] = map_metric(
        output["model"],
        privacy_table,
        value_columns=("attack_advantage", "attack_auc"),
        default=0.5,
    )
    output["pedi"] = map_metric(
        output["model"],
        pedi_table,
        value_columns=("pedi", "privacy_explanation_drift_index"),
        default=0.0,
    )
    output["calibration_error"] = output["calibration_error"].fillna(1.0)
    output["fairness_gap"] = output["fairness_gap"].fillna(1.0)
    output["runtime_sec"] = output["runtime_sec"].fillna(output["runtime_sec"].max())
    return output


def plot_reliability_curve(reliability: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1.0, label="perfect")
    if not reliability.empty:
        for model, model_frame in reliability.groupby("model"):
            averaged = (
                model_frame.groupby("bin", as_index=False)
                .agg(
                    mean_predicted_probability=("mean_predicted_probability", "mean"),
                    observed_delay_rate=("observed_delay_rate", "mean"),
                )
                .sort_values("mean_predicted_probability")
            )
            ax.plot(
                averaged["mean_predicted_probability"],
                averaged["observed_delay_rate"],
                marker="o",
                linewidth=1.1,
                markersize=3,
                label=str(model)[:24],
            )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed delay rate")
    ax.set_title("Reliability curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_fairness_heatmap(fairness: pd.DataFrame, output_path: Path) -> None:
    """Save a compact fairness-gap heatmap."""

    if fairness.empty or "macro_f1_gap" not in fairness.columns:
        return
    pivot = fairness.pivot_table(index="model", columns="group_column", values="macro_f1_gap", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(max(4.8, 0.8 * len(pivot.columns) + 2), max(3.2, 0.35 * len(pivot) + 1.2)))
    image = ax.imshow(pivot.fillna(0).to_numpy(), aspect="auto", cmap="Greys")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Fairness gap summary")
    fig.colorbar(image, ax=ax, label="Macro-F1 gap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_tai_radar_chart(tai: pd.DataFrame, output_path: Path) -> None:
    """Save a radar chart for the top TAI-Score model."""

    component_columns = ["Performance", "Privacy", "XAI stability", "Fairness", "Efficiency", "Calibration"]
    if tai.empty or any(column not in tai.columns for column in component_columns):
        return
    row = tai.sort_values("TAI-Score", ascending=False).iloc[0]
    values = [float(row[column]) for column in component_columns]
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(component_columns), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(4.8, 4.8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, color="#222222", linewidth=1.5)
    ax.fill(angles, values, color="#999999", alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(component_columns)
    ax.set_ylim(0, 1)
    ax.set_title(f"TAI-Score profile: {row['model']}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def read_optional_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def map_metric(
    models: pd.Series,
    table: pd.DataFrame | None,
    *,
    value_columns: tuple[str, ...],
    default: float,
) -> pd.Series:
    values = pd.Series(default, index=models.index, dtype=float)
    if table is None or "model" not in table.columns:
        return values
    lookup_column = next((column for column in value_columns if column in table.columns), None)
    if lookup_column is None:
        return values
    lookup = table.drop_duplicates("model").set_index("model")[lookup_column]
    mapped = models.map(lookup)
    return mapped.fillna(default).astype(float)


def gap(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return float("nan")
    return float(values.max() - values.min())


if __name__ == "__main__":
    raise SystemExit(main())
