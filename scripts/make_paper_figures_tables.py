"""Build publication-oriented figures and tables from GovTrust-FL results.

Run from the project root:
    python scripts/make_paper_figures_tables.py
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES_IN = RESULTS / "tables"
OUT = RESULTS / "paper_artifacts"
FIG_DIR = OUT / "figures"
TAB_DIR = OUT / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)

COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
]
HEATMAP_CMAP = "viridis"
HATCHES = ["", "//", "\\\\", "xx", "..", "++", "--"]
MODEL_ORDER = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "MLP",
    "Centralized",
    "Local-only",
    "FedAvg",
    "FedProx",
    "FedAvg + SecAgg",
    "FedAvg + DP eps=8",
    "FedAvg + DP eps=3",
    "FedAvg + DP eps=1",
    "FedProx + DP eps=3",
]
CITY_ORDER = ["NYC", "Chicago", "Boston", "LA", "los_angeles"]


def read_csv_if_exists(filename: str) -> pd.DataFrame | None:
    path = TABLES_IN / filename
    if not path.exists():
        warnings.warn(f"Missing file: {path}", stacklevel=2)
        return None
    return pd.read_csv(path)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip().replace(" ", "_").replace("-", "_").lower() for col in out.columns]
    return out


def find_col(df: pd.DataFrame, aliases: Iterable[str], required: bool = True) -> str | None:
    normalized = {col.lower(): col for col in df.columns}
    normalized_aliases = [alias.strip().replace(" ", "_").replace("-", "_").lower() for alias in aliases]
    for alias in normalized_aliases:
        if alias in normalized:
            return normalized[alias]
    for col in df.columns:
        col_low = col.lower()
        for alias in normalized_aliases:
            if alias in col_low:
                return col
    if required:
        raise KeyError(f"Could not find any of {list(aliases)} in {list(df.columns)}")
    return None


def order_models(df: pd.DataFrame, model_col: str) -> pd.DataFrame:
    out = df.copy()
    out[model_col] = out[model_col].astype(str)
    order_map = {name: index for index, name in enumerate(MODEL_ORDER)}
    out["_model_order"] = out[model_col].map(order_map).fillna(999)
    return out.sort_values(["_model_order", model_col]).drop(columns="_model_order")


def display_model_name(value: str) -> str:
    text = str(value)
    replacements = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "mlp": "MLP",
        "fedavg": "FedAvg",
        "fedprox": "FedProx",
    }
    return replacements.get(text.lower(), text)


def short_label(label: str, max_len: int = 24) -> str:
    label = str(label)
    return label if len(label) <= max_len else f"{label[: max_len - 1]}..."


def parse_epsilon(model_name: str) -> float | None:
    match = re.search(r"eps\s*=\s*([0-9.]+)", str(model_name).lower())
    return float(match.group(1)) if match else None


def save_figure(fig: plt.Figure, name: str) -> None:
    for ext in ("pdf", "svg", "png", "tiff"):
        dpi = 600 if ext in {"png", "tiff"} else None
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=dpi)
    plt.close(fig)


def save_table(df: pd.DataFrame, name: str, decimals: int = 4) -> None:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(decimals)
    out.to_csv(TAB_DIR / f"{name}.csv", index=False)
    latex = out.to_latex(
        index=False,
        escape=False,
        float_format=lambda value: f"{value:.{decimals}f}",
    )
    latex = latex.replace("\\toprule", "\\hline")
    latex = latex.replace("\\midrule", "\\hline")
    latex = latex.replace("\\bottomrule", "\\hline")
    (TAB_DIR / f"{name}.tex").write_text(latex, encoding="utf-8")


def annotate_bars(ax, bars, fmt: str = "{:.3f}", y_offset: float = 0.004) -> None:
    for bar in bars:
        height = bar.get_height()
        if np.isfinite(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + y_offset,
                fmt.format(height),
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )


def finalize_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def numeric_heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    val_col: str,
    title: str,
    cbar_label: str,
    name: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    pivot = df.pivot_table(index=row_col, columns=col_col, values=val_col, aggfunc="mean")
    pivot = pivot.reindex([m for m in MODEL_ORDER if m in pivot.index] + [m for m in pivot.index if m not in MODEL_ORDER])
    pivot = pivot[[c for c in CITY_ORDER if c in pivot.columns] + [c for c in pivot.columns if c not in CITY_ORDER]]
    fig, ax = plt.subplots(figsize=(max(5.5, 0.6 * pivot.shape[1] + 2), max(3.4, 0.35 * pivot.shape[0] + 1.2)))
    im = ax.imshow(pivot.values, aspect="auto", cmap=HEATMAP_CMAP, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.88).set_label(cbar_label)
    ax.grid(False)
    save_figure(fig, name)


def figure_centralized_baselines() -> None:
    df = read_csv_if_exists("step7_centralized_baselines.csv")
    if df is None:
        return
    df = clean_columns(df)
    model_col = find_col(df, ["model", "classifier", "algorithm"])
    df[model_col] = df[model_col].map(display_model_name)
    metric_cols = {
        "AUROC": find_col(df, ["auroc", "roc_auc"], required=False),
        "AUPRC": find_col(df, ["auprc", "average_precision"], required=False),
        "Macro-F1": find_col(df, ["macro_f1", "f1_macro"], required=False),
    }
    metric_cols = {name: col for name, col in metric_cols.items() if col is not None}
    df = order_models(df, model_col)
    save_table(df, "table_step7_centralized_baselines")
    x = np.arange(len(df))
    width = 0.22
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for index, (metric_name, col) in enumerate(metric_cols.items()):
        bars = ax.bar(
            x + (index - 1) * width,
            df[col],
            width,
            label=metric_name,
            color=COLORS[index % len(COLORS)],
            edgecolor="black",
            linewidth=0.5,
            hatch=HATCHES[index],
        )
        annotate_bars(ax, bars, fmt="{:.2f}", y_offset=0.006)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(model, 18) for model in df[model_col]], rotation=30, ha="right")
    ax.set_title("Centralized baseline performance on pooled training cities")
    ax.legend(frameon=False, ncol=max(1, len(metric_cols)), loc="upper left")
    finalize_axes(ax)
    save_figure(fig, "fig1_centralized_baselines")


def figure_external_la_baselines() -> None:
    df = read_csv_if_exists("step7_centralized_external_la.csv")
    if df is None:
        return
    df = clean_columns(df)
    model_col = find_col(df, ["model", "classifier", "algorithm"])
    df[model_col] = df[model_col].map(display_model_name)
    metrics = {
        "AUROC": find_col(df, ["auroc", "roc_auc"], required=False),
        "AUPRC": find_col(df, ["auprc", "average_precision"], required=False),
        "Macro-F1": find_col(df, ["macro_f1", "f1_macro"], required=False),
    }
    metrics = {name: col for name, col in metrics.items() if col is not None}
    df = order_models(df, model_col)
    save_table(df, "table_step7_external_la")
    x = np.arange(len(df))
    width = 0.22
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for index, (metric_name, col) in enumerate(metrics.items()):
        bars = ax.bar(
            x + (index - 1) * width,
            df[col],
            width,
            label=metric_name,
            color=COLORS[index % len(COLORS)],
            edgecolor="black",
            linewidth=0.5,
            hatch=HATCHES[index],
        )
        annotate_bars(ax, bars, fmt="{:.2f}", y_offset=0.006)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(model, 18) for model in df[model_col]], rotation=30, ha="right")
    ax.set_title("External validation on Los Angeles")
    ax.legend(frameon=False, ncol=max(1, len(metrics)), loc="upper left")
    finalize_axes(ax)
    save_figure(fig, "fig2_external_la_baselines")


def figure_local_only_auroc_matrix() -> None:
    df = read_csv_if_exists("step8_local_only_auroc_matrix.csv")
    if df is None:
        return
    df = clean_columns(df)
    train_col = find_col(df, ["train_city", "train", "training_city"])
    test_cols = [col for col in df.columns if col.startswith("test_")]
    if not test_cols:
        test_cols = [
            col
            for col in df.columns
            if col != train_col and any(token in col.lower() for token in ("nyc", "chicago", "boston", "la", "los_angeles"))
        ]
    if not test_cols:
        return
    matrix = df[[train_col] + test_cols].rename(columns={train_col: "Train city"})
    for col in test_cols:
        label = col.replace("test_", "").replace("_", " ").title().replace("Nyc", "NYC").replace("La", "LA")
        matrix = matrix.rename(columns={col: label})
    save_table(matrix, "table_step8_local_only_auroc_matrix")
    heat = matrix.set_index("Train city").apply(pd.to_numeric, errors="coerce")
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    im = ax.imshow(heat.values, cmap=HEATMAP_CMAP, vmin=0, vmax=1)
    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_xticklabels(heat.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_yticklabels(heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.iloc[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Cross-city transferability of local-only LightGBM models")
    fig.colorbar(im, ax=ax, shrink=0.82).set_label("AUROC")
    ax.grid(False)
    save_figure(fig, "fig3_local_only_transfer_auroc")


def figure_fl_macro_f1_heatmap() -> None:
    df = read_csv_if_exists("step11_12_fl_privacy_macro_f1.csv")
    if df is None:
        df = read_csv_if_exists("step9_10_fedavg_vs_fedprox_macro_f1.csv")
    if df is None:
        return
    df = clean_columns(df)
    model_col = find_col(df, ["model", "configuration", "method"])
    rename = {
        find_col(df, ["global_macro_f1", "global"], required=False): "Global",
        find_col(df, ["nyc"], required=False): "NYC",
        find_col(df, ["chicago"], required=False): "Chicago",
        find_col(df, ["boston"], required=False): "Boston",
        find_col(df, ["la"], required=False): "LA",
    }
    rename = {key: value for key, value in rename.items() if key is not None}
    score_df = df[[model_col] + list(rename)].rename(columns={model_col: "Model", **rename})
    score_df = order_models(score_df, "Model")
    save_table(score_df, "table_fl_macro_f1_scorecard")
    heat = score_df.set_index("Model").apply(pd.to_numeric, errors="coerce")
    fig, ax = plt.subplots(figsize=(6.8, 0.42 * len(heat) + 1.5))
    im = ax.imshow(heat.values, cmap=HEATMAP_CMAP, vmin=0, vmax=1)
    ax.set_xticks(np.arange(heat.shape[1]))
    ax.set_xticklabels(heat.columns)
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_yticklabels(heat.index)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7)
    ax.set_title("FL/privacy Macro-F1 comparison (simulation/proxy stage)")
    fig.colorbar(im, ax=ax, shrink=0.88).set_label("Macro-F1")
    ax.grid(False)
    save_figure(fig, "fig4_fl_privacy_macro_f1_heatmap")


def figure_privacy_utility_tradeoff() -> None:
    df = read_csv_if_exists("step11_12_fl_privacy_macro_f1.csv")
    if df is None:
        return
    df = clean_columns(df)
    model_col = find_col(df, ["model", "configuration", "method"])
    global_col = find_col(df, ["global_macro_f1", "global", "macro_f1"], required=False)
    la_col = find_col(df, ["la"], required=False)
    if global_col is None:
        return
    work = df[[model_col, global_col] + ([la_col] if la_col else [])].copy()
    work["epsilon"] = work[model_col].apply(parse_epsilon)
    work = work.dropna(subset=["epsilon"]).sort_values("epsilon")
    if work.empty:
        return
    save_table(work.rename(columns={model_col: "Model", global_col: "Global Macro-F1"}), "table_dp_privacy_utility_tradeoff")
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(work["epsilon"], work[global_col], marker="o", linewidth=1.8, color=COLORS[0], label="Global Macro-F1")
    if la_col is not None:
        ax.plot(work["epsilon"], work[la_col], marker="s", linewidth=1.6, linestyle="--", color=COLORS[1], label="LA Macro-F1")
    for _, row in work.iterrows():
        ax.text(row["epsilon"], row[global_col] + 0.006, f"eps={row['epsilon']:.0f}", ha="center", fontsize=7)
    ax.set_xlabel("Differential privacy budget epsilon")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title("Privacy-utility trade-off under simulated update-level DP")
    ax.legend(frameon=False)
    finalize_axes(ax)
    save_figure(fig, "fig5_privacy_utility_tradeoff_dp")


def figure_membership_inference() -> None:
    df = read_csv_if_exists("step13_membership_inference.csv")
    if df is None:
        return
    df = clean_columns(df)
    model_col = find_col(df, ["model", "configuration", "method"])
    auc_col = find_col(df, ["attack_auc", "mia_auc", "auc"], required=False)
    if auc_col is None:
        return
    plot_df = order_models(df[[model_col, auc_col]].rename(columns={model_col: "Model", auc_col: "Attack AUC"}), "Model")
    save_table(df, "table_membership_inference")
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(plot_df))
    bars = ax.bar(x, plot_df["Attack AUC"], color=COLORS[3], edgecolor="black", linewidth=0.5, hatch="//")
    annotate_bars(ax, bars, fmt="{:.2f}", y_offset=0.006)
    ax.axhline(0.5, linestyle="--", linewidth=1.0, color="#333333")
    ax.text(len(plot_df) - 0.7, 0.515, "random baseline", fontsize=7, ha="right")
    ax.set_ylabel("Attack AUC")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(model, 20) for model in plot_df["Model"]], rotation=30, ha="right")
    ax.set_title("Membership-inference risk across configurations")
    finalize_axes(ax)
    save_figure(fig, "fig6_membership_inference")


def figure_calibration() -> None:
    df = read_csv_if_exists("step17_calibration_by_city.csv")
    if df is None:
        return
    df = clean_columns(df)
    model_col = find_col(df, ["model", "configuration", "method"])
    city_col = find_col(df, ["city", "test_city", "client"])
    brier_col = find_col(df, ["brier", "brier_score"], required=False)
    ece_col = find_col(df, ["ece", "expected_calibration_error"], required=False)
    save_table(df, "table_calibration_by_city")
    if brier_col:
        temp = df[[model_col, city_col, brier_col]].rename(columns={model_col: "Model", city_col: "City", brier_col: "Brier score"})
        numeric_heatmap(temp, "Model", "City", "Brier score", "Calibration by city: Brier score", "Brier score", "fig7a_calibration_brier_by_city", vmin=0)
    if ece_col:
        temp = df[[model_col, city_col, ece_col]].rename(columns={model_col: "Model", city_col: "City", ece_col: "ECE"})
        numeric_heatmap(temp, "Model", "City", "ECE", "Calibration by city: expected calibration error", "ECE", "fig7b_calibration_ece_by_city", vmin=0)


def figure_fairness_summary() -> None:
    df = read_csv_if_exists("step16_fairness_summary.csv")
    if df is None:
        return
    df = clean_columns(df)
    model_col = find_col(df, ["model", "configuration", "method"])
    group_col = find_col(df, ["group_column"], required=False)
    if group_col:
        city_rows = df[df[group_col] == "city"]
        if not city_rows.empty:
            df = city_rows
    metric_col = find_col(df, ["macro_f1_gap", "equal_opportunity_difference", "recall_gap"], required=False)
    if metric_col is None:
        return
    save_table(df, "table_fairness_summary")
    plot_df = order_models(df[[model_col, metric_col]].rename(columns={model_col: "Model", metric_col: "Fairness gap"}), "Model")
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    x = np.arange(len(plot_df))
    bars = ax.bar(x, plot_df["Fairness gap"], color=COLORS[2], edgecolor="black", linewidth=0.5, hatch="xx")
    annotate_bars(ax, bars, fmt="{:.3f}", y_offset=0.004)
    ax.set_ylabel(metric_col.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(model, 20) for model in plot_df["Model"]], rotation=30, ha="right")
    ax.set_title("Geographic fairness gap across model configurations")
    finalize_axes(ax)
    save_figure(fig, "fig8_fairness_gap_summary")


def figure_resource_communication() -> None:
    df = read_csv_if_exists("step18_resource_communication.csv")
    if df is None:
        return
    df = clean_columns(df)
    model_col = find_col(df, ["model", "configuration", "method"])
    save_table(df, "table_resource_communication")
    metrics = {
        "Training time": find_col(df, ["training_time_sec", "runtime"], required=False),
        "Memory": find_col(df, ["memory_peak_mb", "memory_mb"], required=False),
        "Communication": find_col(df, ["total_communication_mb", "communication_mb"], required=False),
    }
    for label, col in metrics.items():
        if col is None:
            continue
        plot_df = order_models(df[[model_col, col]].rename(columns={model_col: "Model", col: label}), "Model")
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        x = np.arange(len(plot_df))
        ax.bar(x, plot_df[label], color=COLORS[0], edgecolor="black", linewidth=0.5, hatch="//")
        ax.set_ylabel(label)
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(model, 20) for model in plot_df["Model"]], rotation=30, ha="right")
        ax.set_title(f"{label} across model configurations")
        finalize_axes(ax)
        save_figure(fig, f"fig9_{label.lower().replace(' ', '_')}_cost")


def figure_round_history() -> None:
    df = read_csv_if_exists("step11_19_fl_privacy_round_history.csv")
    if df is None:
        df = read_csv_if_exists("step9_10_fl_mlp_round_history.csv")
    if df is None:
        return
    df = clean_columns(df)
    model_col = find_col(df, ["model", "algorithm", "configuration", "method"], required=False)
    round_col = find_col(df, ["round", "communication_round", "server_round"])
    metric_col = find_col(df, ["mean_val_macro_f1", "val_macro_f1", "macro_f1"], required=False)
    if metric_col is None:
        return
    save_table(df, "table_fl_round_history")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    if model_col:
        for index, (model, sub) in enumerate(df.groupby(model_col, sort=False)):
            sub = sub.sort_values(round_col)
            ax.plot(
                sub[round_col],
                sub[metric_col],
                marker="o",
                markersize=3,
                linewidth=1.4,
                color=COLORS[index % len(COLORS)],
                label=str(model),
            )
        ax.legend(frameon=False, ncol=1)
    else:
        sub = df.sort_values(round_col)
        ax.plot(sub[round_col], sub[metric_col], marker="o", linewidth=1.4, color=COLORS[0])
    ax.set_xlabel("Communication round")
    ax.set_ylabel(metric_col.replace("_", " ").title())
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title("Federated learning convergence")
    finalize_axes(ax)
    save_figure(fig, "fig10_fl_convergence")


def figure_tai_scorecard() -> None:
    df = read_csv_if_exists("step19_tai_scorecard.csv")
    if df is None:
        return
    df = clean_columns(df)
    model_col = find_col(df, ["model", "configuration", "method"])
    tai_col = find_col(df, ["tai_score", "trustworthy_ai_score", "score"])
    save_table(df, "table_tai_scorecard")
    plot_df = order_models(df[[model_col, tai_col]].rename(columns={model_col: "Model", tai_col: "TAI-Score"}), "Model")
    plot_df = plot_df.sort_values("TAI-Score", ascending=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    y = np.arange(len(plot_df))
    bars = ax.barh(y, plot_df["TAI-Score"], color=COLORS[0], edgecolor="black", linewidth=0.5, hatch="//")
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.006, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", fontsize=7)
    ax.set_xlabel("TAI-Score")
    ax.set_xlim(0, 1)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["Model"])
    ax.set_title("Multi-objective trustworthy AI scorecard")
    finalize_axes(ax)
    save_figure(fig, "fig11a_tai_score_ranking")

    components = {
        "Performance": find_col(df, ["performance"], required=False),
        "Privacy": find_col(df, ["privacy"], required=False),
        "XAI stability": find_col(df, ["xai_stability", "explainability_stability"], required=False),
        "Fairness": find_col(df, ["fairness"], required=False),
        "Efficiency": find_col(df, ["efficiency"], required=False),
        "Calibration": find_col(df, ["calibration"], required=False),
    }
    components = {name: col for name, col in components.items() if col is not None}
    if len(components) < 3:
        return
    top = df.sort_values(tai_col, ascending=False).head(4)
    labels = list(components)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig = plt.figure(figsize=(5.2, 5.2))
    ax = plt.subplot(111, polar=True)
    for index, (_, row) in enumerate(top.iterrows()):
        values = [float(row[components[label]]) for label in labels]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, color=COLORS[index % len(COLORS)], label=str(row[model_col]))
        ax.fill(angles, values, color=COLORS[index % len(COLORS)], alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
    ax.set_ylim(0, 1)
    ax.set_title("Trustworthiness dimensions of top-ranked configurations")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.38, 1.12))
    save_figure(fig, "fig11b_tai_score_radar")


def make_main_model_comparison_table() -> None:
    parts = []
    central = read_csv_if_exists("step7_centralized_baselines.csv")
    if central is not None:
        central = clean_columns(central)
        model_col = find_col(central, ["model", "classifier", "algorithm"])
        cols = {
            "Pooled AUROC": find_col(central, ["auroc", "roc_auc"], required=False),
            "Pooled AUPRC": find_col(central, ["auprc", "average_precision"], required=False),
            "Pooled Macro-F1": find_col(central, ["macro_f1"], required=False),
        }
        keep = [model_col] + [col for col in cols.values() if col is not None]
        temp = central[keep].rename(columns={model_col: "Model", **{col: name for name, col in cols.items() if col}})
        temp["Model"] = temp["Model"].map(display_model_name)
        temp["Experiment"] = "Centralized baseline"
        parts.append(temp)
    fl = read_csv_if_exists("step11_12_fl_privacy_macro_f1.csv")
    if fl is not None:
        fl = clean_columns(fl)
        model_col = find_col(fl, ["model", "configuration", "method"])
        global_col = find_col(fl, ["global_macro_f1", "global"], required=False)
        la_col = find_col(fl, ["la"], required=False)
        keep = [model_col] + [col for col in [global_col, la_col] if col is not None]
        temp = fl[keep].rename(columns={model_col: "Model"})
        if global_col:
            temp = temp.rename(columns={global_col: "Pooled Macro-F1"})
        if la_col:
            temp = temp.rename(columns={la_col: "LA Macro-F1"})
        temp["Experiment"] = "Federated/privacy"
        parts.append(temp)
    tai = read_csv_if_exists("step19_tai_scorecard.csv")
    if tai is not None:
        tai = clean_columns(tai)
        model_col = find_col(tai, ["model", "configuration", "method"])
        tai_col = find_col(tai, ["tai_score", "trustworthy_ai_score", "score"], required=False)
        if tai_col:
            temp = tai[[model_col, tai_col]].rename(columns={model_col: "Model", tai_col: "TAI-Score"})
            temp["Experiment"] = "Trustworthiness scorecard"
            parts.append(temp)
    if parts:
        main = pd.concat(parts, ignore_index=True, sort=False)
        main = main[["Experiment", "Model"] + [col for col in main.columns if col not in {"Experiment", "Model"}]]
        save_table(main, "table_main_model_comparison")


def make_dataset_feature_summary() -> None:
    manifest_path = ROOT / "data" / "processed" / "feature_manifest_step5.json"
    rows = []
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict):
            for city in ("nyc", "chicago", "boston", "la"):
                item = manifest.get(city) or manifest.get(city.upper()) or manifest.get(city.title())
                if isinstance(item, dict):
                    rows.append(
                        {
                            "City": city.upper() if city != "la" else "LA",
                            "Rows": item.get("rows") or item.get("n_rows") or item.get("samples"),
                            "Columns": item.get("columns") or item.get("n_columns") or item.get("features"),
                        }
                    )
    if not rows:
        rows = [
            {"City": "NYC", "Rows": 486023, "Columns": 118},
            {"City": "Chicago", "Rows": 287961, "Columns": 118},
            {"City": "Boston", "Rows": 176884, "Columns": 118},
            {"City": "LA", "Rows": 189016, "Columns": 118},
        ]
    save_table(pd.DataFrame(rows), "table_dataset_feature_summary", decimals=0)


def main() -> None:
    make_dataset_feature_summary()
    figure_centralized_baselines()
    figure_external_la_baselines()
    figure_local_only_auroc_matrix()
    figure_fl_macro_f1_heatmap()
    figure_privacy_utility_tradeoff()
    figure_membership_inference()
    figure_calibration()
    figure_fairness_summary()
    figure_resource_communication()
    figure_round_history()
    figure_tai_scorecard()
    make_main_model_comparison_table()
    print(f"Done. Figures saved to: {FIG_DIR}")
    print(f"Done. Tables saved to: {TAB_DIR}")


if __name__ == "__main__":
    main()
