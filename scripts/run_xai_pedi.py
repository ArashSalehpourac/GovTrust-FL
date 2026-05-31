"""Compute SHAP-based explanation stability and PEDI metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config, resolve_path  # noqa: E402
from src.features import build_xy  # noqa: E402
from src.pedi import privacy_explanation_drift_index, spearman_rank_stability, top_k_overlap  # noqa: E402
from src.xai import compute_shap_values, mean_absolute_importance  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--reference-model", type=Path, default=None)
    parser.add_argument("--private-model", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    splits_dir = resolve_path(config, "data.splits_dir")
    models_dir = resolve_path(config, "outputs.models_dir")
    tables_dir = resolve_path(config, "outputs.tables_dir")
    figures_dir = resolve_path(config, "outputs.figures_dir")
    reference_path = args.reference_model or models_dir / f"centralized_{config['model']['type']}.joblib"
    private_path = args.private_model or reference_path
    if not reference_path.exists() or not private_path.exists():
        raise FileNotFoundError("Reference/private model path does not exist")

    reference_model = joblib.load(reference_path)
    private_model = joblib.load(private_path)
    frame = pd.read_parquet(splits_dir / "nyc" / "nyc_test.parquet")
    frame = frame.sample(min(int(config["xai"]["sample_size"]), len(frame)), random_state=int(config["project"]["random_seed"]))
    x, _ = build_xy(frame)

    ref_importance, ref_method = model_importance_vector(reference_model, x, config)
    private_importance, private_method = model_importance_vector(private_model, x, config)
    row = {
        "reference_model": reference_path.name,
        "private_model": private_path.name,
        "spearman_rank_stability": spearman_rank_stability(ref_importance, private_importance),
        "top_k_overlap": top_k_overlap(ref_importance, private_importance, k=int(config["xai"]["top_k"])),
        "pedi": privacy_explanation_drift_index(ref_importance, private_importance),
        "method": f"reference={ref_method}; private={private_method}",
    }

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(tables_dir / "pedi_metrics.csv", index=False)
    plot_importance(ref_importance, figures_dir / "shap_summary_reference.png", "Reference importance")
    plot_importance(private_importance, figures_dir / "shap_summary_private.png", "Private importance")
    print(f"Wrote PEDI metrics: {tables_dir / 'pedi_metrics.csv'}")
    return 0


def model_importance_vector(model, x: pd.DataFrame, config: dict) -> tuple[np.ndarray, str]:
    """Compute SHAP importance when possible, otherwise use a smoke-test proxy."""

    try:
        _, shap_values = compute_shap_values(
            model,
            x,
            sample_size=int(config["xai"]["sample_size"]),
            random_state=int(config["project"]["random_seed"]),
        )
        importance = mean_absolute_importance(shap_values, list(x.columns))
        values = importance["importance"].to_numpy(dtype=float)
        if values.size:
            return values / max(values.sum(), 1e-12), "SHAP via src.xai.compute_shap_values"
    except Exception as exc:  # pragma: no cover - depends on optional SHAP/model support
        fallback_reason = f"proxy fallback after SHAP failure: {type(exc).__name__}"
    else:
        fallback_reason = "proxy fallback after empty SHAP importance"

    estimator = model.named_steps.get("model") if hasattr(model, "named_steps") else model
    if hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_).ravel()
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    else:
        values = np.ones(max(1, x.shape[1]), dtype=float)
    if values.size == 0:
        values = np.ones(max(1, x.shape[1]), dtype=float)
    return values / max(values.sum(), 1e-12), fallback_reason


def plot_importance(values: np.ndarray, output_path: Path, title: str) -> None:
    top = np.argsort(values)[::-1][: min(15, len(values))]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(range(len(top)), values[top][::-1])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"f{index}" for index in top[::-1]])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
