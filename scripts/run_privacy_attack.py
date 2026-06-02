"""Run a simple confidence-based membership-inference evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEDERATED_CLIENTS, ensure_project_dirs  # noqa: E402
from src.config_loader import load_config, resolve_path  # noqa: E402
from src.features import build_xy  # noqa: E402
from src.metrics import predict_scores  # noqa: E402
from src.privacy import attack_advantage, membership_inference_auc  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    ensure_project_dirs()
    splits_dir = resolve_path(config, "data.splits_dir")
    tables_dir = resolve_path(config, "outputs.tables_dir")
    models_dir = resolve_path(config, "outputs.models_dir")
    model_path = args.model_path or models_dir / f"centralized_{config['model']['type']}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found for privacy attack: {model_path}")

    model = joblib.load(model_path)
    train = pd.concat(
        [pd.read_parquet(splits_dir / city / f"{city}_train.parquet") for city in FEDERATED_CLIENTS],
        ignore_index=True,
    )
    nonmember = pd.concat(
        [pd.read_parquet(splits_dir / city / f"{city}_test.parquet") for city in FEDERATED_CLIENTS],
        ignore_index=True,
    )
    member_scores = confidence_scores(model, train)
    nonmember_scores = confidence_scores(model, nonmember)
    attack_auc = membership_inference_auc(member_scores, nonmember_scores)
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    scores = np.concatenate([member_scores, nonmember_scores])
    thresholds = np.quantile(scores, np.linspace(0.05, 0.95, 19))
    attack_accuracy = max(float(((scores >= threshold).astype(int) == labels).mean()) for threshold in thresholds)
    row = {
        "model": model_path.stem,
        "attack_accuracy": attack_accuracy,
        "attack_auc": attack_auc,
        "attack_advantage": attack_advantage(attack_auc),
        "member_rows": len(member_scores),
        "nonmember_rows": len(nonmember_scores),
    }
    tables_dir.mkdir(parents=True, exist_ok=True)
    output_path = tables_dir / "privacy_attack_metrics.csv"
    output = pd.DataFrame([row])
    output.to_csv(output_path, index=False)
    output.to_csv(tables_dir / "membership_inference_results.csv", index=False)
    print(f"Wrote privacy attack metrics: {output_path}")
    return 0


def confidence_scores(model, frame: pd.DataFrame) -> np.ndarray:
    x, _ = build_xy(frame)
    scores = predict_scores(model, x)
    return np.maximum(scores, 1.0 - scores)


if __name__ == "__main__":
    raise SystemExit(main())
