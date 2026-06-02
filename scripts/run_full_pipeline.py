"""Run the synthetic validation pipeline end to end.

This command validates code behavior only. It writes synthetic inputs and
outputs to an isolated work directory so real municipal downloads under
``data/raw`` are never mixed into smoke-test runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    """Parse full-pipeline smoke-run arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-per-city", type=int, default=100)
    parser.add_argument("--work-dir", type=Path, default=None, help="Optional isolated output workspace.")
    return parser.parse_args()


def main() -> int:
    """Run the reproducible synthetic validation workflow."""

    args = parse_args()
    work_dir = args.work_dir or Path(tempfile.mkdtemp(prefix="govtrust_fl_pipeline_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / "synthetic_pipeline_config.yaml"
    write_synthetic_config(config_path, work_dir)

    commands = [
        ["scripts/make_synthetic_data.py", "--rows-per-city", str(args.rows_per_city), "--output-dir", str(work_dir / "data" / "raw")],
        ["scripts/prepare_data.py", "--config", str(config_path)],
        ["scripts/run_baselines.py", "--config", str(config_path)],
        ["scripts/run_federated.py", "--config", str(config_path), "--algorithm", "fedavg"],
        ["scripts/run_privacy_attack.py", "--config", str(config_path)],
        ["scripts/run_xai_pedi.py", "--config", str(config_path)],
        ["scripts/run_trustworthiness_eval.py", "--config", str(config_path)],
        ["scripts/make_transparency_record.py", "--config", str(config_path)],
    ]
    for command in commands:
        print("Running:", " ".join(command), flush=True)
        completed = subprocess.run([sys.executable, *command], cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode

    print(f"Synthetic validation workspace: {work_dir}", flush=True)
    print("Synthetic outputs are code-validation artifacts, not manuscript findings.", flush=True)
    return 0


def write_synthetic_config(path: Path, work_dir: Path) -> None:
    """Write an isolated config for synthetic validation runs."""

    def rel(name: str) -> str:
        return str((work_dir / name).resolve())

    config = {
        "project": {"name": "GovTrust-FL-synthetic-validation", "random_seed": 42},
        "cities": {
            "training": ["nyc", "chicago", "boston"],
            "external_validation": "los_angeles",
        },
        "data": {
            "raw_dir": rel("data/raw"),
            "processed_dir": rel("data/processed"),
            "splits_dir": rel("data/splits"),
            "min_category_samples": 1,
            "test_size": 0.15,
            "validation_size": 0.15,
        },
        "schema": {
            "required_columns": [
                "request_id",
                "created_date",
                "closed_date",
                "status",
                "category",
                "descriptor",
                "agency",
                "latitude",
                "longitude",
                "area",
                "city",
            ]
        },
        "target": {
            "column": "delayed",
            "resolution_hours_column": "resolution_hours",
            "delay_threshold": "city_category_q3",
            "service_routing_target": "service_routing_target",
        },
        "model": {"type": "logistic_regression", "random_state": 42},
        "fl": {
            "clients": ["nyc", "chicago", "boston"],
            "rounds": 1,
            "local_epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.001,
            "hidden_dim": 16,
            "fedprox_mu": 0.01,
        },
        "dp": {
            "enabled": False,
            "noise_multiplier": 1.0,
            "max_grad_norm": 1.0,
            "delta": 1e-5,
        },
        "secure_aggregation": {"enabled": False, "overhead_factor": 0.25},
        "calibration": {"n_bins": 5},
        "fairness": {"group_columns": ["city", "category", "area"]},
        "xai": {"sample_size": 20, "top_k": 5},
        "tai_score": {
            "weights": {
                "predictive_utility": 0.25,
                "privacy_protection": 0.20,
                "explanation_stability": 0.20,
                "fairness": 0.15,
                "efficiency": 0.10,
                "calibration": 0.10,
            }
        },
        "outputs": {
            "tables_dir": rel("results/tables"),
            "figures_dir": rel("results/figures"),
            "logs_dir": rel("results/logs"),
            "models_dir": rel("results/models"),
            "transparency_record": rel("results/algorithmic_transparency_record.md"),
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
