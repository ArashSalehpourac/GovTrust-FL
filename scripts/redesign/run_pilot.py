"""Run the small redesign pilot: 4 folds x {fedavg, fedprox} x epsilon x seeds.

Usage:
    python scripts/redesign/run_pilot.py --rounds 5 --seeds 0 1 2 --epsilons inf 5 1
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.redesign.config import (  # noqa: E402
    FOLDS_DIR,
    REDESIGN_RESULTS_DIR,
    PilotConfig,
    build_folds,
)
from src.redesign.federated import FederatedConfig, run_federated  # noqa: E402
from src.redesign.folds import load_fold  # noqa: E402


def parse_epsilon(value: str) -> float:
    if value.lower() in {"inf", "infinity", "none"}:
        return float("inf")
    return float(value)


def main() -> int:
    defaults = PilotConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=defaults.rounds)
    parser.add_argument("--local-epochs", type=int, default=defaults.local_epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(defaults.seeds))
    parser.add_argument(
        "--epsilons", type=parse_epsilon, nargs="+", default=list(defaults.epsilons)
    )
    parser.add_argument("--algorithms", nargs="+", default=list(defaults.algorithms))
    parser.add_argument("--delta", type=float, default=defaults.delta)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--proximal-mu", type=float, default=defaults.proximal_mu)
    parser.add_argument("--folds", nargs="+", default=None, help="Fold names to run")
    parser.add_argument("--folds-dir", type=Path, default=FOLDS_DIR)
    parser.add_argument("--results-dir", type=Path, default=REDESIGN_RESULTS_DIR)
    args = parser.parse_args()

    fold_names = args.folds or [spec.name for spec in build_folds()]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = args.results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    subgroup_rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for fold_name in fold_names:
        fold = load_fold(fold_name, args.folds_dir)
        for algorithm in args.algorithms:
            for epsilon in args.epsilons:
                for seed in args.seeds:
                    config = FederatedConfig(
                        algorithm=algorithm,
                        rounds=args.rounds,
                        local_epochs=args.local_epochs,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        proximal_mu=args.proximal_mu,
                        seed=seed,
                        target_epsilon=epsilon,
                        delta=args.delta,
                        max_grad_norm=args.max_grad_norm,
                    )
                    label = (
                        f"{fold_name}__{algorithm}__eps"
                        f"{'inf' if math.isinf(epsilon) else epsilon:g}"
                        if not math.isinf(epsilon)
                        else f"{fold_name}__{algorithm}__epsinf"
                    )
                    label = f"{label}__seed{seed}"
                    print(f"Running {label}", flush=True)
                    result = run_federated(fold, config)
                    (runs_dir / f"{label}.json").write_text(
                        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
                    )
                    rows.extend(_summary_rows(result, label))
                    for entry in result.subgroups:
                        subgroup_rows.append(
                            {
                                "run": label,
                                "fold": result.fold,
                                "held_out_city": result.held_out_city,
                                "algorithm": algorithm,
                                "target_epsilon": epsilon,
                                "seed": seed,
                                **entry,
                            }
                        )
                    print(
                        f"  internal pooled MAE={rows[-1]['internal_pooled_mae_log1p_hours']:.4f} "
                        f"external MAE={rows[-1]['external_mae_log1p_hours']:.4f} "
                        f"({result.runtime_seconds:.1f}s)",
                        flush=True,
                    )

    summary = pd.DataFrame(rows)
    summary_path = args.results_dir / "pilot_summary.csv"
    summary.to_csv(summary_path, index=False)
    if subgroup_rows:
        pd.DataFrame(subgroup_rows).to_csv(
            args.results_dir / "pilot_subgroup_reliability.csv", index=False
        )
    print(
        f"Wrote {summary_path} ({len(summary)} runs) in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return 0


def _summary_rows(result, label: str) -> list[dict[str, object]]:
    internal = result.internal["pooled_internal_test"]
    external = result.external["report"]
    privacy = result.privacy["by_city"]
    epsilons = [
        report["epsilon_spent"] for report in privacy.values() if isinstance(report, dict)
    ]
    noise = [
        report["noise_multiplier"] for report in privacy.values() if isinstance(report, dict)
    ]
    return [
        {
            "run": label,
            "fold": result.fold,
            "held_out_city": result.held_out_city,
            "algorithm": result.config["algorithm"],
            "target_epsilon": result.config["target_epsilon"],
            "delta": result.config["delta"],
            "seed": result.config["seed"],
            "best_round": result.selection["best_round"],
            "dp_enabled": result.privacy["enabled"],
            "max_epsilon_spent": max(epsilons) if epsilons else None,
            "noise_multiplier_min": min(noise) if noise else None,
            "noise_multiplier_max": max(noise) if noise else None,
            "internal_pooled_mae_log1p_hours": internal["primary"]["mae_log1p_hours"],
            "internal_pooled_rmse_log1p_hours": internal["primary"]["rmse_log1p_hours"],
            "internal_pooled_mean_pinball": internal["primary"]["mean_pinball"],
            "internal_pooled_interval_coverage": internal["primary"][
                "interval_empirical_coverage"
            ],
            "internal_pooled_brier": internal["secondary_sla"]["brier"],
            "internal_pooled_ece": internal["secondary_sla"]["ece"],
            "internal_pooled_auroc": internal["secondary_sla"]["auroc"],
            "external_mae_log1p_hours": external["primary"]["mae_log1p_hours"],
            "external_rmse_log1p_hours": external["primary"]["rmse_log1p_hours"],
            "external_mean_pinball": external["primary"]["mean_pinball"],
            "external_interval_coverage": external["primary"]["interval_empirical_coverage"],
            "external_brier": external["secondary_sla"]["brier"],
            "external_ece": external["secondary_sla"]["ece"],
            "external_auroc": external["secondary_sla"]["auroc"],
            "runtime_seconds": result.runtime_seconds,
        }
    ]


if __name__ == "__main__":
    raise SystemExit(main())
