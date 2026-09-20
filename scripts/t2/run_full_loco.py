from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t2.config import CITIES, SEEDS, T2Config, full_loco_plan
from src.t2.full_runner import run_full_loco_one

FULL_LOCO_EXECUTION_ENABLED = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preview or run one full-data four-fold T2 LOCO configuration. "
            "Execution remains fail-closed until the experiment-design gate passes."
        )
    )
    parser.add_argument("--harmonized-dir", required=True)
    parser.add_argument("--harmonized-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--heldout", choices=CITIES, required=True)
    parser.add_argument(
        "--mode",
        choices=["nonprivate", "private", "clipped_no_noise"],
        required=True,
    )
    parser.add_argument("--epsilon", default="inf")
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--epsilon-tolerance", type=float, default=0.05)
    parser.add_argument("--total-effective-epochs", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--print-matrix",
        action="store_true",
        help="print the frozen four-fold matrix and exit",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.print_matrix:
        print(
            json.dumps(
                full_loco_plan(),
                indent=2,
                default=lambda value: (
                    "inf" if value == float("inf") else value
                ),
            )
        )
        return 0

    epsilon = (
        float(args.epsilon)
        if args.mode == "private"
        else float("inf")
    )
    if args.mode == "private" and epsilon not in {1.0, 5.0}:
        raise SystemExit("full T2 matrix permits private epsilon 1 or 5")
    if args.mode != "private" and args.epsilon.lower() not in {
        "inf",
        "infinity",
    }:
        raise SystemExit("non-private controls require epsilon infinity")

    config = T2Config(
        mode=args.mode,
        target_epsilon=epsilon,
        rounds=args.rounds,
        local_epochs=1,
        training_budget="fixed_total_effective_epochs",
        total_effective_epochs=args.total_effective_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        checkpoint_patience=args.patience,
        stop_on_patience=False,
        epsilon_tolerance=args.epsilon_tolerance,
        device=args.device,
    )
    config.validate()

    preview = {
        "held_out_city": args.heldout,
        "source_cities": [city for city in CITIES if city != args.heldout],
        "mode": args.mode,
        "target_epsilon": (
            "inf" if epsilon == float("inf") else epsilon
        ),
        "seed": args.seed,
        "rounds": args.rounds,
        "training_budget": config.training_budget,
        "total_effective_epochs": config.total_effective_epochs,
        "batch_size": config.batch_size,
        "external_primary_year": 2025,
        "execution_enabled": FULL_LOCO_EXECUTION_ENABLED,
    }
    print(json.dumps(preview, indent=2))

    if not args.execute:
        print("No training started.")
        return 0

    if not FULL_LOCO_EXECUTION_ENABLED:
        raise SystemExit(
            "Full-data LOCO execution remains fail-closed until the "
            "experiment-design/feasibility gate is explicitly passed."
        )

    run_full_loco_one(
        harmonized_dir=Path(args.harmonized_dir).expanduser().resolve(),
        harmonized_manifest_path=Path(
            args.harmonized_manifest
        ).expanduser().resolve(),
        held_out_city=args.heldout,
        config=config,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        command=" ".join(sys.argv),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
