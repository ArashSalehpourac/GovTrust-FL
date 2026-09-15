from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import T2Config, diagnostic_plan
from .folds import loco_spec
from .provenance import sha256_file
from .runner import run_one


def _read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".gz"} or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"unsupported input format: {path}")


def _city_paths(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise ValueError("--city expects city=/path/to/file")
        city, raw_path = item.split("=", 1)
        out[city.strip()] = Path(raw_path).expanduser().resolve()
    return out


def cmd_plan(_: argparse.Namespace) -> int:
    print(json.dumps(diagnostic_plan(), indent=2, default=lambda x: "inf" if x == float("inf") else x))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    paths = _city_paths(args.city)
    spec = loco_spec(args.held_out)
    expected = {*spec.source_cities, spec.held_out_city}
    if set(paths) != expected:
        raise SystemExit(f"exactly these cities are required: {sorted(expected)}")
    for path in paths.values():
        if not path.is_file():
            raise SystemExit(f"missing input file: {path}")

    mode = args.mode
    epsilon = float(args.epsilon) if mode == "private" else float("inf")
    config = T2Config(
        mode=mode,
        target_epsilon=epsilon,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_grad_norm=args.max_grad_norm,
        checkpoint_patience=args.patience,
        stop_on_patience=False,
        epsilon_tolerance=args.epsilon_tolerance,
    )
    config.validate()

    # Physically open/hash only source-city data before training/selection.
    source_frames = {city: _read_frame(paths[city]) for city in spec.source_cities}
    source_hashes = {city: sha256_file(paths[city]) for city in spec.source_cities}

    external_called = False

    def external_loader() -> tuple[pd.DataFrame, str]:
        nonlocal external_called
        if external_called:
            raise RuntimeError("external loader may be called only once")
        external_called = True
        path = paths[spec.held_out_city]
        return _read_frame(path), sha256_file(path)

    run = run_one(
        raw_source_frames=source_frames,
        source_input_sha256=source_hashes,
        external_loader=external_loader,
        held_out_city=spec.held_out_city,
        config=config,
        output_dir=Path(args.output_dir),
        command=" ".join(sys.argv),
    )
    print(json.dumps({
        "run_uuid": run["manifest"]["run_uuid"],
        "held_out_city": run["held_out_city"],
        "mode": run["mode"],
        "target_epsilon": "inf" if run["target_epsilon"] == float("inf") else run["target_epsilon"],
        "best_round": run["selection"]["best_round"],
        "source_macro_mae_log1p_hours": run["source_macro_mae_log1p_hours"],
        "external_mae_log1p_hours": run["external"]["mae_log1p_hours"],
        "output_dir": str(Path(args.output_dir) / str(run["manifest"]["run_uuid"])),
    }, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="T2 diagnostic validity gate")
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="print the frozen diagnostic matrix; no training")
    plan.set_defaults(func=cmd_plan)

    run = sub.add_parser("run", help="run exactly one authorized configuration")
    run.add_argument("--city", action="append", required=True, help="city=/path/to/file; repeat four times")
    run.add_argument("--heldout", required=True, choices=["boston", "los_angeles"], help="diagnostic gate permits Boston/LA only")
    run.add_argument("--mode", required=True, choices=["nonprivate", "private", "clipped_no_noise"])
    run.add_argument("--epsilon", default="inf")
    run.add_argument("--seed", type=int, required=True, choices=[0, 1, 2])
    run.add_argument("--rounds", type=int, default=20)
    run.add_argument("--local-epochs", type=int, default=1)
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--learning-rate", type=float, default=0.02)
    run.add_argument("--max-grad-norm", type=float, default=1.0)
    run.add_argument("--patience", type=int, default=5)
    run.add_argument("--epsilon-tolerance", type=float, default=0.05)
    run.add_argument("--output-dir", required=True)
    run.set_defaults(func=cmd_run)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "run":
        if args.mode == "private" and args.epsilon not in {"1", "1.0", "5", "5.0"}:
            raise SystemExit("diagnostic gate permits private epsilon 1 or 5 only")
        if args.mode != "private" and args.epsilon not in {"inf", "infinity"}:
            raise SystemExit("non-private controls must use epsilon infinity")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
