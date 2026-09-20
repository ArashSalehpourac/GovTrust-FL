from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t2.config import (
    CITIES,
    T2Config,
    delta_for_client,
    full_loco_plan,
)
from src.t2.federated import training_step_budget
from src.t2.full_data import (
    load_harmonized_manifest,
    load_source_city_bundle,
)
from src.t2.preprocessing import build_fixed_preprocessor
from src.t2.provenance import git_state


def _analysis_by_city(report: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = report.get("cities")
    if not isinstance(rows, list):
        raise TypeError("analysis audit cities must be a list")
    out: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("analysis audit city row must be an object")
        out[str(row["city"])] = row
    return out


def audit_full_design(
    *,
    harmonized_dir: Path,
    harmonized_manifest_path: Path,
    analysis_audit_path: Path,
    report_path: Path,
    batch_size: int,
    rounds: int,
    total_effective_epochs: float,
) -> dict[str, object]:
    blockers: list[str] = []

    harmonized_manifest = load_harmonized_manifest(harmonized_manifest_path)
    analysis = json.loads(analysis_audit_path.read_text(encoding="utf-8"))
    if analysis.get("gate") != "PASS":
        blockers.append("chronology/leakage/data-quality audit is not PASS")
    if analysis.get("ready_for_experiment_design_review") is not True:
        blockers.append("analysis audit does not authorize experiment-design review")
    by_city = _analysis_by_city(analysis)

    preprocessor = build_fixed_preprocessor()
    config = T2Config(
        rounds=rounds,
        batch_size=batch_size,
        training_budget="fixed_total_effective_epochs",
        total_effective_epochs=total_effective_epochs,
        device="cpu",
    )
    config.validate()

    city_rows: list[dict[str, object]] = []
    resident_by_city: dict[str, int] = {}

    for city in CITIES:
        bundle = load_source_city_bundle(
            harmonized_dir=harmonized_dir,
            harmonized_manifest=harmonized_manifest,
            city=city,
            preprocessor=preprocessor,
        )
        expected = dict(by_city[city]["purged_source_chronology"])
        observed = {
            "train": len(bundle.splits["train"]),
            "val": len(bundle.splits["val"]),
            "internal_test": len(bundle.splits["internal_test"]),
        }
        if observed["train"] != int(expected["train_after_purge"]):
            blockers.append(f"{city}: compact train count differs from analysis audit")
        if observed["val"] != int(expected["validation_after_purge"]):
            blockers.append(f"{city}: compact val count differs from analysis audit")
        if observed["internal_test"] != int(expected["internal_test_2025"]):
            blockers.append(
                f"{city}: compact internal-test count differs from analysis audit"
            )

        dense_rows = sum(
            int(getattr(dataset, "dense_rows_materialized", -1))
            for dataset in bundle.splits.values()
        )
        if dense_rows != 0:
            blockers.append(f"{city}: compact loader materializes dense full rows")

        resident_bytes = sum(
            int(getattr(dataset, "resident_bytes", 0))
            for dataset in bundle.splits.values()
        )
        resident_by_city[city] = resident_bytes

        n_train = observed["train"]
        loader_batches = max(1, math.ceil(n_train / batch_size))
        steps_per_round, planned_steps = training_step_budget(
            loader_batches,
            config,
        )
        delta = delta_for_client(n_train)
        if not delta < 1.0 / n_train:
            blockers.append(f"{city}: delta is not strictly below 1/N")

        city_rows.append(
            {
                "city": city,
                "rows": observed,
                "resident_bytes_all_splits": resident_bytes,
                "resident_gib_all_splits": resident_bytes / (1024**3),
                "dense_rows_materialized": dense_rows,
                "train_loader_batches": loader_batches,
                "steps_per_round": steps_per_round,
                "planned_private_steps": planned_steps,
                "planned_effective_epochs": planned_steps / loader_batches,
                "delta_rule": delta,
                "input_sha256_by_year": bundle.input_sha256_by_year,
                "coverage_notes": bundle.coverage_notes,
            }
        )
        del bundle
        gc.collect()

    fold_rows: list[dict[str, object]] = []
    for held_out in CITIES:
        sources = [city for city in CITIES if city != held_out]
        resident = sum(resident_by_city[city] for city in sources)
        fold_rows.append(
            {
                "held_out_city": held_out,
                "source_cities": sources,
                "compact_resident_bytes_all_source_splits": resident,
                "compact_resident_gib_all_source_splits": resident / (1024**3),
            }
        )

    primary = full_loco_plan(include_clipped_no_noise=False)
    full = full_loco_plan(include_clipped_no_noise=True)
    if len(primary) != 36:
        blockers.append("primary four-fold matrix is not exactly 36 runs")
    if len(full) != 48:
        blockers.append("full matrix with clipped control is not exactly 48 runs")
    if {row["held_out_city"] for row in full} != set(CITIES):
        blockers.append("full matrix does not cover all four held-out cities")

    sha, dirty = git_state()
    if sha == "UNKNOWN":
        blockers.append("experiment-design audit code SHA is UNKNOWN")
    if dirty:
        blockers.append("experiment-design audit checkout is dirty")

    report = {
        "protocol": "t2_full_data_experiment_design_audit_v1",
        "gate": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "audit_execution_git_sha": sha,
        "audit_git_dirty": dirty,
        "harmonized_execution_git_sha": harmonized_manifest.get("git_sha"),
        "analysis_audit_execution_git_sha": analysis.get(
            "audit_execution_git_sha"
        ),
        "preprocessor_fingerprint": preprocessor.fingerprint,
        "model_input_dim": preprocessor.output_dim,
        "training_budget": {
            "rounds": rounds,
            "batch_size": batch_size,
            "mode": "fixed_total_effective_epochs",
            "target_total_effective_epochs": total_effective_epochs,
            "checkpoint_selection": "source 2024 validation only",
            "internal_evaluation": "source 2025 only",
            "external_primary_evaluation": "held-out city 2025 only",
        },
        "matrix": {
            "primary_runs": len(primary),
            "primary_conditions": [
                "nonprivate epsilon=inf",
                "private epsilon=5",
                "private epsilon=1",
            ],
            "seeds": [0, 1, 2],
            "folds": list(CITIES),
            "clipped_no_noise_ablation_runs": len(full) - len(primary),
            "total_with_ablation": len(full),
        },
        "privacy_contract": {
            "unit": "one source-city training service-request row",
            "accountant": "Opacus RDP",
            "continuous_per_client_across_rounds": True,
            "delta_strictly_below_inverse_n": True,
            "fold_epsilon": "max realized epsilon across source clients",
        },
        "heldout_isolation": {
            "target_bytes_opened_before_source_selection": False,
            "target_2025_loaded_only_after_selected_checkpoint": True,
            "target_used_for_tuning_or_privacy_accounting": False,
        },
        "cities": city_rows,
        "fold_memory": fold_rows,
        "training_authorized_by_this_report": False,
        "ready_to_unlock_full_loco_execution": not blockers,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit T2 full-data four-fold LOCO execution feasibility without training"
        )
    )
    parser.add_argument("--harmonized-dir", required=True)
    parser.add_argument("--harmonized-manifest", required=True)
    parser.add_argument("--analysis-audit", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--total-effective-epochs", type=float, default=1.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = audit_full_design(
        harmonized_dir=Path(args.harmonized_dir).expanduser().resolve(),
        harmonized_manifest_path=Path(
            args.harmonized_manifest
        ).expanduser().resolve(),
        analysis_audit_path=Path(args.analysis_audit).expanduser().resolve(),
        report_path=Path(args.report).expanduser().resolve(),
        batch_size=args.batch_size,
        rounds=args.rounds,
        total_effective_epochs=args.total_effective_epochs,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
