from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from .config import T2Config
from .federated import evaluate_external_after_selection, train_select
from .folds import INTERNAL_TEST_YEAR, loco_spec
from .full_data import (
    load_external_2025,
    load_harmonized_manifest,
    load_source_city_bundle,
)
from .preprocessing import build_fixed_preprocessor, feature_manifest
from .provenance import (
    new_manifest_skeleton,
    sha256_file,
    validate_completed_manifest,
)


def _safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def run_full_loco_one(
    *,
    harmonized_dir: Path,
    harmonized_manifest_path: Path,
    held_out_city: str,
    config: T2Config,
    output_dir: Path,
    command: str,
) -> dict[str, object]:
    """Run one full-data LOCO configuration with delayed target loading.

    Only source-city files are opened and hashed before source-only training and
    checkpoint selection. The held-out city's 2025 file is opened only after
    the selected checkpoint is frozen.
    """

    started = time.perf_counter()
    config.validate()
    if config.training_budget != "fixed_total_effective_epochs":
        raise ValueError(
            "full-data LOCO requires fixed_total_effective_epochs"
        )

    spec = loco_spec(held_out_city)
    preprocessor = build_fixed_preprocessor()
    harmonized_manifest = load_harmonized_manifest(
        harmonized_manifest_path
    )

    source_splits = {}
    source_summaries: dict[str, object] = {}
    source_hashes: dict[str, object] = {}
    source_coverage: dict[str, list[str]] = {}

    # Physically load/hash source cities only.
    for city in spec.source_cities:
        bundle = load_source_city_bundle(
            harmonized_dir=harmonized_dir,
            harmonized_manifest=harmonized_manifest,
            city=city,
            preprocessor=preprocessor,
        )
        source_splits[city] = bundle.splits
        source_summaries[city] = bundle.summaries
        source_hashes[city] = bundle.input_sha256_by_year
        source_coverage[city] = bundle.coverage_notes

    manifest = new_manifest_skeleton(
        config=config.as_dict(),
        fold=spec.name,
        held_out_city=held_out_city,
        source_cities=list(spec.source_cities),
        command=command,
    )
    manifest["input_sha256"] = dict(source_hashes)
    manifest["row_counts"] = {
        city: {
            name: len(dataset)
            for name, dataset in splits.items()
        }
        for city, splits in source_splits.items()
    }
    manifest["date_ranges"] = dict(source_summaries)
    manifest["preprocessor_fingerprint"] = preprocessor.fingerprint
    manifest["features"] = feature_manifest(preprocessor)
    manifest["source_coverage_notes"] = source_coverage
    manifest["external_evaluation_contract"] = {
        "year": INTERNAL_TEST_YEAR,
        "scope": (
            "held-out city eligible administrative completion/closure "
            "observations created in 2025"
        ),
        "loaded_after_source_selection": True,
        "reason": (
            "matches the source-city internal-test calendar year so the "
            "privacy-transfer penalty does not mix temporal windows"
        ),
    }

    # Source-only fitting, DP accounting, validation, and checkpoint selection.
    selected = train_select(source_splits, preprocessor, config)

    # Only now may target bytes be opened or hashed.
    external_dataset, external_summary, target_hashes, target_notes = (
        load_external_2025(
            harmonized_dir=harmonized_dir,
            harmonized_manifest=harmonized_manifest,
            city=held_out_city,
            preprocessor=preprocessor,
        )
    )
    manifest["input_sha256"][held_out_city] = target_hashes
    external_metrics = evaluate_external_after_selection(
        selected,
        external_dataset,
        preprocessor,
    )
    manifest["row_counts"][held_out_city] = {
        "external_2025": len(external_dataset)
    }
    manifest["date_ranges"][held_out_city] = {
        "external_2025": external_summary
    }
    manifest["external_coverage_notes"] = target_notes
    manifest["privacy"] = selected["privacy_by_city"]
    manifest["training"] = {
        "algorithm": "fedavg",
        "primary_target": "log1p(resolution_hours)",
        "estimand_interpretation": (
            "administrative closure/completion duration conditional on "
            "observed completion"
        ),
        "loss": "SmoothL1",
        "best_round": int(selected["best_round"]),
        "source_val_best_macro_mae_log1p_hours": float(
            selected["source_val_best_macro_mae_log1p_hours"]
        ),
        "would_stop_round": selected["would_stop_round"],
        "round_cap": config.rounds,
        "training_budget": config.training_budget,
        "total_effective_epochs_target": config.total_effective_epochs,
        "budget_by_city": selected["training_budget_by_city"],
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "device": selected["device"],
        "selection_scope": "source-city 2024 validation only",
        "internal_test_scope": "source-city 2025 only",
        "external_test_scope": "held-out city 2025 only",
        "dp_scope": "source-city training rows only",
    }

    out_dir = output_dir / str(manifest["run_uuid"])
    out_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(selected["best_state"], checkpoint_path)
    manifest["checkpoint_sha256"] = sha256_file(checkpoint_path)
    manifest["expected_outputs"] = [
        "checkpoint.pt",
        "run.json",
        "manifest.json",
    ]
    manifest["runtime_seconds"] = float(time.perf_counter() - started)
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["status"] = "completed"
    validate_completed_manifest(manifest)

    privacy_eps = [
        float(report["realized_epsilon"])
        for report in selected["privacy_by_city"].values()
        if report["mode"] == "private"
    ]

    run = {
        "held_out_city": held_out_city,
        "source_cities": list(spec.source_cities),
        "seed": config.seed,
        "mode": config.mode,
        "target_epsilon": config.target_epsilon,
        "fold": spec.name,
        "selection": {
            "best_round": selected["best_round"],
            "source_val_best_macro_mae_log1p_hours": selected[
                "source_val_best_macro_mae_log1p_hours"
            ],
            "would_stop_round": selected["would_stop_round"],
        },
        "internal_by_city": selected["internal_by_city"],
        "source_macro_mae_log1p_hours": selected[
            "source_macro_mae_log1p_hours"
        ],
        "external": external_metrics,
        "external_scope": "held-out city 2025 eligible completion/closure rows",
        "fold_realized_epsilon_max_client": (
            max(privacy_eps) if privacy_eps else float("inf")
        ),
        "manifest": manifest,
    }

    (out_dir / "run.json").write_text(
        json.dumps(_safe(run), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run
