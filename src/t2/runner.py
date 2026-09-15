from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from .config import T2Config
from .data import frame_summary, prepare_resolved_frame
from .federated import evaluate_external_after_selection, train_select
from .folds import loco_spec, split_source_cities
from .preprocessing import build_fixed_preprocessor, feature_manifest
from .provenance import new_manifest_skeleton, sha256_file, validate_completed_manifest

ExternalLoader = Callable[[], tuple[pd.DataFrame, str]]


def _safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _date_ranges(source_splits: Mapping[str, Mapping[str, pd.DataFrame]]) -> dict[str, object]:
    return {
        city: {name: frame_summary(frame) for name, frame in splits.items()}
        for city, splits in source_splits.items()
    }


def run_one(
    *,
    raw_source_frames: Mapping[str, pd.DataFrame],
    source_input_sha256: Mapping[str, str],
    external_loader: ExternalLoader,
    held_out_city: str,
    config: T2Config,
    output_dir: str | Path,
    command: str,
) -> dict[str, object]:
    """One provenance-complete run with physically delayed held-out-data loading."""

    started = time.perf_counter()
    spec = loco_spec(held_out_city)
    config.validate()
    if set(raw_source_frames) != set(spec.source_cities):
        raise ValueError("runner must receive exactly the three SOURCE frames")
    if set(source_input_sha256) != set(spec.source_cities):
        raise ValueError("source SHA256 required for all and only source cities")

    prepared_source = {
        city: prepare_resolved_frame(raw_source_frames[city])
        for city in spec.source_cities
    }
    source_splits = split_source_cities(prepared_source, spec)
    preprocessor = build_fixed_preprocessor()

    manifest = new_manifest_skeleton(
        config=config.as_dict(),
        fold=spec.name,
        held_out_city=held_out_city,
        source_cities=list(spec.source_cities),
        command=command,
    )
    manifest["input_sha256"] = dict(source_input_sha256)
    manifest["row_counts"] = {
        city: {name: len(frame) for name, frame in splits.items()}
        for city, splits in source_splits.items()
    }
    manifest["date_ranges"] = _date_ranges(source_splits)
    manifest["preprocessor_fingerprint"] = preprocessor.fingerprint
    manifest["features"] = feature_manifest(preprocessor)

    # Source-only training and checkpoint selection complete before external_loader is called.
    selected = train_select(source_splits, preprocessor, config)

    external_raw, external_sha256 = external_loader()
    manifest["input_sha256"][held_out_city] = external_sha256
    external_frame = prepare_resolved_frame(external_raw)
    external_metrics = evaluate_external_after_selection(
        selected,
        external_frame,
        preprocessor,
    )
    manifest["row_counts"][held_out_city] = {"external": len(external_frame)}
    manifest["date_ranges"][held_out_city] = {
        "external": frame_summary(external_frame)
    }
    manifest["privacy"] = selected["privacy_by_city"]
    manifest["training"] = {
        "algorithm": "fedavg",
        "primary_target": "log1p(resolution_hours)",
        "loss": "SmoothL1",
        "best_round": int(selected["best_round"]),
        "source_val_best_macro_mae_log1p_hours": float(
            selected["source_val_best_macro_mae_log1p_hours"]
        ),
        "would_stop_round": selected["would_stop_round"],
        "round_cap": config.rounds,
        "local_epochs": config.local_epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "selection_scope": "source-city validation only",
        "dp_scope": "source-city training rows only",
    }

    out_dir = Path(output_dir) / str(manifest["run_uuid"])
    out_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(selected["best_state"], checkpoint_path)
    manifest["checkpoint_sha256"] = sha256_file(checkpoint_path)
    manifest["expected_outputs"] = ["checkpoint.pt", "run.json", "manifest.json"]
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
