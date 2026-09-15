from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping

import numpy as np

from .provenance import validate_completed_manifest


def privacy_transfer_penalty(private_run: Mapping[str, object], baseline_run: Mapping[str, object]) -> dict[str, float]:
    """PTP = external privacy cost - internal privacy cost."""
    p_ext = float(private_run["external"]["mae_log1p_hours"])
    b_ext = float(baseline_run["external"]["mae_log1p_hours"])
    p_src = float(private_run["source_macro_mae_log1p_hours"])
    b_src = float(baseline_run["source_macro_mae_log1p_hours"])
    external_cost = p_ext - b_ext
    internal_cost = p_src - b_src
    return {
        "external_privacy_cost": external_cost,
        "internal_privacy_cost": internal_cost,
        "ptp": external_cost - internal_cost,
    }


def validate_run_record(run: Mapping[str, object]) -> None:
    manifest = run.get("manifest")
    if not isinstance(manifest, Mapping):
        raise TypeError("run lacks manifest mapping")
    validate_completed_manifest(manifest)
    for key in ("external", "source_macro_mae_log1p_hours", "held_out_city", "seed", "mode"):
        if key not in run:
            raise ValueError(f"run missing {key}")


def aggregate_ptp(runs: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Pair each private run to same-fold/same-seed nonprivate infinity baseline."""
    runs = list(runs)
    for run in runs:
        validate_run_record(run)
    baselines: dict[tuple[str, int], Mapping[str, object]] = {}
    for run in runs:
        if run["mode"] == "nonprivate":
            baselines[(str(run["held_out_city"]), int(run["seed"]))] = run

    rows: list[dict[str, object]] = []
    for run in runs:
        if run["mode"] != "private":
            continue
        key = (str(run["held_out_city"]), int(run["seed"]))
        if key not in baselines:
            raise ValueError(f"missing infinity baseline for {key}")
        effects = privacy_transfer_penalty(run, baselines[key])
        rows.append({
            "held_out_city": key[0],
            "seed": key[1],
            "target_epsilon": float(run["target_epsilon"]),
            **effects,
        })
    return rows


def summarize_ptp(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    groups: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        groups[float(row["target_epsilon"])].append(float(row["ptp"]))
    return [
        {
            "target_epsilon": eps,
            "n_fold_seed_effects": len(values),
            "mean_ptp": float(np.mean(values)),
            "median_ptp": float(np.median(values)),
            "sd_ptp": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min_ptp": float(np.min(values)),
            "max_ptp": float(np.max(values)),
            "positive_fraction": float(np.mean(np.asarray(values) > 0)),
        }
        for eps, values in sorted(groups.items(), reverse=True)
    ]
