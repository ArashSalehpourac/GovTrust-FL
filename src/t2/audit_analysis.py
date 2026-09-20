from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.t2.data import FINAL_STATUSES, PRIMARY_TARGET
from src.t2.folds import (
    INTERNAL_TEST_START,
    INTERNAL_TEST_YEAR,
    TRAIN_YEARS,
    VALIDATION_START,
    VALIDATION_YEAR,
)
from src.t2.preprocessing import (
    CAT_COLS,
    FORBIDDEN_PRIMARY,
    NUM_COLS,
    TEXT_COL,
    build_fixed_preprocessor,
    feature_manifest,
)
from src.t2.provenance import git_state, sha256_file

AUDIT_COLUMNS = [
    "request_id",
    "created_date",
    "closed_date",
    "status",
    "category",
    "descriptor",
    "city",
]

DURATION_DIAGNOSTIC_HOURS = {
    "gt_30_days": 30.0 * 24.0,
    "gt_180_days": 180.0 * 24.0,
    "gt_365_days": 365.0 * 24.0,
}


def _blank(series: pd.Series) -> pd.Series:
    values = series.astype("string")
    return values.isna() | values.fillna("").str.strip().eq("")


def _manifest_outputs(manifest: dict[str, object]) -> dict[tuple[str, int], dict[str, object]]:
    rows = manifest.get("outputs")
    if not isinstance(rows, list) or len(rows) != 20:
        raise RuntimeError("harmonized manifest must contain exactly 20 outputs")
    out: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("harmonized manifest output must be an object")
        key = (str(row["city"]), int(row["year"]))
        if key in out:
            raise RuntimeError(f"duplicate harmonized manifest key: {key}")
        out[key] = row
    return out


def _structural_files(report: dict[str, object]) -> dict[tuple[str, int], dict[str, object]]:
    if report.get("gate") != "PASS":
        raise RuntimeError("structural harmonized audit must PASS before analysis-frame audit")
    rows = report.get("files")
    if not isinstance(rows, list) or len(rows) != 20:
        raise RuntimeError("structural audit must contain exactly 20 files")
    out: dict[tuple[str, int], dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("structural audit file row must be an object")
        key = (str(row["city"]), int(row["year"]))
        if key in out:
            raise RuntimeError(f"duplicate structural-audit key: {key}")
        out[key] = row
    return out


def audit_analysis_file(
    *,
    path: Path,
    output_entry: dict[str, object],
    structural_entry: dict[str, object],
    batch_size: int = 100_000,
) -> dict[str, object]:
    city = str(output_entry["city"])
    year = int(output_entry["year"])
    blockers: list[str] = []

    if not path.is_file():
        return {
            "city": city,
            "year": year,
            "filename": path.name,
            "gate": "FAIL",
            "blockers": ["missing harmonized file"],
        }

    actual_sha = sha256_file(path)
    actual_size = path.stat().st_size
    if actual_sha != str(output_entry["harmonized_sha256"]):
        blockers.append("SHA256 mismatch vs harmonized execution manifest")
    if actual_size != int(output_entry["harmonized_size_bytes"]):
        blockers.append("size mismatch vs harmonized execution manifest")

    parquet = pq.ParquetFile(path)
    missing_columns = set(AUDIT_COLUMNS) - set(parquet.schema_arrow.names)
    if missing_columns:
        blockers.append(f"missing audit columns: {sorted(missing_columns)}")

    rows = 0
    status_counts: Counter[str] = Counter()
    final_status_rows = 0
    final_missing_or_invalid_outcome = 0
    final_negative_duration = 0
    eligible_resolved_rows = 0
    closed_before_created_all = 0
    descriptor_missing_eligible = 0
    category_missing_eligible = 0
    boundary_purge_rows = 0
    duration_counts = {name: 0 for name in DURATION_DIAGNOSTIC_HOURS}
    max_resolution_hours: float | None = None

    if not missing_columns:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=AUDIT_COLUMNS):
            frame = batch.to_pandas()
            rows += len(frame)

            created = pd.to_datetime(
                frame["created_date"], errors="coerce", utc=True, format="mixed"
            )
            closed = pd.to_datetime(
                frame["closed_date"], errors="coerce", utc=True, format="mixed"
            )
            status = (
                frame["status"]
                .astype("string")
                .fillna("")
                .str.strip()
                .str.lower()
            )
            status_labels = status.replace("", "<blank>")
            status_counts.update(str(value) for value in status_labels)

            final = status.isin(FINAL_STATUSES)
            paired = created.notna() & closed.notna()
            negative_all = paired & closed.lt(created)
            negative_final = final & negative_all
            eligible = final & paired & ~negative_all

            final_status_rows += int(final.sum())
            final_missing_or_invalid_outcome += int((final & ~paired).sum())
            final_negative_duration += int(negative_final.sum())
            eligible_resolved_rows += int(eligible.sum())
            closed_before_created_all += int(negative_all.sum())

            descriptor_missing_eligible += int((eligible & _blank(frame["descriptor"])).sum())
            category_missing_eligible += int((eligible & _blank(frame["category"])).sum())

            if year in TRAIN_YEARS:
                boundary_purge_rows += int(
                    (eligible & closed.ge(VALIDATION_START)).sum()
                )
            elif year == VALIDATION_YEAR:
                boundary_purge_rows += int(
                    (eligible & closed.ge(INTERNAL_TEST_START)).sum()
                )

            if eligible.any():
                hours = (
                    (closed[eligible] - created[eligible]).dt.total_seconds() / 3600.0
                )
                local_max = float(hours.max())
                max_resolution_hours = (
                    local_max
                    if max_resolution_hours is None
                    else max(max_resolution_hours, local_max)
                )
                for name, threshold in DURATION_DIAGNOSTIC_HOURS.items():
                    duration_counts[name] += int(hours.gt(threshold).sum())

    if rows != int(output_entry["rows"]):
        blockers.append("row count mismatch vs harmonized execution manifest")
    if rows != int(structural_entry["rows"]):
        blockers.append("row count mismatch vs structural audit")
    if final_status_rows != int(structural_entry["final_status_rows"]):
        blockers.append("final-status count differs from structural audit")
    if eligible_resolved_rows != int(structural_entry["eligible_resolved_rows"]):
        blockers.append("eligible-resolved count differs from structural audit")
    if closed_before_created_all != int(structural_entry["closed_before_created"]):
        blockers.append("closed-before-created count differs from structural audit")

    decomposition = (
        (rows - final_status_rows)
        + final_missing_or_invalid_outcome
        + final_negative_duration
        + eligible_resolved_rows
    )
    if decomposition != rows:
        blockers.append("analysis-frame exclusion decomposition does not sum to input rows")

    return {
        "city": city,
        "year": year,
        "filename": path.name,
        "gate": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "rows": rows,
        "sha256": actual_sha,
        "size_bytes": actual_size,
        "status_counts_normalized": dict(sorted(status_counts.items())),
        "final_status_rows": final_status_rows,
        "excluded_nonfinal_status": rows - final_status_rows,
        "excluded_final_missing_or_invalid_outcome": final_missing_or_invalid_outcome,
        "excluded_final_negative_duration": final_negative_duration,
        "closed_before_created_all_statuses": closed_before_created_all,
        "eligible_resolved_rows": eligible_resolved_rows,
        "eligible_resolved_fraction": (
            float(eligible_resolved_rows / rows) if rows else 0.0
        ),
        "descriptor_missing_among_eligible": descriptor_missing_eligible,
        "category_missing_among_eligible": category_missing_eligible,
        "boundary_purge_rows": boundary_purge_rows,
        "duration_diagnostics_among_eligible": duration_counts,
        "max_resolution_hours": max_resolution_hours,
        "coverage_note": output_entry.get("coverage_note"),
    }


def _aggregate_city(rows: list[dict[str, object]]) -> dict[str, object]:
    ordered = sorted(rows, key=lambda row: int(row["year"]))
    city = str(ordered[0]["city"])
    status_counts: Counter[str] = Counter()
    for row in ordered:
        status_counts.update(
            {
                str(key): int(value)
                for key, value in dict(row["status_counts_normalized"]).items()
            }
        )

    by_year = {int(row["year"]): row for row in ordered}
    train_candidate = sum(
        int(by_year[year]["eligible_resolved_rows"]) for year in TRAIN_YEARS
    )
    train_purge = sum(int(by_year[year]["boundary_purge_rows"]) for year in TRAIN_YEARS)
    validation_candidate = int(by_year[VALIDATION_YEAR]["eligible_resolved_rows"])
    validation_purge = int(by_year[VALIDATION_YEAR]["boundary_purge_rows"])
    internal_test = int(by_year[INTERNAL_TEST_YEAR]["eligible_resolved_rows"])

    return {
        "city": city,
        "rows": sum(int(row["rows"]) for row in ordered),
        "eligible_resolved_rows": sum(
            int(row["eligible_resolved_rows"]) for row in ordered
        ),
        "status_counts_normalized": dict(sorted(status_counts.items())),
        "analysis_frame": {
            "excluded_nonfinal_status": sum(
                int(row["excluded_nonfinal_status"]) for row in ordered
            ),
            "excluded_final_missing_or_invalid_outcome": sum(
                int(row["excluded_final_missing_or_invalid_outcome"])
                for row in ordered
            ),
            "excluded_final_negative_duration": sum(
                int(row["excluded_final_negative_duration"]) for row in ordered
            ),
            "descriptor_missing_among_eligible": sum(
                int(row["descriptor_missing_among_eligible"]) for row in ordered
            ),
            "category_missing_among_eligible": sum(
                int(row["category_missing_among_eligible"]) for row in ordered
            ),
        },
        "purged_source_chronology": {
            "train_candidate_2021_2023": train_candidate,
            "train_purged_outcome_crossing_2024_boundary": train_purge,
            "train_after_purge": train_candidate - train_purge,
            "validation_candidate_2024": validation_candidate,
            "validation_purged_outcome_crossing_2025_boundary": validation_purge,
            "validation_after_purge": validation_candidate - validation_purge,
            "internal_test_2025": internal_test,
        },
        "coverage_notes": [
            str(row["coverage_note"])
            for row in ordered
            if row.get("coverage_note")
        ],
    }


def audit_analysis_frames(
    *,
    harmonized_dir: Path,
    harmonized_manifest_path: Path,
    structural_audit_path: Path,
    report_path: Path,
    batch_size: int = 100_000,
) -> dict[str, object]:
    manifest = json.loads(harmonized_manifest_path.read_text(encoding="utf-8"))
    structural = json.loads(structural_audit_path.read_text(encoding="utf-8"))

    outputs = _manifest_outputs(manifest)
    structural_files = _structural_files(structural)
    if set(outputs) != set(structural_files):
        raise RuntimeError("harmonized manifest and structural audit keys differ")

    file_rows: list[dict[str, object]] = []
    for key in sorted(outputs):
        entry = outputs[key]
        path = harmonized_dir / str(entry["harmonized_filename"])
        file_rows.append(
            audit_analysis_file(
                path=path,
                output_entry=entry,
                structural_entry=structural_files[key],
                batch_size=batch_size,
            )
        )

    blockers = [
        f"{row['city']} {row['year']}: {message}"
        for row in file_rows
        for message in row.get("blockers", [])
    ]

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in file_rows:
        grouped.setdefault(str(row["city"]), []).append(row)
    cities = [_aggregate_city(grouped[city]) for city in sorted(grouped)]

    for city in cities:
        split = dict(city["purged_source_chronology"])
        if int(split["train_after_purge"]) < 1:
            blockers.append(f"{city['city']}: empty training split after purge")
        if int(split["validation_after_purge"]) < 1:
            blockers.append(f"{city['city']}: empty validation split after purge")
        if int(split["internal_test_2025"]) < 1:
            blockers.append(f"{city['city']}: empty internal-test split")

    preprocessor = build_fixed_preprocessor()
    feature_info = feature_manifest(preprocessor)
    primary_inputs = {TEXT_COL, *CAT_COLS, *NUM_COLS}
    forbidden_outcome_inputs = {
        "status",
        "closed_date",
        "resolution_hours",
        PRIMARY_TARGET,
    }
    if primary_inputs & forbidden_outcome_inputs:
        blockers.append("primary predictor set contains an outcome/post-outcome field")
    if primary_inputs & FORBIDDEN_PRIMARY:
        blockers.append("primary predictor set contains a forbidden jurisdiction shortcut")

    coverage_notes = [
        note
        for city in cities
        for note in city["coverage_notes"]
    ]

    audit_sha, audit_dirty = git_state()
    if audit_sha == "UNKNOWN":
        blockers.append("analysis audit code SHA is UNKNOWN")
    if audit_dirty:
        blockers.append("analysis audit checkout is dirty")

    report = {
        "protocol": "t2_chronology_leakage_data_quality_audit_v1",
        "gate": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "training_authorized_by_this_report": False,
        "ready_for_experiment_design_review": not blockers,
        "harmonized_execution_git_sha": manifest.get("git_sha"),
        "audit_execution_git_sha": audit_sha,
        "audit_git_dirty": audit_dirty,
        "manual_status_semantics_review_required": True,
        "primary_estimand": {
            "population": (
                "municipal service requests reaching a recognized final status "
                "with parseable nonnegative resolution duration"
            ),
            "outcome": "log1p(resolution_hours)",
            "interpretation": (
                "resolution-time reliability conditional on observed resolution; "
                "not a time-to-resolution estimand for all incoming requests"
            ),
            "final_statuses_normalized": sorted(FINAL_STATUSES),
        },
        "analysis_frame_policy": {
            "harmonized_rows_are_never_deleted_or_overwritten": True,
            "exclude_nonfinal_status_in_analysis_frame": True,
            "exclude_missing_or_invalid_closed_date_in_analysis_frame": True,
            "exclude_negative_resolution_duration_in_analysis_frame": True,
            "descriptor_missingness_causes_row_exclusion": False,
            "descriptor_missing_encoding": "empty string into fixed HashingVectorizer",
            "category_missingness_causes_row_exclusion": False,
            "category_missing_encoding": "UNK token into fixed FeatureHasher",
            "no_duration_outlier_threshold_exclusion": True,
        },
        "chronology_policy": {
            "train_created_years": list(TRAIN_YEARS),
            "validation_created_year": VALIDATION_YEAR,
            "internal_test_created_year": INTERNAL_TEST_YEAR,
            "train_outcome_must_precede": VALIDATION_START.isoformat(),
            "validation_outcome_must_precede": INTERNAL_TEST_START.isoformat(),
            "purge_cross_boundary_outcomes": True,
            "split_is_data_independent": True,
        },
        "leakage_policy": {
            "heldout_target_opened_only_after_source_training_and_selection": True,
            "source_only_model_selection": True,
            "source_only_privacy_accounting": True,
            "preprocessing_fit_scope": feature_info["fit_scope"],
            "preprocessor_fingerprint": preprocessor.fingerprint,
            "primary_predictor_inputs": sorted(primary_inputs),
            "explicitly_excluded_primary": feature_info["explicitly_excluded_primary"],
            "forbidden_outcome_or_postoutcome_inputs": sorted(forbidden_outcome_inputs),
        },
        "coverage_and_provenance_notes": coverage_notes,
        "files": file_rows,
        "cities": cities,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
