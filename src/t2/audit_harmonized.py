from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.t2.harmonize_frozen import HARMONIZED_COLUMNS
from src.t2.provenance import sha256_file

FINAL_STATUSES = {"closed", "completed", "complete", "resolved"}


def _blank(series: pd.Series) -> pd.Series:
    values = series.astype("string")
    return values.isna() | values.fillna("").str.strip().eq("")


def _pct(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _expected_by_key(raw_manifest: dict[str, object]) -> dict[tuple[str, int], dict[str, object]]:
    snapshots = raw_manifest.get("snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != 20:
        raise RuntimeError("raw manifest must contain exactly 20 frozen snapshots")
    out: dict[tuple[str, int], dict[str, object]] = {}
    for row in snapshots:
        if not isinstance(row, dict):
            raise TypeError("raw manifest snapshot must be an object")
        key = (str(row["city"]), int(row["year"]))
        if key in out:
            raise RuntimeError(f"duplicate frozen snapshot key: {key}")
        out[key] = row
    return out


def _output_by_key(harmonized_manifest: dict[str, object]) -> dict[tuple[str, int], dict[str, object]]:
    outputs = harmonized_manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 20:
        raise RuntimeError("harmonized manifest must contain exactly 20 outputs")
    out: dict[tuple[str, int], dict[str, object]] = {}
    for row in outputs:
        if not isinstance(row, dict):
            raise TypeError("harmonized output entry must be an object")
        key = (str(row["city"]), int(row["year"]))
        if key in out:
            raise RuntimeError(f"duplicate harmonized output key: {key}")
        out[key] = row
    return out


def audit_one(
    *,
    path: Path,
    raw_entry: dict[str, object],
    output_entry: dict[str, object],
    batch_size: int = 100_000,
) -> dict[str, object]:
    blockers: list[str] = []
    warnings: list[str] = []

    if not path.is_file():
        return {
            "city": raw_entry["city"],
            "year": raw_entry["year"],
            "filename": path.name,
            "gate": "FAIL",
            "blockers": ["missing harmonized file"],
            "warnings": [],
        }

    actual_size = path.stat().st_size
    actual_sha = sha256_file(path)
    if actual_size != int(output_entry["harmonized_size_bytes"]):
        blockers.append("harmonized size mismatch vs execution manifest")
    if actual_sha != str(output_entry["harmonized_sha256"]):
        blockers.append("harmonized SHA256 mismatch vs execution manifest")

    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    if columns != HARMONIZED_COLUMNS:
        blockers.append("harmonized columns/order mismatch")

    rows = 0
    null_ids = 0
    duplicate_ids = 0
    seen: set[str] = set()
    invalid_created = 0
    other_year_created = 0
    created_min = None
    created_max = None
    invalid_closed_nonblank = 0
    closed_before_created = 0
    nonblank_closed = 0
    final_status_rows = 0
    eligible_resolved_rows = 0
    missing: dict[str, int] = {
        "status": 0,
        "category": 0,
        "descriptor": 0,
        "agency": 0,
        "latitude": 0,
        "longitude": 0,
        "area": 0,
        "closed_date": 0,
    }
    observed_cities: set[str] = set()
    year = int(raw_entry["year"])

    for batch in parquet.iter_batches(batch_size=batch_size, columns=HARMONIZED_COLUMNS):
        frame = batch.to_pandas()
        n = len(frame)
        rows += n

        ids = frame["request_id"].astype("string")
        blank_ids = _blank(ids)
        null_ids += int(blank_ids.sum())
        for value in ids[~blank_ids]:
            key = str(value)
            if key in seen:
                duplicate_ids += 1
            else:
                seen.add(key)

        created = pd.to_datetime(
            frame["created_date"], errors="coerce", utc=True, format="mixed"
        )
        invalid_created += int(created.isna().sum())
        valid_created = created.dropna()
        if not valid_created.empty:
            cmin, cmax = valid_created.min(), valid_created.max()
            created_min = cmin if created_min is None else min(created_min, cmin)
            created_max = cmax if created_max is None else max(created_max, cmax)
            other_year_created += int(valid_created.dt.year.ne(year).sum())

        closed_raw = frame["closed_date"].astype("string")
        blank_closed = _blank(closed_raw)
        missing["closed_date"] += int(blank_closed.sum())
        nonblank_closed += int((~blank_closed).sum())
        closed = pd.to_datetime(closed_raw, errors="coerce", utc=True, format="mixed")
        invalid_closed_nonblank += int(((~blank_closed) & closed.isna()).sum())

        paired = created.notna() & closed.notna()
        negative = paired & closed.lt(created)
        closed_before_created += int(negative.sum())

        status_norm = frame["status"].astype("string").fillna("").str.strip().str.lower()
        final_mask = status_norm.isin(FINAL_STATUSES)
        final_status_rows += int(final_mask.sum())
        eligible_resolved_rows += int((final_mask & paired & ~negative).sum())

        for column in ("status", "category", "descriptor", "agency", "latitude", "longitude", "area"):
            missing[column] += int(_blank(frame[column]).sum())

        observed_cities.update(
            frame["city"].astype("string").dropna().str.strip().replace("", pd.NA).dropna().unique()
        )

    if rows != int(raw_entry["rows"]):
        blockers.append("row preservation mismatch vs frozen raw")
    if rows != int(output_entry["rows"]):
        blockers.append("row count mismatch vs harmonized execution manifest")
    if null_ids != int(raw_entry["null_ids"]):
        blockers.append("request-ID null audit mismatch vs frozen raw")
    if duplicate_ids != int(raw_entry["duplicate_ids"]):
        blockers.append("request-ID duplicate audit mismatch vs frozen raw")
    if invalid_created:
        blockers.append(f"{invalid_created} unparseable created_date values")
    if other_year_created:
        blockers.append(f"{other_year_created} created_date values outside snapshot year")
    if observed_cities != {str(raw_entry["city"])}:
        blockers.append(f"unexpected city values: {sorted(observed_cities)}")

    expected_min = pd.to_datetime(str(raw_entry["min_created"]), utc=True)
    expected_max = pd.to_datetime(str(raw_entry["max_created"]), utc=True)
    if created_min != expected_min or created_max != expected_max:
        blockers.append("created_date bounds differ from frozen raw")

    if invalid_closed_nonblank:
        warnings.append(f"{invalid_closed_nonblank} nonblank closed_date values are unparseable")
    if closed_before_created:
        warnings.append(f"{closed_before_created} rows have closed_date before created_date")
    if missing["category"]:
        warnings.append(f"{missing['category']} rows have missing category")
    if missing["descriptor"]:
        warnings.append(f"{missing['descriptor']} rows have missing descriptor")

    return {
        "city": raw_entry["city"],
        "year": raw_entry["year"],
        "filename": path.name,
        "gate": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "warnings": warnings,
        "size_bytes": actual_size,
        "sha256": actual_sha,
        "rows": rows,
        "unique_request_ids": len(seen),
        "null_request_ids": null_ids,
        "duplicate_request_ids": duplicate_ids,
        "created_date_min": created_min.isoformat() if created_min is not None else None,
        "created_date_max": created_max.isoformat() if created_max is not None else None,
        "invalid_created_date": invalid_created,
        "created_date_outside_snapshot_year": other_year_created,
        "observed_cities": sorted(observed_cities),
        "nonblank_closed_date": nonblank_closed,
        "invalid_closed_date_nonblank": invalid_closed_nonblank,
        "closed_before_created": closed_before_created,
        "final_status_rows": final_status_rows,
        "eligible_resolved_rows": eligible_resolved_rows,
        "eligible_resolved_fraction": _pct(eligible_resolved_rows, rows),
        "missing_counts": missing,
        "missing_fractions": {k: _pct(v, rows) for k, v in missing.items()},
    }


def audit_harmonized(
    *,
    harmonized_dir: Path,
    raw_manifest_path: Path,
    harmonized_manifest_path: Path,
    report_path: Path,
    batch_size: int = 100_000,
) -> dict[str, object]:
    raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
    harmonized_manifest = json.loads(harmonized_manifest_path.read_text(encoding="utf-8"))
    expected = _expected_by_key(raw_manifest)
    outputs = _output_by_key(harmonized_manifest)

    if set(expected) != set(outputs):
        missing = sorted(set(expected) - set(outputs))
        extra = sorted(set(outputs) - set(expected))
        raise RuntimeError(f"harmonized manifest key mismatch; missing={missing}, extra={extra}")

    results: list[dict[str, object]] = []
    for key in sorted(expected):
        output_entry = outputs[key]
        filename = str(output_entry["harmonized_filename"])
        results.append(
            audit_one(
                path=harmonized_dir / filename,
                raw_entry=expected[key],
                output_entry=output_entry,
                batch_size=batch_size,
            )
        )

    blockers = [
        f"{row['city']} {row['year']}: {message}"
        for row in results
        for message in row["blockers"]
    ]
    warnings = [
        f"{row['city']} {row['year']}: {message}"
        for row in results
        for message in row["warnings"]
    ]

    report = {
        "protocol": "t2_harmonized_audit_v1",
        "gate": "PASS" if not blockers else "FAIL",
        "blockers": blockers,
        "warnings": warnings,
        "rules": {
            "structural_and_provenance_failures_are_blocking": True,
            "closed_date_and_predictor_missingness_are_quantified_not_silently_filtered": True,
            "no_threshold_was_selected_after_inspecting_results": True,
            "training_authorized_by_this_report": False,
        },
        "files": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
