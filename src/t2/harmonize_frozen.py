from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.t2.provenance import git_state, sha256_file

HARMONIZED_COLUMNS = [
    "request_id", "created_date", "closed_date", "status", "category",
    "descriptor", "agency", "latitude", "longitude", "area", "city",
]

def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())

def _column_map(columns: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for column in columns:
        key = _norm(column)
        if key in out and out[key] != column:
            raise ValueError(f"normalized column collision: {out[key]!r} vs {column!r}")
        out[key] = column
    return out

def _pick(frame: pd.DataFrame, cmap: dict[str, str], aliases: list[str]) -> pd.Series:
    for alias in aliases:
        source = cmap.get(_norm(alias))
        if source is not None:
            return frame[source].astype("string")
    return pd.Series(pd.NA, index=frame.index, dtype="string")

def _coalesce(frame: pd.DataFrame, cmap: dict[str, str], aliases: list[str]) -> pd.Series:
    out = pd.Series(pd.NA, index=frame.index, dtype="string")
    for alias in aliases:
        source = cmap.get(_norm(alias))
        if source is None:
            continue
        values = frame[source].astype("string").str.strip().replace("", pd.NA)
        out = out.fillna(values)
    return out

def harmonize_chunk(frame: pd.DataFrame, city: str) -> pd.DataFrame:
    cmap = _column_map(list(frame.columns))
    out = pd.DataFrame(index=frame.index)

    if city == "nyc":
        spec = {
            "request_id": ["unique_key"],
            "created_date": ["created_date"],
            "closed_date": ["closed_date"],
            "status": ["status"],
            "category": ["complaint_type"],
            "descriptor": ["descriptor"],
            "agency": ["agency"],
            "latitude": ["latitude"],
            "longitude": ["longitude"],
        }
        area = ["borough"]
    elif city == "chicago":
        spec = {
            "request_id": ["sr_number"],
            "created_date": ["created_date"],
            "closed_date": ["closed_date"],
            "status": ["status"],
            "category": ["sr_type"],
            "descriptor": ["sr_short_code"],
            "agency": ["owner_department"],
            "latitude": ["latitude"],
            "longitude": ["longitude"],
        }
        area = ["community_area", "ward"]
    elif city == "boston":
        if _norm("_id") in cmap:
            raise ValueError("Boston frozen raw CSV must not contain CKAN datastore-only _id")
        if len(frame.columns) != 30:
            raise ValueError(f"Boston frozen raw CSV must have 30 columns, found {len(frame.columns)}")
        spec = {
            "request_id": ["case_enquiry_id"],
            "created_date": ["open_dt"],
            "closed_date": ["closed_dt"],
            "status": ["case_status"],
            "category": ["reason"],
            "descriptor": ["type"],
            "latitude": ["latitude"],
            "longitude": ["longitude"],
        }
        out["agency"] = _coalesce(frame, cmap, ["subject", "department"])
        area = ["neighborhood", "ward"]
    elif city == "los_angeles":
        spec = {
            "request_id": ["CaseNumber", "SRNumber", "casenumber", "srnumber"],
            "created_date": ["CreatedDate", "createddate"],
            "closed_date": ["ClosedDate", "closeddate"],
            "status": ["Status", "status"],
            "category": ["RequestType", "requesttype", "type"],
            "descriptor": ["ActionTaken", "actiontaken", "action_taken__c"],
            "agency": ["Owner", "owner", "department_name__c"],
            "latitude": ["Latitude", "latitude"],
            "longitude": ["Longitude", "longitude"],
        }
        area = [
            "NCName", "ncname", "NC", "nc", "CD", "cd",
            "neighborhood_council_name__c", "neighborhood_council__c",
        ]
    else:
        raise ValueError(f"unsupported city: {city}")

    missing_targets = [
        target
        for target, aliases in spec.items()
        if not any(_norm(alias) in cmap for alias in aliases)
    ]
    if missing_targets:
        raise ValueError(
            f"{city}: frozen raw schema missing mapped fields for {missing_targets}"
        )
    if not any(_norm(alias) in cmap for alias in area):
        raise ValueError(f"{city}: frozen raw schema missing every configured area field")
    if city == "boston" and not any(
        _norm(alias) in cmap for alias in ["subject", "department"]
    ):
        raise ValueError("boston: frozen raw schema missing subject/department agency fields")

    for target, aliases in spec.items():
        out[target] = _pick(frame, cmap, aliases)
    out["area"] = _coalesce(frame, cmap, area)
    out["city"] = city
    for column in HARMONIZED_COLUMNS:
        if column not in out:
            out[column] = pd.Series(pd.NA, index=frame.index, dtype="string")
    return out[HARMONIZED_COLUMNS].astype("string")

def _hash_bytes(path: Path) -> tuple[int, str]:
    return path.stat().st_size, sha256_file(path)

def _validate_header(columns: list[str], entry: dict[str, object]) -> None:
    if len(columns) != int(entry["column_count"]):
        raise RuntimeError(
            f"{entry['filename']}: column count mismatch "
            f"{len(columns)} != {entry['column_count']}"
        )
    expected = entry.get("expected_columns")
    if not isinstance(expected, list) or not expected:
        raise RuntimeError(f"{entry['filename']}: frozen expected_columns missing")
    if columns != expected:
        raise RuntimeError(
            f"{entry['filename']}: exact frozen header mismatch; "
            f"observed={columns!r} expected={expected!r}"
        )

def harmonize_snapshot(
    *,
    raw_path: Path,
    output_path: Path,
    entry: dict[str, object],
    chunksize: int = 100_000,
) -> dict[str, object]:
    expected_size = int(entry["size_bytes"])
    expected_sha = str(entry["sha256"])
    actual_size, actual_sha = _hash_bytes(raw_path)
    if actual_size != expected_size:
        raise RuntimeError(f"{raw_path.name}: size mismatch {actual_size} != {expected_size}")
    if actual_sha != expected_sha:
        raise RuntimeError(f"{raw_path.name}: SHA256 mismatch")

    header = pd.read_csv(raw_path, nrows=0)
    _validate_header(list(header.columns), entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing harmonized file: {output_path}")

    writer: pq.ParquetWriter | None = None
    rows = 0
    null_ids = 0
    duplicate_ids = 0
    seen: set[str] = set()
    invalid_created = 0
    created_min = None
    created_max = None

    try:
        for chunk in pd.read_csv(raw_path, dtype="string", chunksize=chunksize, low_memory=False):
            harmonized = harmonize_chunk(chunk, str(entry["city"]))
            ids = harmonized["request_id"].astype("string")
            blank = ids.isna() | ids.fillna("").str.strip().eq("")
            null_ids += int(blank.sum())
            for value in ids[~blank]:
                key = str(value)
                if key in seen:
                    duplicate_ids += 1
                else:
                    seen.add(key)

            created = pd.to_datetime(harmonized["created_date"], errors="coerce", utc=True)
            invalid_created += int(created.isna().sum())
            valid = created.dropna()
            if not valid.empty:
                cmin, cmax = valid.min(), valid.max()
                created_min = cmin if created_min is None else min(created_min, cmin)
                created_max = cmax if created_max is None else max(created_max, cmax)

            table = pa.Table.from_pandas(harmonized, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(harmonized)
    finally:
        if writer is not None:
            writer.close()

    if rows != int(entry["rows"]):
        raise RuntimeError(f"{raw_path.name}: row count mismatch {rows} != {entry['rows']}")
    if null_ids != int(entry["null_ids"]) or duplicate_ids != int(entry["duplicate_ids"]):
        raise RuntimeError(f"{raw_path.name}: request-ID audit mismatch")
    if invalid_created != 0:
        raise RuntimeError(f"{raw_path.name}: {invalid_created} unparseable created dates")

    expected_min = pd.to_datetime(str(entry["min_created"]), utc=True)
    expected_max = pd.to_datetime(str(entry["max_created"]), utc=True)
    if created_min != expected_min or created_max != expected_max:
        raise RuntimeError(
            f"{raw_path.name}: date bounds mismatch "
            f"{created_min}/{created_max} != {expected_min}/{expected_max}"
        )

    tmp_path.replace(output_path)
    out_size, out_sha = _hash_bytes(output_path)
    return {
        "city": entry["city"],
        "year": entry["year"],
        "raw_filename": raw_path.name,
        "raw_size_bytes": actual_size,
        "raw_sha256": actual_sha,
        "harmonized_filename": output_path.name,
        "harmonized_size_bytes": out_size,
        "harmonized_sha256": out_sha,
        "rows": rows,
        "columns": HARMONIZED_COLUMNS,
        "null_request_ids": null_ids,
        "duplicate_request_ids": duplicate_ids,
        "created_date_min": created_min.isoformat() if created_min is not None else None,
        "created_date_max": created_max.isoformat() if created_max is not None else None,
        "coverage_note": entry.get("coverage_note"),
    }

def harmonize_all(
    *, raw_dir: Path, output_dir: Path, manifest_path: Path, chunksize: int = 100_000
) -> dict[str, object]:
    manifest_bytes = manifest_path.read_bytes()
    raw_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    raw_manifest = json.loads(manifest_bytes)
    entries = raw_manifest["snapshots"]
    schemas = raw_manifest.get("schemas")
    if len(entries) != 20:
        raise RuntimeError(f"expected 20 frozen raw snapshots, found {len(entries)}")
    if not isinstance(schemas, dict):
        raise RuntimeError("frozen raw manifest is missing exact schema definitions")

    sha, dirty = git_state()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    for raw_entry in entries:
        entry = dict(raw_entry)
        city = str(entry["city"])
        year = int(entry["year"])
        schema_key = str(entry.get("schema_key", ""))
        expected_columns = schemas.get(schema_key)
        if not isinstance(expected_columns, list) or not expected_columns:
            raise RuntimeError(
                f"{entry['filename']}: unknown or empty schema_key {schema_key!r}"
            )
        entry["expected_columns"] = expected_columns
        raw_path = raw_dir / str(entry["filename"])
        if not raw_path.is_file():
            raise FileNotFoundError(raw_path)
        output_path = output_dir / f"{city.upper()}_{year}_harmonized.parquet"
        results.append(
            harmonize_snapshot(
                raw_path=raw_path,
                output_path=output_path,
                entry=entry,
                chunksize=chunksize,
            )
        )

    harmonized_manifest = {
        "protocol": "t2_frozen_raw_harmonization_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha,
        "git_dirty": dirty,
        "raw_manifest_path": str(manifest_path),
        "raw_manifest_sha256": raw_manifest_sha,
        "source_rule": "frozen Drive raw snapshots only; no live API fetch",
        "row_filtering": "none",
        "outputs": results,
    }
    manifest_out = output_dir / "T2_HARMONIZED_MANIFEST.json"
    manifest_out.write_text(json.dumps(harmonized_manifest, indent=2, sort_keys=True) + "\n")
    checksum_out = output_dir / "T2_HARMONIZED_CHECKSUMS_SHA256.txt"
    checksum_out.write_text(
        "".join(f"{row['harmonized_sha256']}  {row['harmonized_filename']}\n" for row in results)
    )
    return harmonized_manifest
