"""Build harmonized per-city parquet files from frozen raw snapshots only.

No network access. Input: <archive>/01_Raw_Official/CITY_YYYY.jsonl.gz (+ .sha256).
Output: <archive>/02_Harmonized/<city>_2021_2025_harmonized.parquet (+ .sha256),
<archive>/03_Manifests_and_Checksums/HARMONIZATION_MANIFEST.json,
ROW_ACCOUNTING.csv and HARMONIZED_SHA256SUMS.txt.

Harmonization is lossless with respect to rows: every raw row yields exactly one
harmonized row (no dedup, no status/outcome filtering, no geo filtering). Column
mapping only; date strings are parsed to naive timestamps in source-local time;
latitude/longitude are cast to float64. Everything else is kept as string.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import CITIES, SOURCE_BY_KEY, YEARS, SocrataSource  # noqa: E402

CANONICAL_COLUMNS = [
    "request_id",
    "created_date",
    "closed_date",
    "status",
    "category",
    "descriptor",
    "agency",
    "latitude",
    "longitude",
    "area",
    "city",
    "source_year",
]

SCHEMA = pa.schema(
    [
        ("request_id", pa.string()),
        ("created_date", pa.timestamp("us")),
        ("closed_date", pa.timestamp("us")),
        ("status", pa.string()),
        ("category", pa.string()),
        ("descriptor", pa.string()),
        ("agency", pa.string()),
        ("latitude", pa.float64()),
        ("longitude", pa.float64()),
        ("area", pa.string()),
        ("city", pa.string()),
        ("source_year", pa.int16()),
    ]
)

# canonical -> ordered list of source fields (first non-empty wins)
MAPPINGS: dict[str, dict[str, list[str]]] = {
    "nyc": {
        "request_id": ["unique_key"],
        "created_date": ["created_date"],
        "closed_date": ["closed_date"],
        "status": ["status"],
        "category": ["complaint_type"],
        "descriptor": ["descriptor"],
        "agency": ["agency"],
        "latitude": ["latitude"],
        "longitude": ["longitude"],
        "area": ["community_board", "borough"],
    },
    "chicago": {
        "request_id": ["sr_number"],
        "created_date": ["created_date"],
        "closed_date": ["closed_date"],
        "status": ["status"],
        "category": ["sr_type"],
        "descriptor": ["sr_short_code"],
        "agency": ["owner_department"],
        "latitude": ["latitude"],
        "longitude": ["longitude"],
        "area": ["community_area", "ward"],
    },
    "boston": {
        "request_id": ["case_enquiry_id"],
        "created_date": ["open_dt"],
        "closed_date": ["closed_dt"],
        "status": ["case_status"],
        "category": ["reason"],
        "descriptor": ["type"],
        "agency": ["subject", "department"],
        "latitude": ["latitude"],
        "longitude": ["longitude"],
        "area": ["neighborhood", "ward"],
    },
    "los_angeles": {
        "request_id": ["srnumber"],
        "created_date": ["createddate"],
        "closed_date": ["closeddate"],
        "status": ["status"],
        "category": ["requesttype"],
        "descriptor": ["actiontaken"],
        "agency": ["owner"],
        "latitude": ["latitude"],
        "longitude": ["longitude"],
        "area": ["ncname", "nc", "cd"],
    },
    "los_angeles_2025": {
        "request_id": ["casenumber"],
        "created_date": ["createddate"],
        "closed_date": ["closeddate"],
        "status": ["status"],
        "category": ["type"],
        "descriptor": ["action_taken__c"],
        "agency": ["department_name__c"],
        "latitude": ["geolocation__latitude__s"],
        "longitude": ["geolocation__longitude__s"],
        "area": [
            "locator_sr_neigborhood_council_1",
            "locator_sr_neigborhood_council",
            "locator_council_district",
        ],
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_for(city: str, year: int) -> dict[str, list[str]]:
    if city == "los_angeles" and year == 2025:
        return MAPPINGS["los_angeles_2025"]
    return MAPPINGS[city]


def first_nonempty(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in columns:
        if column not in frame:
            continue
        values = frame[column].astype("string").str.strip().replace("", pd.NA)
        out = out.fillna(values)
    return out


CHUNK_ROWS = 500_000


def iter_raw_chunks(path: Path):
    records: list[dict[str, object]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
            if len(records) >= CHUNK_ROWS:
                yield pd.DataFrame.from_records(records).astype("string")
                records = []
    if records:
        yield pd.DataFrame.from_records(records).astype("string")


COUNTER_KEYS = (
    "raw_rows_in", "harmonized_rows_out", "rows_dropped", "request_id_missing",
    "request_id_duplicate_rows", "created_date_unparsed", "created_date_missing",
    "created_date_outside_source_year", "closed_date_present", "closed_date_unparsed",
    "closed_date_missing", "status_missing", "category_missing", "descriptor_missing",
    "agency_missing", "latitude_missing", "longitude_missing", "area_missing",
)


def harmonize_chunk(
    city: str, year: int, raw: pd.DataFrame, seen_ids: set[str], acc: dict[str, object]
) -> pa.Table:
    mapping = mapping_for(city, year)
    out = pd.DataFrame(index=raw.index)
    for canonical, sources in mapping.items():
        out[canonical] = first_nonempty(raw, sources)
    created = pd.to_datetime(out["created_date"], errors="coerce", format="ISO8601")
    closed = pd.to_datetime(out["closed_date"], errors="coerce", format="ISO8601")
    lat = pd.to_numeric(out["latitude"], errors="coerce")
    lon = pd.to_numeric(out["longitude"], errors="coerce")

    dup_rows = 0
    for rid in out["request_id"].dropna():
        if rid in seen_ids:
            dup_rows += 1
        else:
            seen_ids.add(rid)
    counts = {
        "raw_rows_in": len(raw),
        "harmonized_rows_out": len(out),
        "rows_dropped": 0,
        "request_id_missing": out["request_id"].isna().sum(),
        "request_id_duplicate_rows": dup_rows,
        "created_date_unparsed": (created.isna() & out["created_date"].notna()).sum(),
        "created_date_missing": out["created_date"].isna().sum(),
        "created_date_outside_source_year": (created.dt.year.ne(year) & created.notna()).sum(),
        "closed_date_present": closed.notna().sum(),
        "closed_date_unparsed": (closed.isna() & out["closed_date"].notna()).sum(),
        "closed_date_missing": out["closed_date"].isna().sum(),
        "status_missing": out["status"].isna().sum(),
        "category_missing": out["category"].isna().sum(),
        "descriptor_missing": out["descriptor"].isna().sum(),
        "agency_missing": out["agency"].isna().sum(),
        "latitude_missing": lat.isna().sum(),
        "longitude_missing": lon.isna().sum(),
        "area_missing": out["area"].isna().sum(),
    }
    for key in COUNTER_KEYS:
        acc[key] = int(acc.get(key, 0)) + int(counts[key])
    status_counts: dict[str, int] = acc.setdefault("status_value_counts", {})  # type: ignore[assignment]
    for k, v in out["status"].value_counts(dropna=False).items():
        status_counts[str(k)] = status_counts.get(str(k), 0) + int(v)
    acc["field_mapping"] = mapping

    out["created_date"] = created
    out["closed_date"] = closed
    out["latitude"] = lat
    out["longitude"] = lon
    out["city"] = city
    out["source_year"] = year
    return pa.Table.from_pandas(out[CANONICAL_COLUMNS], schema=SCHEMA, preserve_index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--city", choices=CITIES, action="append")
    args = parser.parse_args()
    raw_dir = args.archive_dir / "01_Raw_Official"
    out_dir = args.archive_dir / "02_Harmonized"
    man_dir = args.archive_dir / "03_Manifests_and_Checksums"
    out_dir.mkdir(parents=True, exist_ok=True)
    man_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = man_dir / "HARMONIZATION_MANIFEST.json"
    manifest: dict[str, object] = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {"cities": {}}
    )
    cities_meta: dict[str, object] = manifest["cities"]  # type: ignore[assignment]

    for city in args.city or CITIES:
        target = out_dir / f"{city}_2021_2025_harmonized.parquet"
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing harmonized file: {target}")
        inputs = []
        accounting_rows = []
        seen_ids: set[str] = set()
        tmp = out_dir / f".{target.name}.partial"
        rows_written = 0
        with pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as writer:
            for year in YEARS:
                src = SOURCE_BY_KEY[(city, year)]
                path = raw_dir / f"{city.upper()}_{year}.jsonl.gz"
                recorded = (raw_dir / f"{path.name}.sha256").read_text().split()[0]
                actual = sha256_file(path)
                if actual != recorded:
                    raise RuntimeError(f"raw snapshot hash mismatch for {path.name}")
                print(f"{city} {year}: streaming {path.name} (sha256 verified)", flush=True)
                accounting: dict[str, object] = {}
                for raw in iter_raw_chunks(path):
                    table = harmonize_chunk(city, year, raw, seen_ids, accounting)
                    writer.write_table(table)
                    rows_written += table.num_rows
                    del raw, table
                inputs.append({
                    "snapshot_file": path.name, "sha256": actual,
                    "dataset_id": src.dataset_id if isinstance(src, SocrataSource) else src.resource_id,
                    "coverage_note": src.coverage_note,
                })
                accounting_rows.append({"city": city, "year": year, **accounting})
                print(f"  rows in={accounting['raw_rows_in']} out={accounting['harmonized_rows_out']}",
                      flush=True)
        tmp.chmod(0o444)
        tmp.replace(target)
        digest = sha256_file(target)
        (out_dir / f"{target.name}.sha256").write_text(f"{digest}  {target.name}\n")
        total_in = sum(int(r["raw_rows_in"]) for r in accounting_rows)
        cities_meta[city] = {
            "harmonized_file": target.name,
            "sha256": digest,
            "file_size_bytes": target.stat().st_size,
            "rows": int(rows_written),
            "raw_rows_in_total": total_in,
            "row_accounting_balanced": int(rows_written) == total_in
            and pq.read_metadata(target).num_rows == rows_written,
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "canonical_columns": CANONICAL_COLUMNS,
            "schema": {f.name: str(f.type) for f in SCHEMA},
            "inputs": inputs,
            "per_year": accounting_rows,
            "network_used": False,
        }
        print(f"{city}: wrote {target.name} rows={rows_written} sha256={digest}", flush=True)

    manifest["generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["timestamp_semantics"] = (
        "created_date/closed_date are naive timestamps in the source's local time as "
        "published; no timezone conversion applied."
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    rows = []
    for meta in cities_meta.values():
        for r in meta["per_year"]:  # type: ignore[index]
            rows.append({k: v for k, v in r.items() if not isinstance(v, (dict, list))})
    pd.DataFrame(rows).to_csv(man_dir / "ROW_ACCOUNTING.csv", index=False)
    sums = [f"{m['sha256']}  02_Harmonized/{m['harmonized_file']}" for m in cities_meta.values()]
    (man_dir / "HARMONIZED_SHA256SUMS.txt").write_text("\n".join(sums) + "\n")


if __name__ == "__main__":
    main()
