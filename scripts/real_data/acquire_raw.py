"""Download immutable raw 311 snapshots (one CITY_YYYY.jsonl.gz per city/year).

Rules enforced here:
  * real records only, retrieved directly from the official municipal portal;
  * pagination continues until the source is exhausted (no head-of-list sampling);
  * no status / closed-date / outcome value participates in any query predicate;
  * source values are written verbatim (one JSON object per source row);
  * an existing snapshot is never overwritten (use --force-new-run to write a new
    timestamped file next to it);
  * SHA256 + provenance sidecars are written for every snapshot.

Usage:
  python scripts/real_data/acquire_raw.py --archive-dir <root> --city nyc [--year 2021]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sources import (  # noqa: E402
    CKAN_PAGE_SIZE,
    SOCRATA_PAGE_SIZE,
    SOURCE_BY_KEY,
    CITIES,
    YEARS,
    CkanSource,
    SocrataSource,
)

USER_AGENT = "GovTrust-FL real-data rebuild (research; contact via repository)"
TIMEOUT = (30, 600)
MAX_ATTEMPTS = 8


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_json(session: requests.Session, url: str, params: dict[str, object]) -> object:
    last: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(url, params=params, timeout=TIMEOUT)
            if response.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {response.status_code}", response=response)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
            wait = min(2**attempt, 120)
            print(f"  retry {attempt}/{MAX_ATTEMPTS} after error: {exc} (sleep {wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"request failed permanently: {url} {params}") from last


class SnapshotWriter:
    """Streams source rows to gzip JSONL and accumulates outcome-free statistics."""

    def __init__(self, tmp_path: Path, id_field: str, created_field: str) -> None:
        self.tmp_path = tmp_path
        self.id_field = id_field
        self.created_field = created_field
        self.handle = gzip.open(tmp_path, "wt", encoding="utf-8", compresslevel=6)
        self.rows = 0
        self.ids: set[str] = set()
        self.duplicate_ids: set[str] = set()
        self.missing_id_rows = 0
        self.missing_created_rows = 0
        self.first_created: str | None = None
        self.last_created: str | None = None

    def write_rows(self, rows: list[dict[str, object]]) -> None:
        for row in rows:
            self.handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            self.handle.write("\n")
            self.rows += 1
            rid = row.get(self.id_field)
            if rid is None or str(rid).strip() == "":
                self.missing_id_rows += 1
            else:
                rid = str(rid).strip()
                if rid in self.ids:
                    self.duplicate_ids.add(rid)
                else:
                    self.ids.add(rid)
            created = row.get(self.created_field)
            if created is None or str(created).strip() == "":
                self.missing_created_rows += 1
            else:
                created = str(created)
                if self.first_created is None or created < self.first_created:
                    self.first_created = created
                if self.last_created is None or created > self.last_created:
                    self.last_created = created

    def close(self) -> None:
        self.handle.close()

    def stats(self) -> dict[str, object]:
        return {
            "raw_rows": self.rows,
            "unique_request_ids": len(self.ids),
            "duplicate_request_ids": len(self.duplicate_ids),
            "duplicate_request_id_rows": self.rows - self.missing_id_rows - len(self.ids),
            "rows_missing_request_id": self.missing_id_rows,
            "rows_missing_created": self.missing_created_rows,
            "first_created": self.first_created,
            "last_created": self.last_created,
        }


def fetch_socrata(
    session: requests.Session, src: SocrataSource, writer: SnapshotWriter
) -> dict[str, object]:
    start = f"{src.year}-01-01T00:00:00.000"
    end = f"{src.year}-12-31T23:59:59.999"
    base_where = f"{src.created_field} between '{start}' and '{end}'"
    select = ",".join(src.fields)
    pages: list[dict[str, object]] = []
    last_row_id: str | None = None
    page_no = 0
    while True:
        where = base_where if last_row_id is None else f"{base_where} AND :id > '{last_row_id}'"
        params = {
            "$select": select,
            "$where": where,
            "$order": ":id",
            "$limit": SOCRATA_PAGE_SIZE,
        }
        t0 = time.time()
        payload = get_json(session, src.resource_url, params)
        if not isinstance(payload, list):
            raise TypeError(f"unexpected Socrata payload: {type(payload)}")
        page_no += 1
        writer.write_rows(payload)
        pages.append(
            {
                "page": page_no,
                "rows": len(payload),
                "after_row_id": last_row_id,
                "seconds": round(time.time() - t0, 2),
            }
        )
        print(f"  {src.city}_{src.year} page {page_no}: {len(payload)} rows "
              f"(total {writer.rows}) {pages[-1]['seconds']}s", flush=True)
        if len(payload) < SOCRATA_PAGE_SIZE:
            break
        last_row_id = str(payload[-1][":id"])
    return {
        "source_type": "socrata",
        "official_source_url": src.resource_url,
        "dataset_landing_url": src.landing_url,
        "dataset_id": src.dataset_id,
        "dataset_name": src.dataset_name,
        "query_parameters": {
            "$select": select,
            "$where": base_where,
            "$order": ":id",
            "$limit": SOCRATA_PAGE_SIZE,
            "pagination": "keyset on Socrata row identifier :id (AND :id > <last :id>)",
        },
        "api_pages": page_no,
        "page_log": pages,
    }


def fetch_ckan(
    session: requests.Session, src: CkanSource, writer: SnapshotWriter
) -> dict[str, object]:
    pages: list[dict[str, object]] = []
    offset = 0
    page_no = 0
    reported_total: int | None = None
    while True:
        params = {
            "resource_id": src.resource_id,
            "limit": CKAN_PAGE_SIZE,
            "offset": offset,
            "sort": "_id asc",
            "fields": ",".join(src.fields),
        }
        t0 = time.time()
        payload = get_json(session, src.resource_url, params)
        if not isinstance(payload, dict) or not payload.get("success"):
            raise RuntimeError(f"CKAN failure at offset {offset}: {payload}")
        result = payload["result"]
        records = result["records"]
        if reported_total is None:
            reported_total = int(result.get("total", -1))
        page_no += 1
        writer.write_rows(records)
        pages.append(
            {"page": page_no, "offset": offset, "rows": len(records),
             "seconds": round(time.time() - t0, 2)}
        )
        print(f"  {src.city}_{src.year} page {page_no}: {len(records)} rows "
              f"(total {writer.rows}/{reported_total}) {pages[-1]['seconds']}s", flush=True)
        if len(records) < CKAN_PAGE_SIZE:
            break
        offset += CKAN_PAGE_SIZE
    return {
        "source_type": "ckan",
        "official_source_url": src.resource_url,
        "dataset_landing_url": src.landing_url,
        "dataset_id": src.resource_id,
        "dataset_name": src.dataset_name,
        "query_parameters": {
            "resource_id": src.resource_id,
            "limit": CKAN_PAGE_SIZE,
            "sort": "_id asc",
            "fields": ",".join(src.fields),
            "pagination": "offset (full resource, no row filter)",
        },
        "source_reported_total": reported_total,
        "api_pages": page_no,
        "page_log": pages,
    }


def acquire(city: str, year: int, raw_dir: Path, force_new_run: bool) -> Path:
    src = SOURCE_BY_KEY[(city, year)]
    stem = f"{city.upper()}_{year}"
    final_path = raw_dir / f"{stem}.jsonl.gz"
    if final_path.exists():
        if not force_new_run:
            print(f"SKIP {stem}: snapshot already exists and will not be overwritten: {final_path}")
            return final_path
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        final_path = raw_dir / f"{stem}.rerun-{stamp}.jsonl.gz"
    tmp_path = raw_dir / f".{final_path.name}.partial"
    if tmp_path.exists():
        tmp_path.unlink()

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    started = utc_now()
    print(f"START {stem} at {started}", flush=True)
    writer = SnapshotWriter(tmp_path, src.id_field, src.created_field)
    try:
        if isinstance(src, SocrataSource):
            fetch_meta = fetch_socrata(session, src, writer)
        elif isinstance(src, CkanSource):
            fetch_meta = fetch_ckan(session, src, writer)
        else:
            raise TypeError(src)
    finally:
        writer.close()
    finished = utc_now()

    os.chmod(tmp_path, 0o444)
    os.replace(tmp_path, final_path)
    digest = sha256_file(final_path)
    size = final_path.stat().st_size
    (raw_dir / f"{final_path.name}.sha256").write_text(f"{digest}  {final_path.name}\n")

    provenance = {
        "city": city,
        "year": year,
        "snapshot_file": final_path.name,
        "retrieval_started_utc": started,
        "retrieval_finished_utc": finished,
        "retrieval_utc_timestamp": finished,
        "id_field": src.id_field,
        "created_field": src.created_field,
        "retrieved_fields": list(src.fields),
        "coverage_note": src.coverage_note,
        "outcome_fields_used_in_query_predicate": False,
        "sha256": digest,
        "file_size_bytes": size,
        **fetch_meta,
        **writer.stats(),
    }
    (raw_dir / f"{final_path.name}.provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n"
    )
    print(f"DONE {stem}: rows={writer.rows} pages={fetch_meta['api_pages']} "
          f"sha256={digest[:16]}... size={size}", flush=True)
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--city", choices=CITIES, action="append")
    parser.add_argument("--year", type=int, choices=YEARS, action="append")
    parser.add_argument("--force-new-run", action="store_true",
                        help="write an additional timestamped snapshot instead of skipping")
    args = parser.parse_args()
    raw_dir = args.archive_dir / "01_Raw_Official"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for city in args.city or CITIES:
        for year in args.year or YEARS:
            acquire(city, year, raw_dir, args.force_new_run)


if __name__ == "__main__":
    main()
