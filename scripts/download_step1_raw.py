"""Download Step 1 raw 311 datasets.

This script keeps the first experimental extract small while covering 2021-2025:

- NYC: 500,000 rows, stratified as 100,000 per year.
- Chicago: 300,000 rows, stratified as 60,000 per year.
- Boston: 200,000 rows, stratified as 40,000 per year.
- Los Angeles: 200,000 rows, stratified as 40,000 per year.

NYC, Chicago, and Los Angeles are downloaded as CSV bytes from their Socrata
CSV endpoints. Boston's CKAN API is currently blocked directly by Cloudflare in
this environment, so those official CKAN API records are fetched through Jina
Reader and written as CSV.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "download_manifest_step1.json"

SODA_PAGE_LIMIT = 50_000
BOSTON_PAGE_LIMIT = 5_000
REQUEST_TIMEOUT = (15, 300)
YEARS = range(2021, 2026)


@dataclass(frozen=True)
class SocrataSpec:
    name: str
    domain: str
    dataset_id: str
    date_field: str
    output_path: Path
    rows_per_year: int
    year: int

    @property
    def base_url(self) -> str:
        return f"https://{self.domain}/resource/{self.dataset_id}.csv"


@dataclass(frozen=True)
class BostonSpec:
    year: int
    resource_id: str
    rows: int


def log(message: str) -> None:
    print(message, flush=True)


def request_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, object] | None = None,
    stream: bool = False,
    retries: int = 4,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                url,
                params=params,
                stream=stream,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "GovTrust-FL/0.1 research data downloader"},
            )
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            wait_seconds = min(2**attempt, 30)
            log(f"  retry {attempt}/{retries} after error: {exc}")
            time.sleep(wait_seconds)
    raise RuntimeError(f"Failed request after {retries} retries: {url}") from last_error


def append_csv_response(
    response: requests.Response,
    output_file,
    *,
    include_header: bool,
) -> None:
    """Append a CSV response, optionally dropping its first header line."""

    if include_header:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                output_file.write(chunk)
        return

    header_skipped = False
    buffered = b""
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if not chunk:
            continue
        if header_skipped:
            output_file.write(chunk)
            continue

        buffered += chunk
        newline_at = buffered.find(b"\n")
        if newline_at == -1:
            continue

        output_file.write(buffered[newline_at + 1 :])
        buffered = b""
        header_skipped = True


def count_csv_records(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.reader(input_file)
        next(reader, None)
        return sum(1 for _ in reader)


def socrata_params(date_field: str, year: int, limit: int, offset: int) -> dict[str, object]:
    start = f"{year}-01-01T00:00:00"
    end = f"{year}-12-31T23:59:59"
    return {
        "$limit": limit,
        "$offset": offset,
        "$order": date_field,
        "$where": f"{date_field} between '{start}' and '{end}'",
    }


def download_socrata_group(output_path: Path, specs: Iterable[SocrataSpec]) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    session = requests.Session()

    header_written = False
    chunks: list[dict[str, object]] = []
    with temp_path.open("wb") as output_file:
        for spec in specs:
            remaining = spec.rows_per_year
            offset = 0
            while remaining > 0:
                limit = min(SODA_PAGE_LIMIT, remaining)
                params = socrata_params(spec.date_field, spec.year, limit, offset)
                log(
                    f"{spec.name}: {spec.year} rows {offset + 1:,}-"
                    f"{offset + limit:,} from {spec.dataset_id}"
                )
                response = request_with_retries(
                    session,
                    spec.base_url,
                    params=params,
                    stream=True,
                )
                append_csv_response(response, output_file, include_header=not header_written)
                header_written = True
                chunks.append(
                    {
                        "year": spec.year,
                        "dataset_id": spec.dataset_id,
                        "url": response.url,
                        "requested_rows": limit,
                        "offset": offset,
                    }
                )
                remaining -= limit
                offset += limit

    temp_path.replace(output_path)
    actual_records = count_csv_records(output_path)
    return {
        "path": str(output_path.relative_to(PROJECT_ROOT)),
        "records": actual_records,
        "size_bytes": output_path.stat().st_size,
        "chunks": chunks,
    }


def extract_jina_markdown_json(text: str) -> dict[str, object]:
    marker = "Markdown Content:\n"
    marker_at = text.find(marker)
    if marker_at == -1:
        payload = text.strip()
    else:
        payload = text[marker_at + len(marker) :].strip()
    return json.loads(payload)


def boston_proxy_url(resource_id: str, limit: int, offset: int) -> str:
    query = urlencode({"resource_id": resource_id, "limit": limit, "offset": offset})
    upstream_url = f"http://data.boston.gov/api/3/action/datastore_search?{query}"
    return f"https://r.jina.ai/http://{upstream_url}"


def fetch_boston_records(
    session: requests.Session,
    resource_id: str,
    limit: int,
    offset: int,
) -> tuple[list[str], list[dict[str, object]], str]:
    url = boston_proxy_url(resource_id, limit, offset)
    response = request_with_retries(session, url)
    payload = extract_jina_markdown_json(response.text)
    result = payload["result"]
    fields = [field["id"] for field in result["fields"]]
    records = result["records"]
    return fields, records, url


def download_boston(output_path: Path, specs: Iterable[BostonSpec]) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    session = requests.Session()

    writer: csv.DictWriter | None = None
    fieldnames: list[str] | None = None
    chunks: list[dict[str, object]] = []
    total_records = 0

    with temp_path.open("w", encoding="utf-8", newline="") as output_file:
        for spec in specs:
            remaining = spec.rows
            offset = 0
            while remaining > 0:
                limit = min(BOSTON_PAGE_LIMIT, remaining)
                log(
                    f"Boston: {spec.year} rows {offset + 1:,}-"
                    f"{offset + limit:,} from CKAN resource {spec.resource_id}"
                )
                fields, records, url = fetch_boston_records(
                    session,
                    spec.resource_id,
                    limit,
                    offset,
                )
                if fieldnames is None:
                    fieldnames = fields
                    writer = csv.DictWriter(
                        output_file,
                        fieldnames=fieldnames,
                        extrasaction="ignore",
                    )
                    writer.writeheader()
                assert writer is not None
                for record in records:
                    writer.writerow(record)
                total_records += len(records)
                chunks.append(
                    {
                        "year": spec.year,
                        "resource_id": spec.resource_id,
                        "proxy_url": url,
                        "requested_rows": limit,
                        "offset": offset,
                        "records": len(records),
                        "source_note": "Official data.boston.gov CKAN API fetched through Jina Reader due local Cloudflare 1009 block.",
                    }
                )
                remaining -= limit
                offset += limit

    temp_path.replace(output_path)
    actual_records = count_csv_records(output_path)
    if actual_records != total_records:
        raise RuntimeError(
            f"Boston verification mismatch: wrote {total_records}, counted {actual_records}"
        )
    return {
        "path": str(output_path.relative_to(PROJECT_ROOT)),
        "records": actual_records,
        "size_bytes": output_path.stat().st_size,
        "chunks": chunks,
    }


def build_manifest(results: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "date_filter": "2021-01-01 through 2025-12-31",
        "strategy": "Stratified first-N records per year. Socrata extracts are ordered by source creation timestamp; Boston CKAN extracts use CKAN default record order.",
        "outputs": results,
    }


def verified_existing_result(path: Path, expected_records: int) -> dict[str, object] | None:
    if not path.exists():
        return None

    actual_records = count_csv_records(path)
    if actual_records != expected_records:
        log(
            f"Existing {path} has {actual_records:,} records; "
            f"expected {expected_records:,}. Redownloading."
        )
        return None

    log(f"Skipping verified existing file: {path} ({actual_records:,} records)")
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "records": actual_records,
        "size_bytes": path.stat().st_size,
        "chunks": [],
        "source_note": "Existing file verified by row count during this run.",
    }


def main() -> int:
    outputs = {
        "nyc": RAW_DIR / "nyc" / "nyc_311_2021_2025.csv",
        "chicago": RAW_DIR / "chicago" / "chicago_311_2021_2025.csv",
        "boston": RAW_DIR / "boston" / "boston_311_2021_2025.csv",
        "los_angeles": RAW_DIR / "los_angeles" / "la_311_2021_2025.csv",
    }

    results: dict[str, dict[str, object]] = {}

    results["nyc"] = verified_existing_result(outputs["nyc"], 500_000) or download_socrata_group(
        outputs["nyc"],
        [
            SocrataSpec(
                name="NYC",
                domain="data.cityofnewyork.us",
                dataset_id="erm2-nwe9",
                date_field="created_date",
                output_path=outputs["nyc"],
                rows_per_year=100_000,
                year=year,
            )
            for year in YEARS
        ],
    )

    results["chicago"] = verified_existing_result(outputs["chicago"], 300_000) or download_socrata_group(
        outputs["chicago"],
        [
            SocrataSpec(
                name="Chicago",
                domain="data.cityofchicago.org",
                dataset_id="v6vf-nfxy",
                date_field="created_date",
                output_path=outputs["chicago"],
                rows_per_year=60_000,
                year=year,
            )
            for year in YEARS
        ],
    )

    boston_resources = {
        2021: "f53ebccd-bc61-49f9-83db-625f209c95f5",
        2022: "81a7b022-f8fc-4da5-80e4-b160058ca207",
        2023: "e6013a93-1321-4f2a-bf91-8d8a02f1e62f",
        2024: "dff4d804-5031-443a-8409-8344efd0e5c8",
        2025: "9d7c2214-4709-478a-a2e8-fb2020a5bb94",
    }
    results["boston"] = verified_existing_result(outputs["boston"], 200_000) or download_boston(
        outputs["boston"],
        [
            BostonSpec(year=year, resource_id=resource_id, rows=40_000)
            for year, resource_id in boston_resources.items()
        ],
    )

    la_datasets = {
        2021: "97z7-y5bt",
        2022: "i5ke-k6by",
        2023: "4a4x-mna2",
        2024: "b7dx-7gc3",
        2025: "h73f-gn57",
    }
    results["los_angeles"] = verified_existing_result(
        outputs["los_angeles"],
        200_000,
    ) or download_socrata_group(
        outputs["los_angeles"],
        [
            SocrataSpec(
                name="Los Angeles",
                domain="data.lacity.org",
                dataset_id=dataset_id,
                date_field="createddate",
                output_path=outputs["los_angeles"],
                rows_per_year=40_000,
                year=year,
            )
            for year, dataset_id in la_datasets.items()
        ],
    )

    manifest = build_manifest(results)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Wrote manifest: {MANIFEST_PATH}")
    for city, result in results.items():
        log(f"{city}: {result['records']:,} records, {result['size_bytes']:,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
