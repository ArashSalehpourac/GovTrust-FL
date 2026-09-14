"""Capped pilot extract of the four municipal 311 systems.

This module downloads a deliberately small, reproducible slice of the same
official sources used by the historical Step 1 downloader, harmonizes it into
the shared schema, and applies the historical Step 3 cleaning rules.

It never writes into ``data/raw`` or ``data/processed``; the historical
artifacts stay untouched.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cleaning import clean_frame  # noqa: E402
from src.schema_harmonization import TARGET_SCHEMA, first_nonempty  # noqa: E402
from src.redesign.config import PILOT_CLEAN_DIR, PILOT_RAW_DIR  # noqa: E402

REQUEST_TIMEOUT = (15, 180)
PAGE_LIMIT = 10_000
USER_AGENT = "GovTrust-FL-redesign/0.1 pilot extract"


@dataclass(frozen=True)
class SocrataSource:
    """One Socrata-backed city extract."""

    city: str
    domain: str
    datasets: tuple[tuple[int, str], ...]
    date_field: str
    select_fields: tuple[str, ...]
    mapping: dict[str, str | Callable[[pd.DataFrame], pd.Series]]


@dataclass(frozen=True)
class CkanSource:
    """One CKAN-backed city extract (Boston)."""

    city: str
    base_url: str
    resources: tuple[tuple[int, str], ...]
    mapping: dict[str, str | Callable[[pd.DataFrame], pd.Series]]


SOCRATA_SOURCES: tuple[SocrataSource, ...] = (
    SocrataSource(
        city="nyc",
        domain="data.cityofnewyork.us",
        datasets=((2024, "erm2-nwe9"), (2025, "erm2-nwe9")),
        date_field="created_date",
        select_fields=(
            "unique_key",
            "created_date",
            "closed_date",
            "status",
            "complaint_type",
            "descriptor",
            "agency",
            "latitude",
            "longitude",
            "community_board",
            "borough",
        ),
        mapping={
            "request_id": "unique_key",
            "created_date": "created_date",
            "closed_date": "closed_date",
            "status": "status",
            "category": "complaint_type",
            "descriptor": "descriptor",
            "agency": "agency",
            "latitude": "latitude",
            "longitude": "longitude",
            "area": lambda frame: first_nonempty(frame, ["community_board", "borough"]),
        },
    ),
    SocrataSource(
        city="chicago",
        domain="data.cityofchicago.org",
        datasets=((2024, "v6vf-nfxy"), (2025, "v6vf-nfxy")),
        date_field="created_date",
        select_fields=(
            "sr_number",
            "created_date",
            "closed_date",
            "status",
            "sr_type",
            "sr_short_code",
            "owner_department",
            "latitude",
            "longitude",
            "community_area",
            "ward",
        ),
        mapping={
            "request_id": "sr_number",
            "created_date": "created_date",
            "closed_date": "closed_date",
            "status": "status",
            "category": "sr_type",
            "descriptor": "sr_short_code",
            "agency": "owner_department",
            "latitude": "latitude",
            "longitude": "longitude",
            "area": lambda frame: first_nonempty(frame, ["community_area", "ward"]),
        },
    ),
    SocrataSource(
        city="los_angeles",
        domain="data.lacity.org",
        datasets=((2024, "b7dx-7gc3"), (2025, "h73f-gn57")),
        date_field="createddate",
        select_fields=(
            "srnumber",
            "createddate",
            "closeddate",
            "status",
            "requesttype",
            "actiontaken",
            "owner",
            "latitude",
            "longitude",
            "ncname",
            "nc",
            "cd",
        ),
        mapping={
            "request_id": "srnumber",
            "created_date": "createddate",
            "closed_date": "closeddate",
            "status": "status",
            "category": "requesttype",
            "descriptor": "actiontaken",
            "agency": "owner",
            "latitude": "latitude",
            "longitude": "longitude",
            "area": lambda frame: first_nonempty(frame, ["ncname", "nc", "cd"]),
        },
    ),
)

BOSTON_SOURCE = CkanSource(
    city="boston",
    base_url="https://data.boston.gov/api/3/action/datastore_search",
    resources=(
        (2024, "dff4d804-5031-443a-8409-8344efd0e5c8"),
        (2025, "9d7c2214-4709-478a-a2e8-fb2020a5bb94"),
    ),
    mapping={
        "request_id": "case_enquiry_id",
        "created_date": "open_dt",
        "closed_date": "closed_dt",
        "status": "case_status",
        "category": "reason",
        "descriptor": "type",
        "agency": lambda frame: first_nonempty(frame, ["subject", "department"]),
        "latitude": "latitude",
        "longitude": "longitude",
        "area": lambda frame: first_nonempty(frame, ["neighborhood", "ward"]),
    },
)


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _get_json(session: requests.Session, url: str, params: dict[str, object]) -> object:
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as error:  # noqa: BLE001 - retried below
            last_error = error
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"Failed request after retries: {url}") from last_error


def fetch_socrata(source: SocrataSource, rows_per_city: int) -> pd.DataFrame:
    """Fetch a capped, chronologically ordered slice for one Socrata city."""

    session = _session()
    rows_per_dataset = rows_per_city // len(source.datasets)
    frames: list[pd.DataFrame] = []
    for year, dataset_id in source.datasets:
        remaining = rows_per_dataset
        offset = 0
        while remaining > 0:
            limit = min(PAGE_LIMIT, remaining)
            params = {
                "$select": ",".join(source.select_fields),
                "$where": (
                    f"{source.date_field} between '{year}-01-01T00:00:00' "
                    f"and '{year}-12-31T23:59:59'"
                ),
                "$order": source.date_field,
                "$limit": limit,
                "$offset": offset,
            }
            payload = _get_json(
                session,
                f"https://{source.domain}/resource/{dataset_id}.json",
                params,
            )
            records = list(payload) if isinstance(payload, list) else []
            if not records:
                break
            frames.append(pd.DataFrame.from_records(records).astype("string"))
            offset += limit
            remaining -= limit
    if not frames:
        raise RuntimeError(f"No rows returned for {source.city}")
    return pd.concat(frames, ignore_index=True)


def fetch_ckan(source: CkanSource, rows_per_city: int) -> pd.DataFrame:
    """Fetch a capped slice for the CKAN-backed Boston dataset."""

    session = _session()
    rows_per_resource = rows_per_city // len(source.resources)
    frames: list[pd.DataFrame] = []
    for _year, resource_id in source.resources:
        remaining = rows_per_resource
        offset = 0
        while remaining > 0:
            limit = min(5_000, remaining)
            payload = _get_json(
                session,
                source.base_url,
                {"resource_id": resource_id, "limit": limit, "offset": offset},
            )
            records = payload["result"]["records"] if isinstance(payload, dict) else []
            if not records:
                break
            frames.append(pd.DataFrame.from_records(records).astype("string"))
            offset += limit
            remaining -= limit
    if not frames:
        raise RuntimeError("No rows returned for boston")
    return pd.concat(frames, ignore_index=True)


def harmonize(raw: pd.DataFrame, city: str, mapping: dict[str, object]) -> pd.DataFrame:
    """Map a raw city frame onto the shared harmonized schema."""

    output = pd.DataFrame(index=raw.index)
    for column in TARGET_SCHEMA:
        if column == "city":
            output[column] = city
            continue
        source = mapping[column]
        if callable(source):
            output[column] = source(raw)
        else:
            output[column] = raw[source] if source in raw.columns else pd.NA

    for column in ("request_id", "status", "category", "descriptor", "agency", "area", "city"):
        output[column] = output[column].astype("string").str.strip()
        output[column] = output[column].mask(output[column].eq(""))
    for column in ("created_date", "closed_date"):
        output[column] = pd.to_datetime(output[column], errors="coerce")
    for column in ("latitude", "longitude"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output[TARGET_SCHEMA]


def build_pilot_extract(
    rows_per_city: int,
    raw_dir: Path = PILOT_RAW_DIR,
    clean_dir: Path = PILOT_CLEAN_DIR,
    min_category_samples: int = 50,
) -> dict[str, object]:
    """Download, harmonize, and clean the capped pilot extract for all cities."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, object]] = []
    for source in (*SOCRATA_SOURCES, BOSTON_SOURCE):
        if isinstance(source, SocrataSource):
            raw = fetch_socrata(source, rows_per_city)
            source_note = f"Socrata {source.domain} {[d for _, d in source.datasets]}"
        else:
            raw = fetch_ckan(source, rows_per_city)
            source_note = f"CKAN {source.base_url} {[r for _, r in source.resources]}"

        raw_path = raw_dir / f"{source.city}_pilot_raw.parquet"
        raw.to_parquet(raw_path, index=False)

        harmonized = harmonize(raw, source.city, source.mapping)
        cleaned, counts = clean_frame(harmonized, min_category_samples=min_category_samples)
        clean_path = clean_dir / f"{source.city}_pilot_clean.parquet"
        cleaned.to_parquet(clean_path, index=False)

        outputs.append(
            {
                "city": source.city,
                "source": source_note,
                "raw_path": str(raw_path),
                "clean_path": str(clean_path),
                "raw_rows": int(len(raw)),
                "clean_rows": int(len(cleaned)),
                "cleaning_counts": counts,
                "created_date_min": _iso_or_none(cleaned["created_date"].min()),
                "created_date_max": _iso_or_none(cleaned["created_date"].max()),
                "categories": int(cleaned["category"].nunique(dropna=True)),
            }
        )

    manifest = {
        "extract": "pilot",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rows_per_city_cap": rows_per_city,
        "window": "2024-01-01 through 2025-12-31, earliest-first per source dataset",
        "cleaning_rules": "Historical Step 3 rules from src.cleaning.clean_frame",
        "min_category_samples": min_category_samples,
        "outputs": outputs,
    }
    manifest_path = clean_dir / "pilot_extract_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_clean_city(city: str, clean_dir: Path = PILOT_CLEAN_DIR) -> pd.DataFrame:
    """Load one cleaned pilot city frame."""

    return pd.read_parquet(clean_dir / f"{city}_pilot_clean.parquet")


def _iso_or_none(value) -> str | None:
    if pd.isna(value):
        return None
    return value.isoformat()
