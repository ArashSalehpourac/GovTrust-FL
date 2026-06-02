"""Official municipal open-data download helpers.

These utilities define the canonical source endpoints used by the project.
They intentionally avoid committing downloaded raw data; outputs should be
written under ``data/raw/<city>/`` or another user-specified directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

import json
import pandas as pd
import requests


@dataclass(frozen=True)
class OpenDataSource:
    """Metadata for one official municipal open-data source."""

    city: str
    source_name: str
    base_url: str
    date_field: str
    dataset_ids: dict[int, str]


OPEN_DATA_SOURCES = {
    "nyc": OpenDataSource(
        city="nyc",
        source_name="NYC 311 Service Requests",
        base_url="https://data.cityofnewyork.us/resource/{dataset_id}.csv",
        date_field="created_date",
        dataset_ids={year: "erm2-nwe9" for year in range(2021, 2026)},
    ),
    "chicago": OpenDataSource(
        city="chicago",
        source_name="Chicago 311 Service Requests",
        base_url="https://data.cityofchicago.org/resource/{dataset_id}.csv",
        date_field="created_date",
        dataset_ids={year: "v6vf-nfxy" for year in range(2021, 2026)},
    ),
    "boston": OpenDataSource(
        city="boston",
        source_name="Boston 311 Service Requests",
        base_url="https://data.boston.gov/api/3/action/datastore_search?resource_id={dataset_id}",
        date_field="open_dt",
        dataset_ids={
            2021: "f53ebccd-bc61-49f9-83db-625f209c95f5",
            2022: "81a7b022-f8fc-4da5-80e4-b160058ca207",
            2023: "e6013a93-1321-4f2a-bf91-8d8a02f1e62f",
            2024: "dff4d804-5031-443a-8409-8344efd0e5c8",
            2025: "9d7c2214-4709-478a-a2e8-fb2020a5bb94",
        },
    ),
    "los_angeles": OpenDataSource(
        city="los_angeles",
        source_name="Los Angeles MyLA311 Service Request Data",
        base_url="https://data.lacity.org/resource/{dataset_id}.csv",
        date_field="createddate",
        dataset_ids={
            2021: "97z7-y5bt",
            2022: "i5ke-k6by",
            2023: "4a4x-mna2",
            2024: "b7dx-7gc3",
            2025: "h73f-gn57",
        },
    ),
}


def source_url(city: str, year: int) -> str:
    """Return the official source URL for one city/year pair."""

    source = OPEN_DATA_SOURCES[city]
    return source.base_url.format(dataset_id=source.dataset_ids[year])


def download_csv_sample(
    city: str,
    output_path: str | Path,
    *,
    year: int = 2025,
    limit: int = 1_000,
) -> Path:
    """Download a small CSV sample from a Socrata source.

    Boston's CKAN endpoint is schema-compatible but not Socrata; callers should
    use the existing ``scripts/download_step1_raw.py`` for the full Boston
    multi-year extract. This helper is intended for lightweight ad hoc samples.
    """

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if city == "boston":
        return download_boston_sample(output, year=year, limit=limit)

    source = OPEN_DATA_SOURCES[city]
    url = source_url(city, year)
    query = urlencode(
        {
            "$limit": limit,
            "$order": source.date_field,
        }
    )
    frame = pd.read_csv(f"{url}?{query}")
    frame.to_csv(output, index=False)
    return output


def download_boston_sample(output_path: Path, *, year: int = 2025, limit: int = 1_000) -> Path:
    """Download a small Boston CKAN sample as CSV."""

    source = OPEN_DATA_SOURCES["boston"]
    resource_id = source.dataset_ids[year]
    params = urlencode({"resource_id": resource_id, "limit": limit, "sort": f"{source.date_field} asc"})
    url = f"https://data.boston.gov/api/3/action/datastore_search?{params}"
    response = requests.get(url, timeout=(15, 120), headers={"User-Agent": "GovTrust-FL/0.1"})
    response.raise_for_status()
    payload = json.loads(response.text)
    records = payload["result"]["records"]
    frame = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path
