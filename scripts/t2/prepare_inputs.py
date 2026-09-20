from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t2.provenance import git_state, sha256_file

YEARS = (2021, 2022, 2023, 2024, 2025)
ROWS_PER_YEAR = 4_000
FETCH_PER_YEAR = 6_000
REQUEST_TIMEOUT = (15, 180)
LEGACY_BOUNDED_INPUT_BUILDER_RETIRED = True

NYC_DATASET = "erm2-nwe9"
CHICAGO_DATASET = "v6vf-nfxy"
LA_DATASETS = {
    2021: "97z7-y5bt",
    2022: "i5ke-k6by",
    2023: "4a4x-mna2",
    2024: "b7dx-7gc3",
    # Los Angeles changed the MyLA311 feed/schema in 2025.
    # Official current dataset: "MyLA311 Cases March 2025 to December 2025".
    2025: "73a2-6ar5",
}
BOSTON_RESOURCES = {
    2021: "f53ebccd-bc61-49f9-83db-625f209c95f5",
    2022: "81a7b022-f8fc-4da5-80e4-b160058ca207",
    2023: "e6013a93-1321-4f2a-bf91-8d8a02f1e62f",
    2024: "dff4d804-5031-443a-8409-8344efd0e5c8",
    2025: "9d7c2214-4709-478a-a2e8-fb2020a5bb94",
}

HARMONIZED_COLUMNS = [
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
]
REQUIRED_COLUMNS = set(HARMONIZED_COLUMNS)


def _request_json(url: str, params: dict[str, object]) -> object:
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
                headers={"User-Agent": "GovTrust-FL-T2/1.0 research diagnostic"},
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < 4:
                time.sleep(min(2**attempt, 15))
    raise RuntimeError(f"failed public-data request: {url}") from last_error


def _socrata(
    *,
    domain: str,
    dataset_id: str,
    date_field: str,
    id_field: str,
    year: int,
) -> pd.DataFrame:
    start = f"{year}-01-01T00:00:00"
    end = f"{year}-12-31T23:59:59"
    params = {
        "$limit": FETCH_PER_YEAR,
        "$order": f"{date_field} ASC, {id_field} ASC",
        "$where": f"{date_field} between '{start}' and '{end}'",
    }
    payload = _request_json(
        f"https://{domain}/resource/{dataset_id}.json",
        params,
    )
    if not isinstance(payload, list):
        raise TypeError(f"unexpected Socrata payload for {domain}/{dataset_id}")
    return pd.DataFrame(payload)


def _extract_jina_json(text: str) -> dict[str, object]:
    marker = "Markdown Content:\n"
    payload = text[text.find(marker) + len(marker) :].strip() if marker in text else text.strip()
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise TypeError("unexpected Jina/CKAN payload")
    return parsed


def _boston(year: int) -> pd.DataFrame:
    resource_id = BOSTON_RESOURCES[year]
    params: dict[str, object] = {
        "resource_id": resource_id,
        "limit": FETCH_PER_YEAR,
        "offset": 0,
        "sort": "open_dt asc,case_enquiry_id asc",
    }
    url = "https://data.boston.gov/api/3/action/datastore_search"
    try:
        payload = _request_json(url, params)
    except RuntimeError:
        query = urlencode(params)
        fallback = (
            "https://r.jina.ai/http://data.boston.gov/api/3/action/"
            f"datastore_search?{query}"
        )
        response = requests.get(
            fallback,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "GovTrust-FL-T2/1.0 research diagnostic"},
        )
        response.raise_for_status()
        payload = _extract_jina_json(response.text)
    if not isinstance(payload, dict) or not payload.get("success"):
        raise RuntimeError(f"Boston CKAN request failed for {year}")
    result = payload.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("records"), list):
        raise TypeError(f"unexpected Boston CKAN result for {year}")
    return pd.DataFrame(result["records"])


def _series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(pd.NA, index=frame.index, dtype="object")
    return frame[column]


def _first_nonempty(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in columns:
        values = _series(frame, column).astype("string").str.strip().replace("", pd.NA)
        out = out.fillna(values)
    return out


def _harmonize(raw: pd.DataFrame, city: str) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    if city == "nyc":
        mapping = {
            "request_id": "unique_key",
            "created_date": "created_date",
            "closed_date": "closed_date",
            "status": "status",
            "category": "complaint_type",
            "descriptor": "descriptor",
            "agency": "agency",
            "latitude": "latitude",
            "longitude": "longitude",
        }
        area = _first_nonempty(raw, ["community_board", "borough"])
    elif city == "chicago":
        mapping = {
            "request_id": "sr_number",
            "created_date": "created_date",
            "closed_date": "closed_date",
            "status": "status",
            "category": "sr_type",
            "descriptor": "sr_short_code",
            "agency": "owner_department",
            "latitude": "latitude",
            "longitude": "longitude",
        }
        area = _first_nonempty(raw, ["community_area", "ward"])
    elif city == "boston":
        mapping = {
            "request_id": "case_enquiry_id",
            "created_date": "open_dt",
            "closed_date": "closed_dt",
            "status": "case_status",
            "category": "reason",
            "descriptor": "type",
            "latitude": "latitude",
            "longitude": "longitude",
        }
        out["agency"] = _first_nonempty(raw, ["subject", "department"])
        area = _first_nonempty(raw, ["neighborhood", "ward"])
    elif city == "los_angeles":
        if "casenumber" in raw.columns:
            # 2025+ MyLA311 Cases schema.
            mapping = {
                "request_id": "casenumber",
                "created_date": "createddate",
                "closed_date": "closeddate",
                "status": "status",
                "category": "type",
                "descriptor": "action_taken__c",
                "agency": "department_name__c",
                "latitude": "latitude",
                "longitude": "longitude",
            }
            area = _first_nonempty(
                raw,
                [
                    "neighborhood_council_name__c",
                    "neighborhood_council__c",
                    "ncname",
                    "nc",
                    "cd",
                ],
            )
        else:
            # 2021-2024 legacy MyLA311 Service Request schema.
            mapping = {
                "request_id": "srnumber",
                "created_date": "createddate",
                "closed_date": "closeddate",
                "status": "status",
                "category": "requesttype",
                "descriptor": "actiontaken",
                "agency": "owner",
                "latitude": "latitude",
                "longitude": "longitude",
            }
            area = _first_nonempty(raw, ["ncname", "nc", "cd"])
    else:
        raise ValueError(f"unknown city: {city}")

    for target, source in mapping.items():
        out[target] = _series(raw, source)
    out["area"] = area
    out["city"] = city
    for column in HARMONIZED_COLUMNS:
        if column not in out:
            out[column] = pd.NA
    return out[HARMONIZED_COLUMNS]


def _fetch_harmonized_city(city: str) -> tuple[pd.DataFrame, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, object]] = []
    for year in YEARS:
        coverage_note: str | None = None
        if city == "nyc":
            raw = _socrata(
                domain="data.cityofnewyork.us",
                dataset_id=NYC_DATASET,
                date_field="created_date",
                id_field="unique_key",
                year=year,
            )
            source_id = NYC_DATASET
        elif city == "chicago":
            raw = _socrata(
                domain="data.cityofchicago.org",
                dataset_id=CHICAGO_DATASET,
                date_field="created_date",
                id_field="sr_number",
                year=year,
            )
            source_id = CHICAGO_DATASET
        elif city == "los_angeles":
            source_id = LA_DATASETS[year]
            if year == 2025:
                raw = _socrata(
                    domain="data.lacity.org",
                    dataset_id=source_id,
                    date_field="createddate",
                    id_field="casenumber",
                    year=year,
                )
                coverage_note = (
                    "Official MyLA311 Cases 2025 source begins in March 2025; "
                    "the 2025 diagnostic sample is therefore drawn from the available "
                    "March-December 2025 source period."
                )
            else:
                raw = _socrata(
                    domain="data.lacity.org",
                    dataset_id=source_id,
                    date_field="createddate",
                    id_field="srnumber",
                    year=year,
                )
        elif city == "boston":
            source_id = BOSTON_RESOURCES[year]
            raw = _boston(year)
        else:
            raise ValueError(f"unknown city: {city}")
        frames.append(_harmonize(raw, city))
        source_row: dict[str, object] = {
            "year": year,
            "source_id": source_id,
            "fetched_rows": len(raw),
            "requested_limit": FETCH_PER_YEAR,
        }
        if coverage_note is not None:
            source_row["coverage_note"] = coverage_note
        sources.append(source_row)
    return pd.concat(frames, ignore_index=True), {"yearly_sources": sources}


def _select(frame: pd.DataFrame, city: str) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{city}: harmonized frame missing {sorted(missing)}")

    work = frame.copy()
    observed_cities = set(work["city"].dropna().astype(str).str.strip().unique())
    if observed_cities != {city}:
        raise ValueError(f"{city}: unexpected city values {sorted(observed_cities)}")

    work["request_id"] = work["request_id"].astype("string").str.strip()
    work["_created"] = pd.to_datetime(work["created_date"], errors="coerce", utc=True)
    work = work[work["request_id"].notna() & work["request_id"].ne("")].copy()
    work = work[work["_created"].notna()].copy()
    work = work.sort_values(["_created", "request_id"], kind="stable")
    work = work.drop_duplicates(subset=["request_id"], keep="first").copy()
    work["_year"] = work["_created"].dt.year

    parts: list[pd.DataFrame] = []
    by_year: dict[str, int] = {}
    for year in YEARS:
        group = work[work["_year"].eq(year)].copy()
        if len(group) < ROWS_PER_YEAR:
            raise RuntimeError(
                f"{city}: year {year} has only {len(group)} eligible rows; "
                f"requires {ROWS_PER_YEAR}"
            )
        chosen = group.head(ROWS_PER_YEAR)
        parts.append(chosen)
        by_year[str(year)] = len(chosen)

    selected = pd.concat(parts, ignore_index=True)
    selected = selected.drop(columns=["_created", "_year"])
    if len(selected) != ROWS_PER_YEAR * len(YEARS):
        raise AssertionError("diagnostic extract row count mismatch")
    return selected.reset_index(drop=True), by_year


def prepare_archive(archive_dir: Path) -> dict[str, object]:
    if LEGACY_BOUNDED_INPUT_BUILDER_RETIRED:
        raise RuntimeError(
            "Legacy bounded 20k-per-city diagnostic input builder is retired. "
            "Use only the frozen full-data harmonized corpus after pre-training gates pass."
        )
    sha, dirty = git_state()
    if sha == "UNKNOWN" or dirty:
        raise RuntimeError("T2 input archive must be prepared from a clean Git checkout")

    archive_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, object] = {}
    for city in ("nyc", "chicago", "boston", "los_angeles"):
        print(f"Fetching bounded public diagnostic source for {city}...", flush=True)
        harmonized, source_meta = _fetch_harmonized_city(city)
        selected, by_year = _select(harmonized, city)
        output_path = archive_dir / f"{city}.parquet"
        selected.to_parquet(output_path, index=False)
        output_sha = sha256_file(output_path)
        created = pd.to_datetime(selected["created_date"], errors="coerce", utc=True)
        outputs[city] = {
            **source_meta,
            "archive_path": str(output_path),
            "archive_sha256": output_sha,
            "rows": len(selected),
            "unique_request_ids": int(selected["request_id"].nunique()),
            "rows_by_year": by_year,
            "created_date_min": created.min().isoformat(),
            "created_date_max": created.max().isoformat(),
            "columns": list(selected.columns),
        }

    manifest = {
        "protocol": "t2_diagnostic_input_archive_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha,
        "git_dirty": dirty,
        "source_stage": "bounded public municipal API fetch harmonized to Step-2 schema",
        "historical_step3_cleaned_forbidden": True,
        "selection": {
            "uses_outcome_or_closed_date": False,
            "uses_status": False,
            "uses_only": ["request_id", "created_date", "city"],
            "years": list(YEARS),
            "fetch_limit_per_year_per_city": FETCH_PER_YEAR,
            "rows_per_year_per_city": ROWS_PER_YEAR,
            "order": "created_date ascending, request_id ascending",
            "deduplicate_request_id_before_sampling": True,
        },
        "outputs": outputs,
    }
    manifest_path = archive_dir / "t2_input_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch/create the frozen 20k-per-city T2 diagnostic input archive"
    )
    parser.add_argument("--archive-dir", required=True)
    return parser


def main() -> int:
    if LEGACY_BOUNDED_INPUT_BUILDER_RETIRED:
        raise SystemExit(
            "Legacy bounded diagnostic input builder is retired; no inputs generated."
        )
    args = build_parser().parse_args()
    manifest = prepare_archive(Path(args.archive_dir).expanduser().resolve())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
