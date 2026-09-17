# Real municipal 311 data rebuild (2021-2025)

This directory documents the approved real-data acquisition pathway for the T2
experiments. It supersedes every earlier T2 data workflow: pilot results, earlier
result CSVs, the 80k sampled parquet archive, PRE_VAST reports and any synthetic,
generated or stub metrics are **not** valid experimental inputs or evidence.

## Pipeline (scripts/real_data)

| step | script | network | output |
|---|---|---|---|
| 1 | `acquire_raw.py` | yes (official portals only) | `01_Raw_Official/CITY_YYYY.jsonl.gz` + `.sha256` + `.provenance.json` |
| 2 | `audit_raw.py` | no | `03_Manifests_and_Checksums/RAW_MANIFEST.json`, `RAW_SHA256SUMS.txt`, `RAW_AUDIT_REPORT.md` |
| 3 | `harmonize_from_raw.py` | no | `02_Harmonized/<city>_2021_2025_harmonized.parquet` + `.sha256` + `.manifest.json`, `HARMONIZATION_MANIFEST.json` (aggregated from the per-city `.manifest.json` sidecars), `ROW_ACCOUNTING.csv`, `HARMONIZED_SHA256SUMS.txt` |
| 4 | `report_gates.py` | no | `03_Manifests_and_Checksums/GATE_REPORT.txt` |

```bash
ARCHIVE=/path/to/"Real Datasets — 2021-2025"
python scripts/real_data/acquire_raw.py --archive-dir "$ARCHIVE"          # all 20 city/years
python scripts/real_data/audit_raw.py --archive-dir "$ARCHIVE"
python scripts/real_data/harmonize_from_raw.py --archive-dir "$ARCHIVE"
python scripts/real_data/report_gates.py --archive-dir "$ARCHIVE"
```

## Official sources (frozen in `sources.py`)

| city | source | id |
|---|---|---|
| NYC | Socrata `data.cityofnewyork.us`, "311 Service Requests from 2020 to Present" | `erm2-nwe9` (year filter on `created_date`) |
| Chicago | Socrata `data.cityofchicago.org`, "311 Service Requests" | `v6vf-nfxy` (year filter on `created_date`) |
| Boston | CKAN `data.boston.gov` yearly resources | 2021 `f53ebccd-…`, 2022 `81a7b022-…`, 2023 `e6013a93-…`, 2024 `dff4d804-…`, 2025 `9d7c2214-…` (entire resource) |
| Los Angeles | Socrata `data.lacity.org` yearly datasets | 2021 `97z7-y5bt`, 2022 `i5ke-k6by`, 2023 `4a4x-mna2`, 2024 `b7dx-7gc3`, 2025 `73a2-6ar5` (year filter on `createddate`) |

**Los Angeles 2025**: the official dataset `73a2-6ar5` is "MyLA311 Cases March 2025 to
December 2025". It covers March-December 2025 only (first record 2025-03-28) and uses a
different schema (`casenumber`, `type`, `action_taken__c`, `department_name__c`,
`geolocation__latitude__s/longitude__s`, `locator_sr_neigborhood_council_1`). This is
recorded in the provenance sidecar, `RAW_MANIFEST.json` and `HARMONIZATION_MANIFEST.json`.

## Acquisition rules implemented

- Real records only; every row is returned by the official API and written verbatim
  (one JSON object per line, values untouched).
- Pagination continues until the source is exhausted: Socrata keyset paging on the row
  identifier `:id` with `$limit=50000`; CKAN offset paging with `limit=32000`
  sorted by `_id`. No head-of-list sampling.
- Query predicates use only the creation-date year window (Socrata) or no filter at
  all (CKAN). No status, closed-date or other outcome field is used to select rows.
- Snapshots are written to a `.partial` file and atomically renamed, then made
  read-only. An existing `CITY_YYYY.jsonl.gz` is never overwritten; `--force-new-run`
  writes a separate timestamped file.
- `CITY_YYYY.jsonl.gz.provenance.json` records: official URL, dataset/resource id,
  query parameters, retrieval UTC start/finish, API page count and per-page log, raw
  rows, first/last created timestamp, unique and duplicate request ids, SHA256, size.
- `audit_raw.py` recomputes every statistic and hash from the frozen file independently
  of the downloader.

## Harmonization rules

- Input is exclusively the frozen raw snapshots; hashes are re-verified before reading.
- Row-lossless: every raw row produces exactly one harmonized row. No dedup, no
  status/outcome filtering, no geographic filtering (those are experiment-design
  decisions, taken later and documented separately).
- Canonical columns: `request_id, created_date, closed_date, status, category,
  descriptor, agency, latitude, longitude, area, city, source_year`.
- `created_date`/`closed_date` are parsed to naive microsecond timestamps in source-local
  time; `latitude`/`longitude` to float64; everything else stays string. Empty strings
  become null. Field mapping per city is stored in the manifest.

## Scope

No model training, no Vast usage and no result generation are part of this pathway.
`GATE_REPORT.txt` is the pre-experiment gate record.
