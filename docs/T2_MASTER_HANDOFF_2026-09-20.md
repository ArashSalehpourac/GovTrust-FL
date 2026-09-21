# T2-GovTrust / GovTrust-FL — Master Handoff (2026-09-20)

Canonical Drive handoff document:
https://docs.google.com/document/d/12P2-i43a0J_SQ0Z6-3izEIOorbA_C-TBRzXnSw4pv28/edit

## Frozen research design

**RQ:** How does formal differential privacy affect predictive reliability when federated municipal service models are transferred to an unseen administrative jurisdiction?

**Title:** Privacy Under Institutional Shift: Cross-Jurisdiction Reliability of Federated Learning for Municipal Service-Request Resolution

Rotating LOCO:
- NYC + Chicago + Boston -> LA
- NYC + Chicago + LA -> Boston
- NYC + Boston + LA -> Chicago
- Chicago + Boston + LA -> NYC

Primary target: `log1p(resolution_hours)` regression.  
Primary metric: MAE on log1p(hours).  
FedAvg primary. Formal example-level DP-SGD with continuously composed per-client accountant.  
No target-jurisdiction leakage into fitting/tuning/calibration/accounting.

No old pilot metrics may be treated as scientific evidence. No synthetic/stub results. No training before all data/provenance/harmonization/design gates pass.

## Project locations

- Repo: `ArashSalehpourac/GovTrust-FL`
- Issue #3: T2 implementation gate
- Draft PR #4: `t2/diagnostic-validity-gate` — DO NOT merge without explicit instruction
- Slack: `t2-govtrust-fl`, channel `C0C1WN1CEH2`
- ClickUp task: `z8r3fdf21z`, workspace `1100340000001242`
- Colab: `1-dqS1NogG2MEmGu1xGPXPqeuUDdYf0I5`
- Drive redesign root: `1Aw2vwmUpTOb_Nvr2KICyFvw4ugdHYK61`
- Raw folder: `1-fm_2DtaPSfwGAYi2uSWnSSpvAjtGi1_`
- Harmonized folder: `1xWsfDOzhNmn8VZyjHCzb7H_wCn5mD0gp`
- Manifests/checksums: `1ojONoo3rNHCv2mhTnQIAZtx-qKO4jEW3`

## Raw-data state at handoff

Accepted final raw snapshots: **19/20**.

Complete/frozen:
- NYC 2021-2025: 5/5
- Boston 2021-2025: 5/5
- LA 2021-2025: 5/5
- Chicago 2021-2024: 4/5

LA 2025 is partial-year by the official source. Actual observed coverage:
`2025-03-28 06:10:07` through `2025-12-31 12:59:54`.

## Chicago 2025

Official frozen 2025 source state:
- rows: `1,960,595`
- min: `2025-01-01T00:01:27.000`
- max: `2025-12-31T23:59:32.000`
- null IDs: `0`
- distinct IDs: `1,960,595`

Original acquisition was interrupted by Chicago/Tyler Socrata HTTP 503 at exactly 1,500,000 rows.

Safeguarded prefix:
- file: `CHICAGO_2025_PARTIAL_1500000_DO_NOT_USE.csv`
- Drive ID: `19fup7jdFA2mHrcq9GomYWrZ8Q_dKHqZC`
- size: `296,317,441`
- SHA256: `3c290b5af1200b9c62a5daf8380180f0ca2a00ffbcf15801c56639d7f4289316`
- cursor: `2025-10-05T11:31:55.000 / SR25-01827117`

State JSON:
- Drive ID: `1CdjpwHxm8PfmTw2oxbvxHfTnSFf9aVB6`

Corrected **2025-scoped** safe-resume probe PASS:
- prefix count through cursor: exactly `1,500,000`
- rows remaining at resume start: `460,595`
- `CHICAGO_2025_SAFE_RESUME_GATE=PASS`

Ignore the earlier recovery probe that reported 14,681,423 rows / 2018-2026 bounds; that query accidentally omitted the 2025 filter.

### Live resume snapshot

Saved notebook modified: `2026-09-20T18:02:25.149Z`.

Cell 84 had started final resume + validation + Drive archive:
- `SOURCE_COMPATIBILITY_GATE=PASS`
- `STAGING_INTEGRITY_GATE=PASS`
- `PREFIX_RESTORE_GATE=PASS`
- starting rows: `1,500,000`
- expected new rows: `460,595`
- resume started: `2026-09-20T18:01:05.761684+00:00`

Saved progress:
- `RESUME_PAGE_011`
- total rows: `1,720,000`
- last cursor: `2025-11-13T19:49:21.000 / SR25-02097017`
- rows remaining at that saved snapshot: `240,595`

**Do not start a second resume job while the existing Colab cell is still running.**

Do not delete the interrupted staging artifacts.

Chicago 2025 is final only when all are true:
- `CHICAGO_2025_LOCAL_GATE=PASS`
- `SOURCE_STABILITY_GATE=PASS`
- `CHICAGO_2025_DRIVE_ARCHIVE_GATE=PASS`
- final Drive `CHICAGO_2025_raw.csv` exists and independently matches final size/hash

At this handoff snapshot, final raw Drive folder did **not** yet contain `CHICAGO_2025_raw.csv`.

## After Chicago reaches 20/20

1. Update Slack, ClickUp, Issue #3 and PR #4 once for the 20/20 gate transition.
2. Create 20-file manifests/checksums in Drive `03_Manifests_and_Checksums`.
3. Audit hashes, sizes, source/date bounds, IDs, duplicates, schemas and provenance.
4. Harmonize **only from frozen Drive raw snapshots** into `02_Harmonized`.
5. Hash harmonized outputs and run chronology/leakage/data-quality gates.
6. Conduct experiment-design review.
7. Only then consider FedAvg / DP-SGD training.

Current:
`RAW_DATA_GATE=NOT_READY`
`FINAL_RAW_FILES_ACCEPTED=19/20`
`SCIENTIFIC_TRAINING_STARTED=NO`
`REAL_RESULTS_GENERATED=NO`

## New-chat continuation prompt

> Continue the T2-GovTrust / GovTrust-FL project from the exact saved state. First read the project context and the Google Drive document titled “T2-GovTrust — MASTER HANDOFF & CONTINUITY LOG — 2026-09-20” (Drive document ID `12P2-i43a0J_SQ0Z6-3izEIOorbA_C-TBRzXnSw4pv28`). Also inspect Colab notebook ID `1-dqS1NogG2MEmGu1xGPXPqeuUDdYf0I5`, the T2 Drive folders, Slack channel `C0C1WN1CEH2`, ClickUp task `z8r3fdf21z`, GitHub Issue #3, and Draft PR #4.
>
> Treat the handoff document as continuity source of truth, but independently verify live state before changing any gate.
>
> At handoff, Chicago 2025 is the last raw snapshot. The corrected 2025-scoped safe-resume probe passed. The final resume/validation/archive cell had started from the safeguarded 1,500,000-row prefix. At the saved notebook snapshot it had reached 1,720,000 rows on page 11, cursor `2025-11-13T19:49:21.000 / SR25-02097017`. Do not start another resume job if the existing Colab cell is still running. Check the latest notebook and Drive first.
>
> Only mark 20/20 when `CHICAGO_2025_LOCAL_GATE=PASS`, `SOURCE_STABILITY_GATE=PASS`, `CHICAGO_2025_DRIVE_ARCHIVE_GATE=PASS`, and final Drive `CHICAGO_2025_raw.csv` independently exists with matching size/hash.
>
> After 20/20, update Slack, ClickUp, GitHub Issue #3 and Draft PR #4 once for the gate transition. Then proceed to the 20-file manifest/checksum audit and harmonization from frozen Drive snapshots only. Do not train yet. Do not generate or infer scientific results. No synthetic/stub metrics. Do not merge PR #4 unless explicitly instructed. Preserve all provenance and the Chicago interrupted staging artifacts.


## Continuation update — raw gate closed at 20/20

Live verification after the saved handoff established that Chicago 2025 cell 84 completed successfully.

Chicago 2025 final:
- rows: `1,960,595`
- min/max created: `2025-01-01T00:01:27.000` / `2025-12-31T23:59:32.000`
- null request IDs: `0`
- duplicate request IDs: `0`
- size: `386,539,997` bytes
- MD5: `0746a1e8ce60897e72dcbd479e4c773e`
- SHA256: `30d6344d7db65aa9ac126a05e6e2adde5aa1f15c4db0e8df1628bab592dffd21`
- final Drive ID: `1ebAXhIZrBAYYnlYx6Rldm2Oot7fj_Ivb`
- `CHICAGO_2025_LOCAL_GATE=PASS`
- `SOURCE_STABILITY_GATE=PASS`
- `CHICAGO_2025_DRIVE_ARCHIVE_GATE=PASS`

Current global state:
- `FINAL_RAW_FILES_ACCEPTED=20/20`
- `RAW_DATA_GATE=PASS`
- `SCIENTIFIC_TRAINING_STARTED=NO`
- `REAL_RESULTS_GENERATED=NO`

The one-time 20/20 transition was posted to Slack, ClickUp, Issue #3 and PR #4.
Do not post a second 20/20 transition.

A central raw manifest/checksum register was created in Drive
`03_Manifests_and_Checksums`:
- title: `T2_RAW_MANIFEST_CHECKSUMS_20_FINAL_2026-09-20`
- Sheet ID: `1cNFTji5qSQBe9QRs0DsqcmFkVqRk0FDjdLY7i1vlMT0`

It records all 20 final raw Drive IDs, sizes, source IDs, date bounds,
null/duplicate ID audits, SHA256 values, schemas, and provenance constraints.

Critical repo correction:
- The older `scripts/t2/prepare_inputs.py` performs a fresh bounded API fetch and is
  not authorized for the current harmonization stage.
- Added `configs/t2/frozen_raw_manifest.json`.
- Added `src/t2/harmonize_frozen.py`.
- Added `scripts/t2/harmonize_frozen_raw.py`.
- Added `tests/t2/test_harmonize_frozen.py`.
- Updated `docs/t2/README.md` to require frozen-raw harmonization.
- Harmonization is row-preserving and refuses raw size/SHA mismatch, schema drift,
  Boston datastore-only `_id`, or date/ID audit mismatches.
- Output target remains Drive `02_Harmonized` ID
  `1xWsfDOzhNmn8VZyjHCzb7H_wCn5mD0gp`.

At this update:
- `RAW_MANIFEST_GATE=PASS`
- `HARMONIZATION_GATE=NOT_STARTED`
- `HARMONIZED_HASH_GATE=NOT_STARTED`
- `CHRONOLOGY_LEAKAGE_DATA_QUALITY_GATE=NOT_STARTED`
- `READY_FOR_EXPERIMENT_DESIGN_REVIEW=NO`
- training remains blocked.

PR #4 remains draft/unmerged.


## Continuation update — harmonizer validation green

Live re-verification after the raw 20/20 transition:

- Saved Colab cell 84 completed with `CHICAGO_2025_LOCAL_GATE=PASS`,
  `SOURCE_STABILITY_GATE=PASS`, and `CHICAGO_2025_DRIVE_ARCHIVE_GATE=PASS`.
- Chicago 2025 final Drive artifact:
  - rows: `1,960,595`
  - size: `386,539,997` bytes
  - SHA256: `30d6344d7db65aa9ac126a05e6e2adde5aa1f15c4db0e8df1628bab592dffd21`
  - Drive ID: `1ebAXhIZrBAYYnlYx6Rldm2Oot7fj_Ivb`
- `FINAL_RAW_FILES_ACCEPTED=20/20`
- `RAW_DATA_GATE=PASS`
- Central raw manifest/checksum Sheet ID:
  `1cNFTji5qSQBe9QRs0DsqcmFkVqRk0FDjdLY7i1vlMT0`
- Raw manifest, SHA256, date-range, duplicate-ID, schema and provenance gates are PASS.

Pre-harmonization code gate:
- README Drive path corrected to the live redesign-root mount.
- Ruff findings in the frozen harmonizer/test were repaired without changing scientific semantics.
- Validated branch commit:
  `64de331fde4554c9af1bdcd306acb903dcff590f`
- GitHub Actions T2 Validation run:
  `35530429768`
- preflight, 31 T2 tests, full test discovery, Ruff, compile, and notebook-artifact checks: PASS.
- `FROZEN_RAW_HARMONIZER_CODE_GATE=PASS`

Live data-stage state:
- Drive `02_Harmonized` is still empty.
- `HARMONIZATION_GATE=NOT_STARTED`
- `HARMONIZED_HASH_GATE=NOT_STARTED`
- `CHRONOLOGY_LEAKAGE_DATA_QUALITY_GATE=NOT_STARTED`
- `READY_FOR_EXPERIMENT_DESIGN_REVIEW=NO`
- `SCIENTIFIC_TRAINING_STARTED=NO`
- `REAL_RESULTS_GENERATED=NO`

Important later-stage blocker:
`scripts/t2/run_diagnostic_matrix.py` still requires the obsolete
`t2_diagnostic_input_archive_v2` bounded-API input archive. That path is not
authorized for the current frozen-raw study. Do not execute or adapt training
until harmonization and the subsequent chronology/leakage/data-quality and
experiment-design reviews define the frozen-harmonized training input contract.

Preserve Boston's 30-column authoritative CSV rule, LA 2025 partial coverage,
Chicago interrupted staging artifacts, and full provenance. PR #4 remains
draft/unmerged.


## Harmonization attempt 1 — Boston mixed-timestamp parser fix

The first frozen-raw harmonization attempt used code SHA
`64de331fde4554c9af1bdcd306acb903dcff590f`.

Observed derived state after failure:
- NYC 2021-2025 harmonized Parquet outputs completed.
- Chicago 2021-2025 harmonized Parquet outputs completed.
- `BOSTON_2021_harmonized.parquet.tmp` remained incomplete.
- No final harmonized manifest/checksum sidecars were written.
- Frozen raw files were not modified.
- The harmonization gate remains unpassed pending a clean complete rerun.

Root cause reproduced against frozen Boston 2021:
- raw size `156,065,195` bytes
- SHA256 `f22ab1e845dde506c685b8f27c265d9cd4ca276a08d0d116eaef85139777b3c2`
- 30 authoritative CSV columns
- null IDs `0`
- duplicate IDs `0`
- default pandas mixed-string timestamp inference falsely coerced
  `136,362 / 269,665` `open_dt` values to `NaT`
- `format="mixed"` yields 0 invalid created timestamps and the frozen
  Boston 2021 bounds.

Fix:
- frozen harmonizer created-date parsing uses `format="mixed"`
- harmonized audit created/closed parsing uses `format="mixed"`
- downstream resolved-frame created/closed parsing uses `format="mixed"`
- regression tests cover mixed Boston timestamp formats

Validated fix SHA:
`1d6e49fcbb91e1a99a46fd7788e304bb91f874d7`

GitHub Actions T2 Validation:
`35532001133` — PASS.

Recovery policy:
quarantine, do not reuse or silently delete, the failed-run derived artifacts;
keep the canonical `02_Harmonized` folder ID unchanged; regenerate all 20
outputs from the untouched frozen raw snapshots under the single validated fix
SHA; no training.


## Harmonization attempt 2 — LA 2021 exact-schema correction

Attempt 2 used timestamp-fixed execution SHA
`1d6e49fcbb91e1a99a46fd7788e304bb91f874d7`.

Observed state:
- NYC 2021-2025 harmonized Parquets completed.
- Chicago 2021-2025 harmonized Parquets completed.
- Boston 2021-2025 harmonized Parquets completed.
- 15 total completed outputs.
- Failure occurred before LA output creation.
- No final harmonized manifest, checksum file, or audit report was written.
- Frozen raw snapshots were not modified.
- `HARMONIZATION_GATE=NOT_PASSED`.

Failure:
`RuntimeError: LA_2021_raw.csv: column count mismatch 33 != 34`.

Independent resolution:
- original acquisition/validation Colab cell 69 explicitly reports
  `CSV_COLUMN_COUNT = 33`
- LA 2021 exact CSV header has 33 fields
- `CreatedByUserOrganization` is absent from LA 2021
- LA 2022-2024 each independently validate at 34 columns and include that field
- LA 2025 remains its separate 34-column schema
- LA 2021 raw object is unchanged:
  - Drive ID `145-M-RUq1MygJ0rROvtpUfdMyJbJb3Yc`
  - size `508,715,329` bytes
  - SHA256 `2713ab746cb1f1f11df6a93f4f5fb1307a4d309fb1812d30f2453e1faa3a9a87`

Corrections:
- central Drive raw manifest corrected to exact LA 2021 33-column schema
- repository frozen manifest uses dedicated `la_2021_v1`
- regression test enforces LA 2021 = 33 columns and LA 2022-2024 = 34
- no raw bytes changed

Validated execution SHA:
`f51669502048e2d511edeead31b1c75ed2f92400`

GitHub Actions T2 Validation:
`35533169219` — PASS.

Current canonical `02_Harmonized` state:
exactly 15 completed Parquets (NYC 5, Chicago 5, Boston 5), no LA outputs,
and no final harmonized manifest/checksum/audit sidecars.

Recovery policy:
quarantine the 15 attempt-2 derived files; preserve attempt-1 quarantine; keep
the canonical `02_Harmonized` folder unchanged; regenerate all 20 outputs
from untouched frozen raw snapshots under the single validated execution SHA
`f51669502048e2d511edeead31b1c75ed2f92400`; no training.


## Harmonization attempt 3 — complete gate transition

Validated execution SHA:
`f51669502048e2d511edeead31b1c75ed2f92400`.

Clean all-20 harmonization completed from untouched frozen raw snapshots.

Independent Drive/sidecar verification:
- 20 final Parquets in canonical `02_Harmonized`: NYC 5, Chicago 5, Boston 5, Los Angeles 5.
- `T2_HARMONIZED_MANIFEST.json` present.
- `T2_HARMONIZED_CHECKSUMS_SHA256.txt` present.
- `T2_HARMONIZED_AUDIT.json` present.
- execution manifest: protocol `t2_frozen_raw_harmonization_v1`, `git_sha=f51669502048e2d511edeead31b1c75ed2f92400`, `git_dirty=false`, `row_filtering=none`, 20 outputs.
- checksum sidecar: exactly 20 unique entries; exact filename/hash agreement with both manifest and audit JSON.
- structural audit: protocol `t2_harmonized_audit_v1`, gate PASS, 20 files, 0 blockers, 13 quantified warnings.

Gate transition:
- `HARMONIZATION_GATE=PASS`
- `HARMONIZED_HASH_GATE=PASS`
- `HARMONIZED_STRUCTURAL_DATA_AUDIT_GATE=PASS`
- `SCIENTIFIC_TRAINING_STARTED=NO`
- `REAL_RESULTS_GENERATED=NO`

Retained data-quality warnings for the next gate:
- Los Angeles 2021: 3,369 closed-before-created rows.
- Los Angeles 2022: 31 closed-before-created rows.
- Los Angeles 2025: 810,979 missing descriptor rows.
- NYC 2021: 11,988 closed-before-created rows; 91,040 missing descriptor rows.
- NYC 2022: 8,151 closed-before-created rows; 97,843 missing descriptor rows.
- NYC 2023: 4,313 closed-before-created rows; 97,868 missing descriptor rows.
- NYC 2024: 988 closed-before-created rows; 112,291 missing descriptor rows.
- NYC 2025: 916 closed-before-created rows; 84,294 missing descriptor rows.

Next stage:
`CHRONOLOGY_LEAKAGE_DATA_QUALITY_REVIEW`. Training remains forbidden until that gate and experiment-design review pass.


## Chronology/leakage gate pass and full-data execution blocker

Chronology/leakage/data-quality audit passed with zero blockers.

- report: `T2_CHRONOLOGY_LEAKAGE_DATA_QUALITY_AUDIT.json`
- harmonized execution SHA: `f51669502048e2d511edeead31b1c75ed2f92400`
- analysis-audit execution SHA: `cc917e1e3a1d2b71f295c7842de4950ea1eed268`
- source chronology: created 2021-2023 train, 2024 validation, 2025 internal test
- purge training outcomes crossing 2024 boundary and validation outcomes crossing 2025 boundary
- fixed preprocessing fit scope: none
- held-out target remains physically unopened through source-only training/checkpoint selection
- descriptor is excluded from primary predictors because LA maps it from post-service `ActionTaken`
- explicit analysis-frame exclusions only: nonfinal status, missing/invalid close time, negative duration
- harmonized data are never overwritten by these exclusions

Manual status semantics:
`PASS_WITH_INTERPRETATION`.

Use administrative completion/closure time, not guaranteed physical service-fulfillment time.
Recognized completion states:
- Boston: `Closed`
- NYC: `Closed`
- Chicago: `Completed` plus the rare `Closed`
- Los Angeles: `Closed`

Do not treat canceled/cancelled, referred-out, duplicate-administrative, pending/open/in-progress states as resolved observations.

### Experiment-design blocker before training

The current final-study execution path is not authorized.

`src/t2/cli.py` is still legacy Boston/LA-only and fail-closed, and
`src/t2/federated.py` materializes full dense `N x 583` float32 feature
tensors for every source client.

Audited post-purge training rows imply feature memory alone:

| city | rows | dense N×583 float32 |
|---|---:|---:|
| Boston | 783,095 | ~1.83 GB |
| Chicago | 5,223,833 | ~12.18 GB |
| Los Angeles | 3,887,364 | ~9.07 GB |
| NYC | 9,191,768 | ~21.44 GB |

Three-source fold feature tensors alone therefore require about 23.07-42.68 GB,
before pandas, labels, loaders, Opacus per-sample gradients, model states, and
device copies.

Required before training:
1. batch-wise/streamed deterministic hashing or an equivalent compact file-backed representation;
2. full four-fold LOCO runner;
3. physical delayed held-out loading;
4. continuous per-client Opacus accounting across all private optimizer steps;
5. provenance-complete full-data input contract;
6. CI and execution-feasibility preflight.

`SCIENTIFIC_TRAINING_STARTED=NO`
`REAL_RESULTS_GENERATED=NO`


## Validated scalable four-fold full-data execution redesign

Validated scientific redesign SHA:
`7dd875d06ce03b56a63e3c70c706cae453b0770b`

GitHub Actions run:
`35543297170` — SUCCESS.

The validated full-data path now contains:

- compact batch-wise feature construction numerically equivalent to the frozen
  583-dimensional fixed representation;
- no persistent full `N x 583` source matrix;
- one-city-year-at-a-time harmonized loading with size/SHA verification;
- source train = eligible 2021-2023 rows with close/completion before 2024;
- source validation = eligible 2024 rows with close/completion before 2025;
- source internal test = eligible 2025 rows;
- held-out primary external test = eligible held-out-city 2025 rows;
- physical target-byte loading only after source-only checkpoint selection;
- four rotating LOCO folds;
- primary matrix of 36 runs:
  4 folds x 3 seeds x {nonprivate infinity, private epsilon 5, private epsilon 1};
- 12 clipped/no-noise mechanism-ablation runs, 48 total including controls;
- fixed-total-effective-epoch source training budget with deterministic planned
  optimizer steps for DP calibration;
- administrative closure/completion estimand semantics;
- PTP aggregation guard requiring matched external evaluation scope.

The full-data CLI remains deliberately fail-closed:
`FULL_LOCO_EXECUTION_ENABLED = False`.

Next required gate:
run `scripts/t2/audit_full_design.py` on the real 20 harmonized Parquets.
The audit must verify exact post-purge counts against the chronology audit,
compact resident memory, zero persistent dense full-feature rows, DP step/delta
schedule, all four folds, 36 primary/48 total conditions, and held-out
isolation.

`SCIENTIFIC_TRAINING_STARTED=NO`
`REAL_RESULTS_GENERATED=NO`


## Real-data experiment-design feasibility gate PASS

Saved Drive report:
`T2_FULLDATA_EXPERIMENT_DESIGN_AUDIT.json`

Drive ID:
`1m-gxAVrC2cHNJHBQONcHMjGCHHfO4-mX`

Report SHA256:
`ebfdc18dbfeebd9d760868cae6e438f1e138cc4356e5093d3cf4b1cf9fddc628`

Independent verification:
- protocol `t2_full_data_experiment_design_audit_v1`
- gate `PASS`
- blockers `0`
- audit execution SHA `7dd875d06ce03b56a63e3c70c706cae453b0770b`
- harmonized execution SHA `f51669502048e2d511edeead31b1c75ed2f92400`
- analysis-audit SHA `cc917e1e3a1d2b71f295c7842de4950ea1eed268`
- model input dimension 583
- 36 primary runs, 12 clipped/no-noise ablations, 48 total
- all four held-out cities and seeds 0/1/2
- held-out target 2025 loaded only after source checkpoint selection
- target excluded from tuning and privacy accounting
- continuous per-client Opacus RDP accounting
- delta strictly below 1/N
- fixed-total-effective-epoch budget, target 1.0 effective epoch over 20 rounds
- logical batch size 1024

Real compact memory:
- NYC: ~0.5995 GiB all splits
- Chicago: ~0.3328 GiB
- Boston: ~0.0465 GiB
- Los Angeles: ~0.2477 GiB
- maximum three-source fold: ~1.1800 GiB
- persistent dense full-feature rows: 0

Gate transition:
`EXPERIMENT_DESIGN_GATE=PASS`
`FULL_LOCO_EXECUTION_UNLOCK_READY=YES`

The training launcher is being unlocked only behind exact verification of the
saved report hash and its provenance/gate fields. Training remains off until
the unlock commit passes CI.

`SCIENTIFIC_TRAINING_STARTED=NO`
`REAL_RESULTS_GENERATED=NO`


## Scientific execution checkpoint — 2026-09-21

Authoritative execution-state snapshot:
`docs/T2_SCIENTIFIC_EXECUTION_CHECKPOINT_2026-09-21.md`

Checkpoint commit:
`433a1be674c876aff559c02383b5257e4f898db9`

Drive mirror:
`T2-GovTrust — Scientific Execution Checkpoint — 2026-09-21`
Document ID: `11MknM21eziIM26LDX8t-MB056lMKU61CueZp-dVDtvw`

State frozen in that checkpoint:
- accepted NYC seed0 nonprivate run
- accepted NYC seed0 private epsilon=5 run
- accepted NYC seed0 private epsilon=1 retry
- failed pre-training epsilon=1 import attempt preserved
- exact run UUIDs, checkpoint hashes, DP accountant values, metrics, privacy-cost/PTP values
- next authorized step: NYC seed0 clipped_no_noise only
- no matrix-wide expansion yet
