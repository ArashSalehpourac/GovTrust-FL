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
