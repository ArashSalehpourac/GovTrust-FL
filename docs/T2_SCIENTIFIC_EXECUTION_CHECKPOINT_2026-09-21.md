# T2-GovTrust — Scientific Execution Checkpoint — 2026-09-21

## Frozen paper identity

**Privacy Under Institutional Shift: Cross-Jurisdiction Reliability of Federated Learning for Municipal Service-Request Resolution**

Primary estimand interpretation remains **administrative closure/completion duration conditional on observed completion**, not guaranteed physical service fulfillment.

## Validated scientific execution contract

- Scientific execution SHA: `998a0e17b03986a7adf2edffec61d1e7e5618d96`
- Experiment-design audit SHA: `7dd875d06ce03b56a63e3c70c706cae453b0770b`
- Harmonization SHA: `f51669502048e2d511edeead31b1c75ed2f92400`
- Chronology/leakage audit SHA: `cc917e1e3a1d2b71f295c7842de4950ea1eed268`
- Design report SHA256: `ebfdc18dbfeebd9d760868cae6e438f1e138cc4356e5093d3cf4b1cf9fddc628`
- Preprocessor fingerprint: `ab286be51a209e1e2cb91913cc40adae184a4c270b7491fff37eedffc6246dac`
- Model input dimension: 583
- Rounds: 20
- Logical batch size: 1024
- Training budget: fixed total effective epochs
- Target total effective epochs: 1.0
- External primary evaluation: held-out city, year 2025 only
- Matrix: 36 primary runs + 12 clipped/no-noise ablations = 48 total

## Gate state

- HARMONIZATION_GATE=PASS
- CHRONOLOGY_LEAKAGE_DATA_QUALITY_GATE=PASS
- STATUS_SEMANTICS_GATE=PASS_WITH_INTERPRETATION
- EXPERIMENT_DESIGN_GATE=PASS
- FULL_LOCO_EXECUTION_UNLOCK=PASS
- FIRST_NONPRIVATE_BASELINE_GATE=PASS
- PRIVATE_EPS5_RUN_GATE=PASS
- PRIVATE_EPS1_RUN_GATE=PASS
- PRIVACY_ACCOUNTING_GATE=PASS

Scientific training has started and real results have been generated.

## Accepted run 1 — NYC held-out, seed 0, nonprivate

Run UUID: `fdea59ec-7888-4ff3-bd9e-cead627f8034`

- mode: nonprivate
- epsilon: infinity
- best round: 20
- source validation best macro MAE log1p-hours: 1.23305896315119
- source internal 2025 macro MAE log1p-hours: 1.6128832418483479
- NYC 2025 external MAE log1p-hours: 2.293948561574594
- NYC 2025 external RMSE log1p-hours: 2.6567688126124005
- NYC 2025 MAE hours: 271.8240779241973
- NYC 2025 median AE hours: 57.221864955542486
- NYC 2025 p90 AE hours: 512.3466671144106
- checkpoint SHA256: `1da59f5dcc3a5b60757626c3c6072bf86cc237a57adb96f48dd34613dabe1e76`

A post-run reporting-only KeyError requested the nonexistent key `median_ae_log1p_hours`; the scientific run had already completed and all artifacts were durably saved. Do not rerun this baseline.

## Accepted run 2 — NYC held-out, seed 0, private epsilon 5

Run UUID: `e7683b11-6962-44b8-a6d3-55eb164c6247`

- best round: 20
- source validation best macro MAE log1p-hours: 1.3128068883834991
- source internal 2025 macro MAE log1p-hours: 1.6762134958338093
- NYC 2025 external MAE log1p-hours: 2.233129242714567
- NYC 2025 external RMSE log1p-hours: 2.661424904811585
- NYC 2025 MAE hours: 259.3359973184775
- NYC 2025 median AE hours: 55.528273269359126
- NYC 2025 p90 AE hours: 327.90844779130157
- checkpoint SHA256: `5195e85383e177f6b8944e9f235ae561ea714a947c3dca5c3e0ec8adacb7f0e1`

Per-client DP accounting:
- Boston: epsilon 4.995833410161681; noise 0.54901123046875; planned=executed=accounted steps 780
- Chicago: epsilon 4.9937346273891725; noise 0.50262451171875; planned=executed=accounted steps 5120
- Los Angeles: epsilon 4.997480635399884; noise 0.5084228515625; planned=executed=accounted steps 3800
- Fold epsilon = max client epsilon = 4.997480635399884
- All delta values are strictly below 1/N.
- `secure_rng=false` is explicitly documented as a research-experiment setting, not a production cryptographic deployment claim.

Matched against nonprivate:
- external privacy cost: -0.06081931886002678
- internal privacy cost: +0.06333025398546144
- PTP: -0.12414957284548822

Single-fold/single-seed descriptive only.

## Accepted run 3 — NYC held-out, seed 0, private epsilon 1

Initial attempt failed **before training** because Opacus was not installed. Failed log is preserved as:
`REAL_RUN_NYC_SEED0_PRIVATE_EPS1.log`

Successful retry run UUID:
`28c5abb4-067e-4f46-b336-c2c358c5c266`

Retry environment:
- Opacus 1.6.0
- torch 2.11.0+cu128
- CUDA active on NVIDIA A100-SXM4-80GB
- exact design-audit hash matched
- exact execution SHA matched

Performance:
- best round: 20
- source validation best macro MAE log1p-hours: 1.3126809538076287
- source internal 2025 macro MAE log1p-hours: 1.6761096752426523
- NYC 2025 external MAE log1p-hours: 2.2325713831217557
- NYC 2025 external RMSE log1p-hours: 2.66056383764581
- NYC 2025 MAE hours: 259.2304880114976
- NYC 2025 median AE hours: 55.50574312072261
- NYC 2025 p90 AE hours: 327.9191945195259
- checkpoint SHA256: `10880de0685456549834288418d8005bad9b196bc16718603628a791dc5f5cec`

Per-client DP accounting:
- Boston: epsilon 0.9966398141937871; noise 1.031494140625; planned=executed=accounted steps 780
- Chicago: epsilon 0.995837560421344; noise 0.9686279296875; planned=executed=accounted steps 5120
- Los Angeles: epsilon 0.9931173023325692; noise 0.966796875; planned=executed=accounted steps 3800
- Fold epsilon = max client epsilon = 0.9966398141937871
- All target-within-tolerance checks passed.
- All delta values are strictly below 1/N.

Matched against nonprivate:
- external privacy cost: -0.06137717845283808
- internal privacy cost: +0.06322643339430445
- PTP: -0.12460361184714253

Single-fold/single-seed descriptive only.

## Current Drive execution folder

`04_Full_LOCO_Runs`
Drive folder ID: `1wucXTCWdDwx2pyPHm9kXOmTT2sLkaTdP`

Accepted run directories:
- `fdea59ec-7888-4ff3-bd9e-cead627f8034`
- `e7683b11-6962-44b8-a6d3-55eb164c6247`
- `28c5abb4-067e-4f46-b336-c2c358c5c266`

Preserved logs:
- `FIRST_REAL_RUN_NYC_SEED0_NONPRIVATE.log`
- `REAL_RUN_NYC_SEED0_PRIVATE_EPS5.log`
- `REAL_RUN_NYC_SEED0_PRIVATE_EPS1.log` — failed before training
- `REAL_RUN_NYC_SEED0_PRIVATE_EPS1_RETRY1.log` — successful retry

## Next authorized scientific step

Run **NYC / seed 0 / clipped_no_noise** only.

Purpose: separate clipping + Poisson-sampling effects from added DP noise before any matrix-wide expansion.

Do not yet launch the remaining seeds/cities until this control is completed and independently audited.
