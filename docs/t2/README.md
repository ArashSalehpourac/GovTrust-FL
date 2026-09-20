# T2 diagnostic validity gate

## Current authorized data stage — frozen 20/20 archive

As of 2026-09-20, `FINAL_RAW_FILES_ACCEPTED=20/20` and `RAW_DATA_GATE=PASS`.
The current harmonization stage is authorized to read **only** the frozen files in
Google Drive `Real Datasets — 2021-2025/01_Raw_Official`. It must not re-fetch
municipal APIs, select bounded samples, or substitute historical Step-3 cleaned data.

The frozen registry is committed at `configs/t2/frozen_raw_manifest.json` and is
also mirrored in the Drive spreadsheet `T2_RAW_MANIFEST_CHECKSUMS_20_FINAL_2026-09-20`.
The harmonizer verifies every raw file's byte size and SHA256 before reading it,
preserves every row, enforces the Boston 30-column CSV rule (no datastore-only
`_id`), and preserves LA 2025's official partial coverage.

From the existing Colab Drive mount, the authorized command is:

```bash
python scripts/t2/harmonize_frozen_raw.py \
  --raw-dir "/content/drive/MyDrive/T2-GovTrust/Real Datasets — 2021-2025/01_Raw_Official" \
  --output-dir "/content/drive/MyDrive/T2-GovTrust/Real Datasets — 2021-2025/02_Harmonized"
```

The command writes one Parquet artifact per city-year plus
`T2_HARMONIZED_MANIFEST.json` and `T2_HARMONIZED_CHECKSUMS_SHA256.txt`.
Existing harmonized outputs are never overwritten.

**Do not run training yet.** Harmonized hashes, chronology/leakage/data-quality
gates, and the experiment-design review must pass first.

`scripts/t2/prepare_inputs.py` is retained only for the earlier bounded-API
diagnostic design. Its live-API fetch path is **not authorized for the current
harmonization stage**.


This package is an additive redesign of GovTrust-FL. Historical Step 1-19 code is not modified.

## Scientific question

How does formal example-level differential privacy affect predictive reliability when a federated municipal service-request model is transferred to an unseen administrative jurisdiction?

## Frozen primary endpoint

- Target: `log1p(resolution_hours)` point regression.
- Loss: SmoothL1.
- Primary error: MAE on `log1p(hours)`.
- Secondary descriptive errors: RMSE on log scale, MAE/median AE/p90 AE in hours.
- Source performance is the unweighted macro-average across the three source-city internal-test losses.
- Privacy Transfer Penalty (PTP) = external privacy cost - internal privacy cost.

## Input boundary and leakage controls

The first diagnostic uses a dedicated bounded public-data fetch and harmonizes directly to the Step-2 schema. Historical Step-3 cleaned files are forbidden because Step 3 applies whole-city category-frequency filtering and coordinate-based row removal before chronological splitting.

`scripts/t2/prepare_inputs.py` creates exactly 20,000 requests per city: the first 4,000 unique request IDs in chronological order from each year 2021-2025, after a deterministic bounded oversample from the official municipal APIs. Selection uses only `request_id`, `created_date`, and the asserted city identifier; it does not inspect status, `closed_date`, resolution time, or another outcome. The archive manifest stores the derived SHA256 for every city.

Los Angeles changed its MyLA311 source/schema in 2025. Years 2021-2024 use the legacy Service Request datasets; 2025 uses the official `73a2-6ar5` MyLA311 Cases source and maps `casenumber`, `type`, `action_taken__c`, and `department_name__c` into the common T2 schema. That official source begins in March 2025, so the 2025 LA diagnostic sample is drawn from the available March-December source period; the coverage note is preserved in the input manifest.

Each LOCO fold has three source cities and one held-out city. Source cities are split chronologically. The primary representation is completely data-independent: descriptor text uses fixed-dimensional feature hashing, category uses a separate fixed-dimensional feature hasher, and calendar variables use deterministic cyclical encoding. No vocabulary, category level, imputer, scaler, or other statistic is fit on municipal records. The primary feature set explicitly excludes raw city ID, agency, latitude/longitude, coordinate grids, area, and ZIP code.

The executable runner invokes the held-out data loader only after source-only checkpoint selection finishes.

## Privacy scope

Private clients use Opacus example-level DP-SGD with Poisson subsampling, per-sample clipping, Gaussian noise, and a persistent RDP accountant across every private optimizer step and FL round. Each source client receives its own delta via `min(1e-5, 0.1/N)`, strictly below `1/N`. The fold-level guarantee is reported conservatively as the maximum realized epsilon across disjoint source clients. A target epsilon is never declared achieved unless the executed accountant falls within the configured tolerance.

The protected unit is **one source-city training service-request row**. Validation, internal-test, and held-out evaluation rows are not included in the epsilon-delta DP guarantee. Opacus `secure_mode` is disabled for the research experiment and is recorded; this work does not claim production cryptographic hardening against implementation-level RNG attacks.

The matched `clipped_no_noise` control uses the same Opacus clipping/sampling path with zero noise and is explicitly non-private.

## Diagnostic matrix only

The first authorized compute gate is limited to held-out Boston and Los Angeles, FedAvg, seeds 0/1/2, epsilon infinity/5/1, plus the matched clipped-no-noise control: 24 jobs total. The round cap is 20. Source-only checkpoint selection is evaluated every round. The default diagnostic does not terminate on patience because fixed planned private steps are needed for auditable epsilon calibration; `would_stop_round` records the convergence diagnostic.

The matrix runner is resumable only when the executable Git SHA, exact config hash, and all four input SHA256 values match a previously completed run. The full four-city/full-epsilon matrix remains blocked until this diagnostic is interpreted.

## Reproducible Colab execution

From a clean checkout of the pinned execution commit, create the bounded input archive in mounted Google Drive:

```bash
python scripts/t2/prepare_inputs.py --archive-dir /content/drive/MyDrive/T2-GovTrust/T2_Diagnostic_Inputs
```

Inspect the matrix without training:

```bash
python scripts/t2/run_diagnostic_matrix.py --input-dir /content/drive/MyDrive/T2-GovTrust/T2_Diagnostic_Inputs --output-dir /content/drive/MyDrive/T2-GovTrust/T2_Diagnostic_Results --device auto
```

Execute/resume the authorized 24-job diagnostic only after the validation gate is green:

```bash
python scripts/t2/run_diagnostic_matrix.py --input-dir /content/drive/MyDrive/T2-GovTrust/T2_Diagnostic_Inputs --output-dir /content/drive/MyDrive/T2-GovTrust/T2_Diagnostic_Results --device auto --execute
```

Keep inputs and results on Google Drive so runtime disconnects do not destroy completed provenance-valid jobs. Keep the repository as a clean detached checkout of the pinned execution SHA.

## Resolved-only limitation

The study conditions on requests with an observed completion time. Missing/open/unresolved requests are excluded after outcome-independent archive selection. Results therefore characterize completed requests and must not be generalized to all incoming/open requests without a censoring/open-case sensitivity analysis.

## FedProx

FedProx is deliberately excluded from inferential T2 runs. It may be reintroduced only after separate implementation and behavioral validation.

## Authenticity

Real result aggregation requires a completed manifest with clean Git SHA, config hash, input hashes, split metadata, fixed-preprocessor fingerprint, training configuration, privacy trace, runtime/hardware/package metadata, command, checkpoint hash, and output inventory. Incomplete or dirty-worktree runs are rejected.
