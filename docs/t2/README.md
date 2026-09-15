# T2 diagnostic validity gate

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

T2 consumes **Step-2 harmonized files**, not historical Step-3 cleaned files. Step 3 is forbidden for T2 because it performs whole-city category-frequency filtering and coordinate-based row removal before chronological splitting. T2 owns its resolved-only cleaning and request-ID deduplication.

Each LOCO fold has three source cities and one held-out city. Source cities are split chronologically. The primary representation is completely data-independent: descriptor text uses fixed-dimensional feature hashing, category uses a separate fixed-dimensional feature hasher, and calendar variables use deterministic cyclical encoding. No vocabulary, category level, imputer, scaler, or other statistic is fit on municipal records. The primary feature set explicitly excludes raw city ID, agency, latitude/longitude, coordinate grids, area, and ZIP code.

The executable runner is designed so the held-out data loader is invoked only after source-only checkpoint selection finishes. Do not replace this with eager loading of all four cities.

## Privacy scope

Private clients use Opacus example-level DP-SGD with Poisson subsampling, per-sample clipping, Gaussian noise, and a persistent RDP accountant across every private optimizer step and FL round. Each source client receives its own delta via `min(1e-5, 0.1/N)`, which is strictly less than `1/N`. The fold-level guarantee is reported conservatively as the maximum realized epsilon across disjoint source clients. A target epsilon is never declared achieved unless the executed accountant falls within the configured tolerance.

The protected unit is **one source-city training service-request row**. Validation, internal-test, and held-out evaluation rows are not included in the epsilon-delta DP guarantee. This scope must be stated exactly in the manuscript. The data-independent representation is necessary so preprocessing does not create an unprotected record-dependent side channel.

Opacus `secure_mode` is disabled for the research experiment and is recorded in every privacy report. The accountant and mechanism are therefore auditable research DP, but this work does not claim production cryptographic hardening against implementation-level RNG attacks.

The matched `clipped_no_noise` control uses the same Opacus clipping and sampling path with zero noise and is explicitly non-private.

## Diagnostic matrix only

The first authorized compute gate is limited to held-out Boston and Los Angeles, FedAvg, seeds 0/1/2, epsilon infinity/5/1, plus the matched clipped-no-noise control. The round cap is 20. Source-only checkpoint selection is evaluated every round. The default diagnostic does not terminate on patience because fixed planned private steps are needed for auditable epsilon calibration; `would_stop_round` records the convergence diagnostic.

The four-city/full-epsilon matrix is blocked until this gate passes.

## Resolved-only limitation

The current study conditions on requests with an observed completion time. Missing/open/unresolved requests are excluded. Results therefore characterize completed requests and must not be generalized to all incoming/open requests without a censoring or open-case sensitivity analysis.

## FedProx

FedProx is deliberately excluded from inferential T2 runs. The earlier prototype applied a post-optimizer shrinkage rather than a canonical proximal objective and empirically behaved almost identically to FedAvg. It may be reintroduced only after a separate implementation/behavioral validation.

## Authenticity

Real result aggregation requires a completed manifest with clean Git SHA, config hash, input hashes, split metadata, fixed-preprocessor fingerprint, training configuration, privacy trace, runtime/hardware/package metadata, command, checkpoint hash, and output inventory. Incomplete or dirty-worktree runs are rejected.
