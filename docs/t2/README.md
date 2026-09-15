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

## Leakage controls

Each LOCO fold has three source cities and one held-out city. Source cities are chronologically split before any preprocessing fit. TF-IDF, category levels, numeric scaling, training, checkpoint selection, and all model decisions use source-city training/validation data only. The primary feature set explicitly excludes raw city ID, agency, latitude/longitude, coordinate grids, area, and ZIP code. Unknown target categories are handled as OOV/ignored values rather than refitting.

The executable runner is designed so the held-out data loader is invoked only after source-only checkpoint selection finishes. Do not replace this with eager loading of all four cities.

## Privacy

Private clients use Opacus example-level DP-SGD with Poisson subsampling, per-sample clipping, Gaussian noise, and a persistent RDP accountant across every optimizer step and FL round. Each source client receives its own delta via `min(1e-5, 0.1/N)`, which is strictly less than `1/N`. The fold-level guarantee is reported conservatively as the maximum realized epsilon across disjoint source clients. A target epsilon is never declared achieved unless the executed accountant falls within the configured tolerance.

The matched `clipped_no_noise` control uses the same Opacus clipping and sampling path with zero noise and is explicitly non-private.

## Diagnostic matrix only

The first authorized compute gate is limited to held-out Boston and Los Angeles, FedAvg, seeds 0/1/2, epsilon infinity/5/1, plus the matched clipped-no-noise control. The round cap is 20. Source-only checkpoint selection is evaluated every round. The default diagnostic does not terminate on patience because fixed planned private steps are needed for auditable epsilon calibration; `would_stop_round` records the convergence diagnostic.

The four-city/full-epsilon matrix is blocked until this gate passes.

## Resolved-only limitation

The current study conditions on requests with an observed completion time. Missing/open/unresolved requests are excluded. Results therefore characterize completed requests and must not be generalized to all incoming/open requests without a censoring or open-case sensitivity analysis.

## FedProx

FedProx is deliberately excluded from inferential T2 runs. The earlier prototype applied a post-optimizer shrinkage rather than a canonical proximal objective and empirically behaved almost identically to FedAvg. It may be reintroduced only after a separate implementation/behavioral validation.

## Authenticity

Real result aggregation requires a completed manifest with clean Git SHA, config hash, input hashes, split metadata, preprocessing fingerprint, training configuration, privacy trace, runtime/hardware/package metadata, command, checkpoint hash, and output inventory. Incomplete or dirty-worktree runs are rejected.
