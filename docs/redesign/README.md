# T2 redesign: cross-jurisdiction predictive reliability under formal DP

This package (`src/redesign`, `scripts/redesign`, `tests/redesign`) is **additive**.
The historical Step 1–19 pipeline in `src/` and `scripts/` and all existing outputs
under `data/` and `results/` are untouched and still run as before.

## 1. Scientific question

> How does *formally quantified* example-level differential privacy alter the
> **cross-jurisdiction predictive reliability** of federated models for municipal
> service-request resolution, under institutional shift between city
> administrative systems?

The contribution is not FL, DP, calibration, or domain generalization in isolation:
it is the leakage-safe empirical characterization of the privacy × jurisdictional-shift
interaction, with internal (source-city) and external (zero-shot held-out-city)
reliability reported separately, at matched, audited privacy budgets.

Explicitly out of scope of the main pipeline: TAI-Score, PEDI, simulated
secure-aggregation overhead, the service-routing task, demographic fairness claims,
SHAP/counterfactual analysis, resource metrics, membership inference, and the
Algorithmic Transparency Record. Historical implementations of those remain on disk
but are not part of this contribution.

## 2. Fold protocol

Four leave-one-city-out folds over `nyc`, `chicago`, `boston`, `los_angeles`
(`src/redesign/config.py::build_folds`). For each fold:

| Role | Cities | Use |
| --- | --- | --- |
| Source | the 3 remaining cities | chronological 70 / 15 / 15 train / val / internal-test; all fitting and model selection |
| Held out | 1 city | zero-shot external evaluation only |

Operation order in `src/redesign/folds.py::build_fold` — this order *is* the leakage control:

1. cap rows per city (most recent rows, chronological order preserved);
2. build creation-time features only (`src/redesign/features.py`; trailing 7-day
   request counts are strictly backward-looking and exclude the current row);
3. split each **source** city chronologically (no shuffling, no target-aware logic);
4. fit the target policy on pooled **source-city training rows** only;
5. fit TF-IDF, categorical imputer/one-hot, numeric imputer/scaler on the **same**
   pooled source-city training rows;
6. transform everything else — source val, source internal test, and the held-out
   city — with those already-fitted artifacts.

## 3. Targets

Primary (regression / quantile): `y_log1p_resolution_hours = log1p(resolution_hours)`,
winsorized at the **train-fitted** 99.5th percentile.

Secondary (binary, explicitly policy-defined and train-only):
`y_sla_breach = resolution_hours > sla_threshold_hours`, where per-category thresholds
are the 75th percentile of resolution hours **within source-city training rows**
(categories with fewer than 50 training rows fall back to the global train threshold).

This replaces the historical `delayed` target, whose city/category Q3 thresholds were
fitted over the full cleaned data of every city including the external city.

## 4. Leakage controls and the proof they hold

Every fold writes `data/redesign/folds/<fold>/fold_manifest.json` containing the fit
scope (cities, split, row count, max `created_date`), the preprocessing fingerprint,
a SHA-256 digest of the source-train request IDs, the target policy, per-split
summaries, and isolation checks (cities present per split, source/external
`request_id` overlap).

`tests/redesign` enforces the controls:

- `test_four_leave_one_city_out_folds` — all four folds, no city in its own source set;
- `test_source_splits_are_chronological_and_disjoint`, `test_chronological_split_does_not_shuffle`;
- `test_held_out_city_absent_from_every_fitting_frame`;
- `test_poisoning_held_out_city_changes_no_fitted_artifact` — the decisive test:
  corrupting **every** held-out-city row must leave the preprocessor fingerprint,
  the target policy, the source-train ID digest, and all source-split targets
  bit-identical;
- `test_preprocessor_vocabulary_excludes_held_out_tokens` — poisoned tokens/levels
  never enter the TF-IDF vocabulary or one-hot categories;
- `test_target_thresholds_are_train_only`;
- `test_training_and_selection_never_touch_the_held_out_city` — poisoning the held-out
  city leaves model selection and every internal metric unchanged, while (as expected)
  the external metric moves;
- `test_dp_accounting_matches_the_training_mechanism` — epsilon is recomputed
  independently from the reported mechanism parameters.

Model selection uses the pooled **source-city validation** mean pinball loss only.

## 5. Privacy definition

Formal, example-level DP-SGD via Opacus `PrivacyEngine` (`src/redesign/dp.py`),
replacing the historical update-level Gaussian noise approximation (which remains
untouched in `src/fl_pytorch.py` and is **not** used here).

- mechanism: per-example gradient clipping to `max_grad_norm` + Gaussian noise,
  Poisson subsampling via Opacus `DPDataLoader`;
- accountant: Opacus RDP; the noise multiplier is calibrated once for the *whole* run
  (all local steps across all rounds) via `get_noise_multiplier`;
- unit of privacy: **one service request row within its own city's dataset**;
- guarantee: each city's reported epsilon bounds the influence of one of its own rows
  on every message that city releases across all rounds.

Each run reports, per city: mechanism, accountant, target epsilon, epsilon spent,
delta, sample rate, expected batch size, dataset size, clip norm, noise multiplier,
planned steps, steps taken, accounted steps, and the full accountant history.

The FedProx proximal term is applied as a data-independent post-step update using only
the global parameters, so it stays outside the clipped/noised gradient path and costs
no privacy budget.

## 6. Pilot commands

```bash
# 1. capped official-source extract (20k rows/city, writes only under data/redesign/)
python scripts/redesign/fetch_pilot_data.py --rows-per-city 20000

# 2. build the four leave-one-city-out folds + manifests
python scripts/redesign/build_folds.py --max-rows-per-city 20000

# 3. leakage / DP-accounting tests
python -m pytest tests/redesign -q

# 4. small pilot: 4 folds x {fedavg, fedprox} x eps {inf, 5, 1} x 3 seeds, 5 rounds
python scripts/redesign/run_pilot.py --rounds 5 --seeds 0 1 2 --epsilons inf 5 1
```

## 7. Outputs

| Path | Content |
| --- | --- |
| `data/redesign/clean_pilot/pilot_extract_manifest.json` | source APIs, caps, date window, per-city row counts |
| `data/redesign/folds/<fold>/fold_manifest.json` | fit scope, fingerprints, isolation checks, target policy |
| `data/redesign/folds/<fold>/{preprocessor.joblib,target_policy.json}` | fold-specific fitted artifacts |
| `data/redesign/folds/folds_summary.json` | per-fold row counts, feature dims, fingerprints |
| `results/redesign/runs/<run>.json` | full per-run record: config, privacy, selection, history, internal + external metrics, subgroups |
| `results/redesign/pilot_summary.csv` | one row per run, internal vs external metrics side by side |
| `results/redesign/pilot_subgroup_reliability.csv` | operational subgroup reliability on the held-out city |

Reliability outputs: primary — MAE/RMSE (log1p and hours), bias, per-quantile pinball
loss, per-quantile coverage, 10–90 interval nominal vs empirical coverage and gap,
interval width; secondary — Brier, ECE, AUROC, AUPRC, positive rate, mean predicted.

Subgroup summaries are computed on the held-out city by **service category**,
**area group**, and **request-volume stratum**. These are operational strata of the
administrative system, **not** demographic groups, and must not be reported as
demographic fairness metrics.

## 8. Known limitations

- The pilot is deliberately small: 20k rows/city, 5 rounds, 1 local epoch, 3 seeds.
  Absolute reliability numbers are not publication-grade; the privacy × shift
  *contrast* is what the pilot is designed to expose.
- Epsilon is per city, record-level, over that city's released updates. There is no
  central/server-side or cross-city composition claim, and no secure aggregation:
  the server sees each city's noised update in the clear.
- DP-SGD noise is calibrated once for the planned number of local steps; if a run is
  shortened, the realized epsilon (recomputed from the accountant history) is lower
  than the target rather than higher.
- Each city keeps a persistent local optimizer/privacy-engine state across rounds
  while parameters are re-synced to the global model each round.
- Cities are only partially comparable: schemas are harmonized to a common set of
  fields, but category taxonomies, closure conventions, and reporting channels differ
  by jurisdiction, which is part of the shift being measured, not noise to be removed.
- Boston's capped extract yields fewer usable rows than the other cities after
  cleaning, so `holdout_boston` has a smaller external set.
- Resolution-hours definitions rely on each city's `closed_date`; administrative
  closure is a proxy for actual service completion.
- At the pilot's `proximal_mu = 0.01`, FedProx is numerically almost indistinguishable
  from FedAvg; separating the two requires a mu sweep that the pilot does not run.
- No hyperparameter search is performed; architecture and learning rate are fixed
  across folds and privacy levels so the comparison stays matched.
