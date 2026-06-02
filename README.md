# GovTrust-FL

GovTrust-FL is a reproducible research codebase for trustworthy federated learning in multi-city municipal service-request triage. The core task is delayed-resolution risk prediction using NYC, Chicago, and Boston as federated clients, with Los Angeles held out for external validation.

This repository is paper-aligned, but it does not include real municipal datasets. Real 311/MyLA311 files must be downloaded by the user from official open-data portals and placed under `data/raw/<city>/`. Synthetic data are provided only for smoke tests and CI.

The manuscript label definition is city-category specific:

```text
delayed = 1 if resolution_hours > Q3(resolution_hours | city, category) else 0
```

Do not use Los Angeles for training, validation, preprocessing fit, calibration fitting, model selection, threshold tuning, or score normalization.

## Repository Structure

- `configs/default.yaml`: experiment settings for cities, targets, models, FL, DP, fairness, calibration, XAI, and outputs.
- `src/`: reusable data, model, federated learning, privacy, XAI, fairness, calibration, and scorecard modules.
- `scripts/`: command-line entry points for data prep, training, privacy attack, XAI/PEDI, trustworthiness evaluation, and transparency records.
- `docs/`: schema, reproducibility, experimental protocol, model-card, and transparency-record documentation.
- `tests/`: synthetic-data unit tests for preprocessing, metrics, PEDI, and TAI-Score.
- `data/`: ignored local raw, processed, and split datasets.
- `results/`: ignored local tables, figures, logs, and trained models.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Graphviz system binaries are needed only for rendering DOT figures:

```powershell
pip install graphviz
```

## Synthetic Smoke Workflow

Use synthetic data when real city exports are unavailable:

```powershell
python scripts/make_synthetic_data.py --rows-per-city 100
python scripts/prepare_data.py
python scripts/run_baselines.py
python scripts/run_federated.py --algorithm fedavg
python scripts/run_privacy_attack.py
python scripts/run_xai_pedi.py
python scripts/run_trustworthiness_eval.py
python scripts/make_transparency_record.py
```

To run the complete synthetic validation pipeline in an isolated temporary workspace:

```powershell
python scripts/run_full_pipeline.py
```

This command prints the temporary workspace path. Its outputs validate code behavior only and are not manuscript findings.

The numbered manuscript workflow wrappers are also available:

```powershell
python scripts/01_download_data.py
python scripts/02_harmonize_schema.py
python scripts/03_clean_data.py
python scripts/04_create_labels.py
python scripts/05_engineer_features.py
python scripts/06_create_splits.py
python scripts/07_train_baselines.py
python scripts/08_train_federated.py
python scripts/09_run_privacy_evaluation.py
python scripts/10_run_xai_pedi.py
python scripts/11_run_fairness_calibration.py
python scripts/12_compute_efficiency.py
python scripts/13_compute_tai_score.py
python scripts/14_generate_transparency_record.py
```

Key outputs:

- `results/tables/baseline_metrics.csv`
- `results/tables/federated_metrics.csv`
- `results/tables/privacy_attack_metrics.csv`
- `results/tables/pedi_metrics.csv`
- `results/tables/fairness_metrics.csv`
- `results/tables/calibration_metrics.csv`
- `results/tables/efficiency_results.csv`
- `results/tables/tai_scorecard.csv`
- `results/tables/tai_score_ranking.csv`
- `results/algorithmic_transparency_record.md`
- `results/governance/algorithmic_transparency_record.json`
- `results/governance/algorithmic_transparency_record.md`

## Real Data Workflow

Place canonical or harmonized raw files under:

```text
data/raw/nyc/
data/raw/chicago/
data/raw/boston/
data/raw/los_angeles/
```

The required shared schema is:

```text
request_id, created_date, closed_date, status, category, descriptor,
agency, latitude, longitude, area, city
```

Then run:

```powershell
python scripts/prepare_data.py --config configs/default.yaml
```

Los Angeles is written only to `data/splits/external/los_angeles_external_test.parquet`. It must never be included in training or model selection.

## Federated Learning

FedAvg:

```powershell
python scripts/run_federated.py --algorithm fedavg
```

FedProx:

```powershell
python scripts/run_federated.py --algorithm fedprox
```

Optional modes:

```powershell
python scripts/run_federated.py --algorithm fedavg --secure-aggregation
python scripts/run_federated.py --algorithm fedavg --dp
```

Secure aggregation is a simulation-layer overhead hook, not a production cryptographic protocol. DP uses Opacus when enabled and writes accounting metadata when available.

## Trustworthiness Outputs

After models are trained, run:

```powershell
python scripts/run_privacy_attack.py
python scripts/run_xai_pedi.py
python scripts/run_trustworthiness_eval.py
python scripts/make_transparency_record.py
```

Fairness is framed as geographic, area, and service-category fairness. The repository does not make demographic fairness claims unless demographic attributes are added and governed separately.

## Reproducibility Notes

- Configuration lives in `configs/default.yaml`.
- The canonical schema is documented in `docs/DATA_SCHEMA.md`.
- The full experimental protocol is documented in `docs/EXPERIMENTAL_PROTOCOL.md`.
- Reproduction instructions are documented in `docs/REPRODUCIBILITY.md`.
- Raw data and generated `results/` artifacts are intentionally ignored by git.
- Manuscript architecture figures are committed under `paper_figures/`.

## Architecture Figures

Editable DOT sources are stored in `paper_figures/architecture/dot/`. Rendered SVG, PDF, and PNG versions can be regenerated with:

```powershell
python scripts/render_architecture_figures.py
```

These figures are manuscript assets and are separate from experimental outputs under `results/`.

## Tests

```powershell
pytest -q
```

The CI workflow installs lightweight test dependencies and runs unit tests on synthetic inputs only. Large generated datasets, trained models, and result artifacts are ignored by git.
