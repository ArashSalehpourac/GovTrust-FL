# Reproducibility

This project is designed so every table and figure can be regenerated from fixed configuration files and deterministic seeds.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Synthetic Smoke Run

Synthetic data validate code behavior only; they are not paper results.

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

## Manuscript Protocol

1. Download official NYC, Chicago, Boston, and Los Angeles service-request data.
2. Harmonize all cities into the canonical schema.
3. Clean records and compute `resolution_hours`.
4. Create city-category Q3 delayed-resolution labels.
5. Engineer creation-time features only.
6. Split NYC, Chicago, and Boston temporally into 70/15/15 train/validation/test partitions.
7. Keep Los Angeles as external validation only.
8. Train centralized, local-only, and federated configurations.
9. Evaluate utility, privacy, explanation drift, fairness, calibration, and efficiency.
10. Compute TAI-Score and generate transparency records.

## Outputs

- Tables: `results/tables/`
- Figures: `results/figures/`
- Governance records: `results/governance/`
- Architecture manuscript assets: `paper_figures/architecture/`

Los Angeles must never be used for preprocessing fit, hyperparameter tuning, model selection, calibration fitting, or score normalization.
