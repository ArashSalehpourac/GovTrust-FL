# GovTrust-FL

Federated learning starter project for delayed-resolution prediction across city service-request data.

The initial design treats NYC, Chicago, and Boston as federated clients, with Los Angeles held out for external validation. The main target is delayed resolution. Evaluation is planned across predictive performance, privacy, fairness, calibration, XAI stability, runtime, memory, and communication cost.

The T2 redesign (four-fold leave-one-city-out, leakage-safe time-to-resolution target, formal
example-level DP-SGD) lives alongside this pipeline in `src/redesign/`, `scripts/redesign/`, and
`tests/redesign/`; see `docs/redesign/README.md`. The Step 1-19 pipeline below is unchanged.

## Project Layout

- `data/raw/`: original city downloads.
- `data/processed/`: harmonized city-level datasets.
- `data/splits/`: train, validation, and test split files.
- `notebooks/`: step-by-step experiment notebooks.
- `src/`: reusable pipeline code.
- `results/`: tables, figures, logs, and serialized models.

## Quick Start

```powershell
cd GovTrust-FL
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start with `notebooks/01_data_download.ipynb`, then move through schema harmonization, preprocessing, baselines, federated learning, privacy, XAI, fairness, calibration, and final scorecards.

