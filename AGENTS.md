# GovTrust-FL Agent Instructions

## Environment

- Preferred Python version: Python 3.11.
- Setup command:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
  ```
- Test command:
  ```powershell
  python -m pytest
  ```
- Formatting/linting command:
  ```powershell
  python -m ruff check .
  python -m black --check .
  ```

## Research Rules

- Raw municipal datasets must not be committed.
- Los Angeles must never be used in training, validation, model selection, feature selection, preprocessing fit, calibration fitting, threshold tuning, or score normalization.
- Delayed-resolution labels must be generated with city-category Q3 thresholds:
  `delayed = 1 if resolution_hours > Q3(resolution_hours | city, category) else 0`.
- Do not use a global fixed delay threshold for main manuscript experiments.
- All results must be reproducible with fixed seeds.
- Generated experimental tables and figures go under `results/`.
- Manuscript architecture assets belong under `paper_figures/`.
- Do not fabricate results. Synthetic data are for tests and smoke validation only.
