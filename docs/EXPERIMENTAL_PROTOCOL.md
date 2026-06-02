# Experimental Protocol

## Cities

- Federated clients and model-development cities: NYC, Chicago, Boston.
- Held-out external validation city: Los Angeles.

## Main Task

Delayed-resolution risk prediction:

```text
threshold(city, category) = Q3(resolution_hours | city, category)
delayed = 1 if resolution_hours > threshold(city, category) else 0
```

## Secondary Task

Service routing prediction using `agency` or `category`, configurable by experiment.

## Splitting

NYC, Chicago, and Boston are split independently in temporal order:

- Train: first 70%.
- Validation: next 15%.
- Internal test: final 15%.

Los Angeles is written only as `external_test`.

## Evaluation Dimensions

- Predictive utility: AUROC, AUPRC, accuracy, balanced accuracy, precision, recall, F1, Brier score.
- Privacy: membership-inference AUC, attack advantage, DP accounting when enabled.
- Explainability: SHAP, permutation importance, counterfactuals when feasible, PEDI.
- Fairness: city, area, and service-category subgroup gaps.
- Calibration: ECE and reliability diagrams.
- Efficiency: runtime, peak memory, model size, and FL communication cost.
- Governance: TAI-Score and transparency records.
