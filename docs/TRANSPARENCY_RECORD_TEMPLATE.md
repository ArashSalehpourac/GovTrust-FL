# Algorithmic Transparency Record Template

## Purpose

Delayed-resolution risk prediction for municipal service-request triage research.

## Data Sources

NYC 311, Chicago 311, Boston 311, and Los Angeles MyLA311.

## Target Variable

City-category Q3 delayed-resolution label.

## Excluded Variables

`closed_date`, `resolution_hours`, `delay_threshold_hours`, final status, and other post-resolution fields.

## Training and Validation

NYC, Chicago, and Boston are used for model development. Los Angeles is external validation only.

## Privacy Protections

Document FL, secure aggregation simulation, differential privacy settings, and membership-inference results.

## Explainability

Document SHAP, permutation importance, counterfactuals, and PEDI.

## Fairness Scope

Geographic and service-category fairness only unless demographic data are explicitly added.

## Deployment Risks

Document calibration, subgroup performance, resource cost, and human oversight requirements.
