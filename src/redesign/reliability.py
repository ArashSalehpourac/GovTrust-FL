"""Reliability metrics for the time-to-resolution target and its secondary label."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import QUANTILES


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
    """Quantile (pinball) loss for one quantile level."""

    residual = y_true - y_pred
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def regression_metrics(
    y_true_log: np.ndarray,
    quantile_predictions: dict[float, np.ndarray],
    *,
    winsor_hours: float | None = None,
) -> dict[str, float]:
    """MAE/RMSE on the median head plus pinball losses and interval coverage."""

    y_true_log = np.asarray(y_true_log, dtype=np.float64)
    median_key = min(quantile_predictions, key=lambda tau: abs(tau - 0.5))
    median = np.asarray(quantile_predictions[median_key], dtype=np.float64)

    errors = y_true_log - median
    metrics: dict[str, float] = {
        "n": int(len(y_true_log)),
        "mae_log1p_hours": float(np.mean(np.abs(errors))),
        "rmse_log1p_hours": float(np.sqrt(np.mean(errors**2))),
        "bias_log1p_hours": float(np.mean(median - y_true_log)),
    }

    true_hours = np.expm1(y_true_log)
    pred_hours = np.expm1(median)
    if winsor_hours is not None:
        pred_hours = np.clip(pred_hours, 0.0, winsor_hours)
    metrics["mae_hours"] = float(np.mean(np.abs(true_hours - pred_hours)))
    metrics["rmse_hours"] = float(np.sqrt(np.mean((true_hours - pred_hours) ** 2)))
    metrics["median_absolute_error_hours"] = float(np.median(np.abs(true_hours - pred_hours)))

    for tau, prediction in sorted(quantile_predictions.items()):
        prediction = np.asarray(prediction, dtype=np.float64)
        metrics[f"pinball_q{int(round(tau * 100)):02d}"] = pinball_loss(y_true_log, prediction, tau)
        metrics[f"coverage_below_q{int(round(tau * 100)):02d}"] = float(
            np.mean(y_true_log <= prediction)
        )
    metrics["mean_pinball"] = float(
        np.mean(
            [
                metrics[f"pinball_q{int(round(tau * 100)):02d}"]
                for tau in quantile_predictions
            ]
        )
    )

    lower_tau = min(quantile_predictions)
    upper_tau = max(quantile_predictions)
    if upper_tau > lower_tau:
        lower = np.asarray(quantile_predictions[lower_tau], dtype=np.float64)
        upper = np.asarray(quantile_predictions[upper_tau], dtype=np.float64)
        nominal = upper_tau - lower_tau
        empirical = float(np.mean((y_true_log >= lower) & (y_true_log <= upper)))
        metrics["interval_nominal_coverage"] = float(nominal)
        metrics["interval_empirical_coverage"] = empirical
        metrics["interval_coverage_gap"] = empirical - float(nominal)
        metrics["interval_mean_width_log1p_hours"] = float(np.mean(upper - lower))
    return metrics


def binary_reliability_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    """Brier, ECE, and discrimination for the secondary SLA-breach label."""

    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    metrics = {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)) if len(y_true) else float("nan"),
        "mean_predicted": float(np.mean(y_prob)) if len(y_prob) else float("nan"),
        "brier": float(np.mean((y_prob - y_true) ** 2)) if len(y_true) else float("nan"),
        "ece": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
    }
    if len(np.unique(y_true)) == 2:
        from sklearn.metrics import average_precision_score, roc_auc_score

        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        metrics["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    return metrics


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, edges[1:-1], right=True)
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(float(np.mean(y_true[mask]) - np.mean(y_prob[mask])))
    return float(ece)


def reliability_report(
    y_true_log: np.ndarray,
    quantile_predictions: dict[float, np.ndarray],
    y_secondary: np.ndarray | None,
    y_secondary_prob: np.ndarray | None,
    *,
    winsor_hours: float | None = None,
) -> dict[str, object]:
    """Combined primary + secondary reliability summary."""

    report: dict[str, object] = {
        "primary": regression_metrics(
            y_true_log, quantile_predictions, winsor_hours=winsor_hours
        )
    }
    if y_secondary is not None and y_secondary_prob is not None:
        report["secondary_sla"] = binary_reliability_metrics(y_secondary, y_secondary_prob)
    return report


def subgroup_reliability(
    frame: pd.DataFrame,
    y_true_log: np.ndarray,
    quantile_predictions: dict[float, np.ndarray],
    y_secondary: np.ndarray,
    y_secondary_prob: np.ndarray,
    *,
    min_rows: int = 50,
    winsor_hours: float | None = None,
    max_groups: int = 15,
) -> list[dict[str, object]]:
    """Operational subgroup summaries.

    Groups are service category, geographic/area group, and request-volume
    strata. These are operational strata of the administrative system, not
    demographic groups, and must not be reported as demographic fairness.
    """

    rows: list[dict[str, object]] = []
    cities = sorted({str(value) for value in frame["city"].unique()})
    city_scope = cities[0] if len(cities) == 1 else "+".join(cities)
    groupings = {
        "service_category": frame["category"].astype("string").fillna("missing"),
        "area_group": frame["area"].astype("string").fillna("missing"),
        "request_volume_stratum": _volume_strata(frame["category_request_count_7d"]),
    }

    for grouping_name, keys in groupings.items():
        counts = keys.value_counts()
        eligible = [key for key in counts.index if counts[key] >= min_rows][:max_groups]
        for key in eligible:
            mask = (keys == key).to_numpy()
            if mask.sum() < min_rows:
                continue
            group_quantiles = {tau: values[mask] for tau, values in quantile_predictions.items()}
            entry: dict[str, object] = {
                "city_scope": city_scope,
                "grouping": grouping_name,
                "group": str(key),
                "rows": int(mask.sum()),
                "share_of_split": float(mask.mean()),
            }
            entry.update(
                {
                    f"primary_{name}": value
                    for name, value in regression_metrics(
                        np.asarray(y_true_log)[mask],
                        group_quantiles,
                        winsor_hours=winsor_hours,
                    ).items()
                }
            )
            entry.update(
                {
                    f"secondary_{name}": value
                    for name, value in binary_reliability_metrics(
                        np.asarray(y_secondary)[mask], np.asarray(y_secondary_prob)[mask]
                    ).items()
                }
            )
            rows.append(entry)
    return rows


def _volume_strata(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    try:
        strata = pd.qcut(
            numeric, q=3, labels=("low_volume", "medium_volume", "high_volume"), duplicates="drop"
        )
    except ValueError:
        return pd.Series(["all_volume"] * len(numeric), index=numeric.index, dtype="string")
    return strata.astype("string").fillna("unknown_volume")


DEFAULT_QUANTILES = QUANTILES
