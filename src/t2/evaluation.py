from __future__ import annotations

import numpy as np


def regression_metrics(y_log: np.ndarray, pred_log: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_log, dtype=float)
    p = np.asarray(pred_log, dtype=float)
    err = p - y
    abs_err = np.abs(err)
    true_hours = np.expm1(y)
    pred_hours = np.maximum(0.0, np.expm1(p))
    hour_abs = np.abs(pred_hours - true_hours)
    return {
        "mae_log1p_hours": float(abs_err.mean()),
        "rmse_log1p_hours": float(np.sqrt(np.mean(err ** 2))),
        "mae_hours": float(hour_abs.mean()),
        "median_ae_hours": float(np.median(hour_abs)),
        "p90_ae_hours": float(np.quantile(hour_abs, 0.90)),
    }


def macro_source_mae(by_city: dict[str, dict[str, float]]) -> float:
    if not by_city:
        raise ValueError("no source-city metrics")
    return float(np.mean([m["mae_log1p_hours"] for m in by_city.values()]))
