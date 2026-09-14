"""Leakage-safe target construction.

Primary outcome: ``log1p(resolution_hours)`` (time-to-resolution regression).
Secondary outcome: a binary SLA-breach indicator whose threshold is a *policy*
defined exclusively on source-city training rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import (
    PRIMARY_TARGET_COLUMN,
    RESOLUTION_HOURS_COLUMN,
    SECONDARY_TARGET_COLUMN,
    TargetConfig,
)


@dataclass(frozen=True)
class TargetPolicy:
    """Everything about the target that is estimated from data.

    Every field is fitted on source-city training rows only and then applied
    unchanged to validation, internal test, and held-out-city rows.
    """

    winsor_quantile: float
    winsor_hours: float
    sla_quantile: float
    global_sla_threshold_hours: float
    category_sla_threshold_hours: dict[str, float]
    fitted_on_rows: int
    fitted_on_cities: tuple[str, ...]
    min_category_rows_for_threshold: int

    def threshold_for(self, categories: pd.Series) -> pd.Series:
        """Return the policy SLA threshold per row, defaulting to the global one."""

        keys = categories.astype("string").fillna("__missing__")
        mapped = keys.map(self.category_sla_threshold_hours)
        return pd.to_numeric(mapped, errors="coerce").fillna(self.global_sla_threshold_hours)

    def to_dict(self) -> dict[str, object]:
        return {
            "primary_target": PRIMARY_TARGET_COLUMN,
            "primary_definition": "log1p(min(resolution_hours, winsor_hours))",
            "winsor_quantile": self.winsor_quantile,
            "winsor_hours": self.winsor_hours,
            "secondary_target": SECONDARY_TARGET_COLUMN,
            "secondary_definition": (
                "resolution_hours > per-category policy threshold; "
                "thresholds are train-only quantiles of source-city training rows"
            ),
            "sla_quantile": self.sla_quantile,
            "global_sla_threshold_hours": self.global_sla_threshold_hours,
            "category_sla_threshold_hours": self.category_sla_threshold_hours,
            "min_category_rows_for_threshold": self.min_category_rows_for_threshold,
            "fit_scope": {
                "rows": self.fitted_on_rows,
                "cities": list(self.fitted_on_cities),
                "split": "source-city train only",
            },
        }


def fit_target_policy(train_frame: pd.DataFrame, config: TargetConfig) -> TargetPolicy:
    """Fit winsorization and SLA thresholds on source-city training rows only."""

    hours = pd.to_numeric(train_frame[RESOLUTION_HOURS_COLUMN], errors="coerce").dropna()
    if hours.empty:
        raise ValueError("Cannot fit a target policy on an empty training frame.")

    winsor_hours = float(hours.quantile(config.winsor_quantile))
    global_threshold = float(hours.quantile(config.sla_quantile))

    working = train_frame.loc[:, ["category", RESOLUTION_HOURS_COLUMN]].copy()
    working["category"] = working["category"].astype("string").fillna("__missing__")
    working[RESOLUTION_HOURS_COLUMN] = pd.to_numeric(
        working[RESOLUTION_HOURS_COLUMN], errors="coerce"
    )
    working = working.dropna(subset=[RESOLUTION_HOURS_COLUMN])

    grouped = working.groupby("category", observed=True)[RESOLUTION_HOURS_COLUMN]
    thresholds = grouped.quantile(config.sla_quantile)
    sizes = grouped.size()
    eligible = sizes[sizes >= config.min_category_rows_for_threshold].index
    category_thresholds = {
        str(category): float(thresholds.loc[category]) for category in eligible
    }

    return TargetPolicy(
        winsor_quantile=config.winsor_quantile,
        winsor_hours=winsor_hours,
        sla_quantile=config.sla_quantile,
        global_sla_threshold_hours=global_threshold,
        category_sla_threshold_hours=category_thresholds,
        fitted_on_rows=int(len(train_frame)),
        fitted_on_cities=tuple(sorted(train_frame["city"].astype(str).unique())),
        min_category_rows_for_threshold=config.min_category_rows_for_threshold,
    )


def apply_target_policy(frame: pd.DataFrame, policy: TargetPolicy) -> pd.DataFrame:
    """Attach primary and secondary targets using an already-fitted policy."""

    output = frame.copy()
    hours = pd.to_numeric(output[RESOLUTION_HOURS_COLUMN], errors="coerce")
    winsorized = hours.clip(upper=policy.winsor_hours)
    output[PRIMARY_TARGET_COLUMN] = np.log1p(winsorized.clip(lower=0.0)).astype("float32")

    thresholds = policy.threshold_for(output["category"])
    output["sla_threshold_hours"] = thresholds.astype("float32")
    output[SECONDARY_TARGET_COLUMN] = (hours > thresholds).astype("int8")
    return output
