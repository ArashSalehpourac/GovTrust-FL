"""Label creation for cleaned 311 datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


LABELED_SCHEMA = [
    "request_id",
    "created_date",
    "closed_date",
    "status",
    "category",
    "descriptor",
    "agency",
    "latitude",
    "longitude",
    "area",
    "city",
    "resolution_hours",
    "delay_threshold_hours",
    "delayed",
    "service_routing_target",
]


@dataclass(frozen=True)
class LabelingSpec:
    """Input/output paths for one cleaned city dataset."""

    city: str
    input_path: Path
    output_path: Path


def add_delayed_resolution_label(
    frame: pd.DataFrame,
    service_routing_source: str = "agency",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add city/category Q3 delay labels and return thresholds."""

    working = frame.copy()
    if service_routing_source not in {"agency", "category"}:
        raise ValueError("service_routing_source must be 'agency' or 'category'")

    thresholds = (
        working.groupby(["city", "category"], dropna=False, observed=True)["resolution_hours"]
        .quantile(0.75)
        .rename("delay_threshold_hours")
        .reset_index()
    )

    labeled = working.merge(thresholds, on=["city", "category"], how="left", validate="many_to_one")
    labeled["delayed"] = (
        labeled["resolution_hours"] > labeled["delay_threshold_hours"]
    ).astype("int8")
    primary = labeled[service_routing_source].astype("string").str.strip().fillna("")
    fallback_column = "category" if service_routing_source == "agency" else "agency"
    labeled["service_routing_target"] = primary
    fallback_mask = labeled["service_routing_target"].eq("")
    labeled.loc[fallback_mask, "service_routing_target"] = labeled.loc[fallback_mask, fallback_column].astype("string")
    labeled["service_routing_target"] = labeled["service_routing_target"].astype("string")

    return labeled[LABELED_SCHEMA], thresholds


def label_file(spec: LabelingSpec) -> tuple[dict[str, object], pd.DataFrame]:
    """Label one cleaned city file and write a labeled Parquet file."""

    frame = pd.read_parquet(spec.input_path)
    labeled, thresholds = add_delayed_resolution_label(frame)
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(spec.output_path, index=False)

    delayed_rate = float(labeled["delayed"].mean()) if len(labeled) else None
    result = {
        "city": spec.city,
        "input_path": str(spec.input_path),
        "output_path": str(spec.output_path),
        "rows": int(len(labeled)),
        "delayed_positive": int(labeled["delayed"].sum()),
        "delayed_rate": delayed_rate,
        "threshold_groups": int(len(thresholds)),
        "service_routing_target_classes": int(labeled["service_routing_target"].nunique(dropna=True)),
        "threshold_min": _float_or_none(thresholds["delay_threshold_hours"].min()),
        "threshold_max": _float_or_none(thresholds["delay_threshold_hours"].max()),
    }
    return result, thresholds


def _float_or_none(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)
