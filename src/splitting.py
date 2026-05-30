"""Experimental split creation for GovTrust-FL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


FEDERATED_CITIES = ("nyc", "chicago", "boston")
EXTERNAL_CITY = "los_angeles"


@dataclass(frozen=True)
class SplitSpec:
    """Input features and split output paths for one city."""

    city: str
    input_path: Path
    output_dir: Path


def time_split_frame(
    frame: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict[str, pd.DataFrame]:
    """Split one city by creation time into train/val/test."""

    ordered = frame.sort_values(["created_date", "request_id"], kind="stable").reset_index(drop=True)
    n_rows = len(ordered)
    train_end = int(n_rows * train_frac)
    val_end = int(n_rows * (train_frac + val_frac))
    return {
        "train": ordered.iloc[:train_end].copy(),
        "val": ordered.iloc[train_end:val_end].copy(),
        "test": ordered.iloc[val_end:].copy(),
    }


def write_city_splits(spec: SplitSpec) -> dict[str, object]:
    """Write per-city train/val/test split files."""

    frame = pd.read_parquet(spec.input_path)
    split_frames = time_split_frame(frame)
    spec.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}
    for split_name, split_frame in split_frames.items():
        output_path = spec.output_dir / f"{spec.city}_{split_name}.parquet"
        split_frame.to_parquet(output_path, index=False)
        outputs[split_name] = {
            "path": str(output_path),
            "rows": int(len(split_frame)),
            "created_date_min": _iso_or_none(split_frame["created_date"].min()),
            "created_date_max": _iso_or_none(split_frame["created_date"].max()),
            "delayed_rate": float(split_frame["delayed"].mean()),
        }

    return {
        "city": spec.city,
        "input_path": str(spec.input_path),
        "outputs": outputs,
    }


def write_external_test(city: str, input_path: Path, output_path: Path) -> dict[str, object]:
    """Write LA as external-only validation/test file."""

    frame = pd.read_parquet(input_path).sort_values(["created_date", "request_id"], kind="stable")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return {
        "city": city,
        "input_path": str(input_path),
        "outputs": {
            "external_test": {
                "path": str(output_path),
                "rows": int(len(frame)),
                "created_date_min": _iso_or_none(frame["created_date"].min()),
                "created_date_max": _iso_or_none(frame["created_date"].max()),
                "delayed_rate": float(frame["delayed"].mean()),
            }
        },
    }


def write_combined_split(
    input_paths: list[Path],
    output_path: Path,
    split_name: str,
) -> dict[str, object]:
    """Combine same-named city split files into one pooled split."""

    frames = [pd.read_parquet(path) for path in input_paths]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["created_date", "city", "request_id"], kind="stable")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    return {
        "split": split_name,
        "path": str(output_path),
        "rows": int(len(combined)),
        "created_date_min": _iso_or_none(combined["created_date"].min()),
        "created_date_max": _iso_or_none(combined["created_date"].max()),
        "delayed_rate": float(combined["delayed"].mean()),
        "city_rows": combined["city"].value_counts().sort_index().to_dict(),
    }


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _iso_or_none(value) -> str | None:
    if pd.isna(value):
        return None
    return value.isoformat()

