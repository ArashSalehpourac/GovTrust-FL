from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from .config import CITIES


@dataclass(frozen=True)
class FoldSpec:
    held_out_city: str
    source_cities: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"holdout_{self.held_out_city}"


def loco_spec(held_out_city: str) -> FoldSpec:
    if held_out_city not in CITIES:
        raise ValueError(f"unknown city: {held_out_city}")
    return FoldSpec(held_out_city, tuple(city for city in CITIES if city != held_out_city))


def chronological_split(frame: pd.DataFrame, train_fraction: float = 0.70, val_fraction: float = 0.15) -> dict[str, pd.DataFrame]:
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError("invalid split fractions")
    ordered = frame.sort_values("created_date").reset_index(drop=True)
    n = len(ordered)
    if n < 10:
        raise ValueError("too few rows for chronological split")
    a = max(1, int(n * train_fraction))
    b = max(a + 1, int(n * (train_fraction + val_fraction)))
    b = min(b, n - 1)
    return {
        "train": ordered.iloc[:a].reset_index(drop=True),
        "val": ordered.iloc[a:b].reset_index(drop=True),
        "internal_test": ordered.iloc[b:].reset_index(drop=True),
    }


def split_source_cities(prepared: Mapping[str, pd.DataFrame], spec: FoldSpec) -> dict[str, dict[str, pd.DataFrame]]:
    if spec.held_out_city in spec.source_cities:
        raise AssertionError("held-out city cannot be a source")
    missing = set(spec.source_cities) - set(prepared)
    if missing:
        raise ValueError(f"missing source frames: {sorted(missing)}")
    return {city: chronological_split(prepared[city]) for city in spec.source_cities}
