"""Synthetic municipal data for the redesign leakage tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.redesign.config import CITIES  # noqa: E402

CITY_PROFILES = {
    "nyc": {"categories": ("noise", "pothole", "heat"), "base_hours": 30.0},
    "chicago": {"categories": ("pothole", "graffiti", "rodent"), "base_hours": 60.0},
    "boston": {"categories": ("snow", "pothole", "streetlight"), "base_hours": 45.0},
    "los_angeles": {"categories": ("bulky_items", "graffiti", "pothole"), "base_hours": 90.0},
}


def make_city_frame(city: str, n_rows: int = 400, seed: int = 0) -> pd.DataFrame:
    """Build one synthetic cleaned city frame with the harmonized schema."""

    rng = np.random.default_rng(seed)
    profile = CITY_PROFILES[city]
    categories = rng.choice(profile["categories"], size=n_rows)
    created = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(n_rows) * 3, unit="h")
    hours = np.abs(rng.lognormal(mean=np.log(profile["base_hours"]), sigma=0.8, size=n_rows))
    return pd.DataFrame(
        {
            "request_id": [f"{city}-{index:05d}" for index in range(n_rows)],
            "created_date": created,
            "closed_date": created + pd.to_timedelta(hours, unit="h"),
            "status": "Closed",
            "category": categories,
            "descriptor": [
                f"{category} report detail {index % 7}"
                for index, category in enumerate(categories)
            ],
            "agency": [f"{city}_dept_{index % 3}" for index in range(n_rows)],
            "latitude": 40.0 + rng.normal(0, 0.05, n_rows),
            "longitude": -73.0 + rng.normal(0, 0.05, n_rows),
            "area": [f"{city}_area_{index % 4}" for index in range(n_rows)],
            "city": city,
            "resolution_hours": hours,
        }
    )


@pytest.fixture()
def city_frames() -> dict[str, pd.DataFrame]:
    return {city: make_city_frame(city, seed=index) for index, city in enumerate(CITIES)}


@pytest.fixture()
def poison_frame():
    """Corrupt every row of a city frame beyond recognition.

    Any fitted artifact that changes when only the held-out city is poisoned
    must have touched held-out-city rows.
    """

    def poison(frame: pd.DataFrame) -> pd.DataFrame:
        poisoned = frame.copy()
        poisoned["category"] = "zzz_poison_category"
        poisoned["descriptor"] = "zzzpoisontoken zzzpoisontoken"
        poisoned["agency"] = "zzz_poison_agency"
        poisoned["area"] = "zzz_poison_area"
        poisoned["latitude"] = 1.0
        poisoned["longitude"] = 1.0
        poisoned["resolution_hours"] = 10_000.0
        poisoned["closed_date"] = poisoned["created_date"] + pd.Timedelta(hours=10_000)
        return poisoned

    return poison
