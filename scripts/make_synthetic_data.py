"""Generate small canonical synthetic city datasets for smoke tests and CI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ALL_CITIES, RAW_DATA_DIR, ensure_project_dirs  # noqa: E402


CATEGORIES = ["noise", "sanitation", "road", "housing"]
AGENCIES = ["311", "public_works", "sanitation", "housing"]
AREAS = {
    "nyc": ["Manhattan", "Queens", "Brooklyn"],
    "chicago": ["Ward 1", "Ward 7", "Ward 22"],
    "boston": ["Dorchester", "Roxbury", "Back Bay"],
    "los_angeles": ["Central LA", "Valley", "Harbor"],
}
CENTERS = {
    "nyc": (40.7128, -74.0060),
    "chicago": (41.8781, -87.6298),
    "boston": (42.3601, -71.0589),
    "los_angeles": (34.0522, -118.2437),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-per-city", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_project_dirs()
    for city in ALL_CITIES:
        frame = make_city_frame(city, args.rows_per_city, args.seed)
        city_dir = args.output_dir / city
        city_dir.mkdir(parents=True, exist_ok=True)
        output_path = city_dir / f"{city}_synthetic.csv"
        frame.to_csv(output_path, index=False)
        print(f"Wrote {len(frame):,} synthetic rows: {output_path}")
    return 0


def make_city_frame(city: str, rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + sum(ord(char) for char in city))
    start = pd.Timestamp("2021-01-01")
    created = start + pd.to_timedelta(rng.integers(0, 365 * 2, size=rows), unit="D")
    created = created + pd.to_timedelta(rng.integers(0, 24, size=rows), unit="h")
    category = rng.choice(CATEGORIES, size=rows, p=[0.30, 0.30, 0.25, 0.15])
    category_multiplier = pd.Series(category).map(
        {"noise": 8, "sanitation": 24, "road": 72, "housing": 120}
    ).to_numpy()
    city_multiplier = {"nyc": 1.15, "chicago": 0.95, "boston": 0.85, "los_angeles": 1.25}[city]
    resolution_hours = rng.gamma(shape=2.0, scale=category_multiplier * city_multiplier / 2)
    closed = created + pd.to_timedelta(np.maximum(resolution_hours, 1), unit="h")
    lat_center, lon_center = CENTERS[city]
    descriptors = [f"{name} complaint" for name in category]
    return pd.DataFrame(
        {
            "request_id": [f"{city}-{index:05d}" for index in range(rows)],
            "created_date": created,
            "closed_date": closed,
            "status": "Closed",
            "category": category,
            "descriptor": descriptors,
            "agency": rng.choice(AGENCIES, size=rows),
            "latitude": lat_center + rng.normal(0, 0.03, size=rows),
            "longitude": lon_center + rng.normal(0, 0.03, size=rows),
            "area": rng.choice(AREAS[city], size=rows),
            "city": city,
        }
    ).sort_values(["created_date", "request_id"], kind="stable")


if __name__ == "__main__":
    raise SystemExit(main())
