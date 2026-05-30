"""Project configuration and shared constants."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"

FEDERATED_CLIENTS = ("nyc", "chicago", "boston")
EXTERNAL_VALIDATION_CITY = "los_angeles"
ALL_CITIES = (*FEDERATED_CLIENTS, EXTERNAL_VALIDATION_CITY)

RANDOM_STATE = 42
DELAY_THRESHOLD_DAYS = 30
TARGET_COLUMN = "delayed_resolution"

OPENED_AT_COLUMN = "opened_at"
CLOSED_AT_COLUMN = "closed_at"
RESOLUTION_DAYS_COLUMN = "resolution_days"


@dataclass(frozen=True)
class CityConfig:
    """Resolved paths for one city dataset."""

    name: str
    raw_dir: Path
    processed_path: Path
    split_dir: Path


CITY_CONFIGS = {
    city: CityConfig(
        name=city,
        raw_dir=RAW_DATA_DIR / city,
        processed_path=PROCESSED_DATA_DIR / f"{city}.parquet",
        split_dir=SPLITS_DIR / city,
    )
    for city in ALL_CITIES
}


def ensure_project_dirs() -> None:
    """Create expected local directories."""

    for config in CITY_CONFIGS.values():
        config.raw_dir.mkdir(parents=True, exist_ok=True)
        config.split_dir.mkdir(parents=True, exist_ok=True)

    for path in (
        PROCESSED_DATA_DIR,
        RESULTS_DIR / "tables",
        RESULTS_DIR / "figures",
        RESULTS_DIR / "logs",
        RESULTS_DIR / "models",
    ):
        path.mkdir(parents=True, exist_ok=True)

