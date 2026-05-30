"""Data loading helpers for city-level service request files."""

from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import CITY_CONFIGS


SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".json", ".jsonl", ".xlsx"}


def _resolve_city(city: str) -> str:
    city_key = city.lower()
    if city_key not in CITY_CONFIGS:
        raise ValueError(f"Unknown city '{city}'. Expected one of: {sorted(CITY_CONFIGS)}")
    return city_key


def list_raw_files(city: str) -> list[Path]:
    """Return supported raw data files for one city."""

    city_key = _resolve_city(city)
    raw_dir = CITY_CONFIGS[city_key].raw_dir
    return sorted(
        path
        for path in raw_dir.glob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read a tabular file using its extension."""

    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path, **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(file_path, **kwargs)
    if suffix == ".json":
        return pd.read_json(file_path, **kwargs)
    if suffix == ".jsonl":
        return pd.read_json(file_path, lines=True, **kwargs)
    if suffix == ".xlsx":
        return pd.read_excel(file_path, **kwargs)

    raise ValueError(f"Unsupported file extension: {suffix}")


def load_raw_city(city: str, files: Iterable[str | Path] | None = None) -> pd.DataFrame:
    """Load and concatenate raw files for one city."""

    city_key = _resolve_city(city)
    file_paths = [Path(path) for path in files] if files is not None else list_raw_files(city_key)
    if not file_paths:
        raise FileNotFoundError(f"No raw files found for {city_key}.")

    frames = []
    for file_path in file_paths:
        frame = read_table(file_path)
        frame["source_file"] = file_path.name
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def load_processed_city(city: str) -> pd.DataFrame:
    """Load the processed parquet file for one city."""

    city_key = _resolve_city(city)
    return pd.read_parquet(CITY_CONFIGS[city_key].processed_path)


def save_processed_city(city: str, frame: pd.DataFrame) -> Path:
    """Save a processed city frame and return its path."""

    city_key = _resolve_city(city)
    output_path = CITY_CONFIGS[city_key].processed_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return output_path

