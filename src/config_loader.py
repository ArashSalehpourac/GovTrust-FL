"""YAML configuration loading and validation for reproducible experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"

REQUIRED_PATHS = (
    "project.random_seed",
    "cities.training",
    "cities.external_validation",
    "data.raw_dir",
    "data.processed_dir",
    "data.splits_dir",
    "schema.required_columns",
    "target.column",
    "target.resolution_hours_column",
    "model.type",
    "fl.rounds",
    "fl.local_epochs",
    "fl.batch_size",
    "dp.enabled",
    "calibration.n_bins",
    "fairness.group_columns",
    "xai.sample_size",
    "tai_score.weights",
    "outputs.tables_dir",
    "outputs.figures_dir",
    "outputs.logs_dir",
    "outputs.models_dir",
)


class ConfigError(ValueError):
    """Raised when a configuration file is missing required fields."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate an experiment YAML file."""

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Validate required fields and common value constraints."""

    missing = [key for key in REQUIRED_PATHS if _get(config, key) is None]
    if missing:
        formatted = "\n".join(f"- {key}" for key in missing)
        raise ConfigError(f"Missing required configuration fields:\n{formatted}")

    training_cities = _get(config, "cities.training")
    if not isinstance(training_cities, list) or not training_cities:
        raise ConfigError("cities.training must be a non-empty list")

    external_city = _get(config, "cities.external_validation")
    if external_city in training_cities:
        raise ConfigError("External validation city must not appear in cities.training")

    weights = _get(config, "tai_score.weights")
    if not isinstance(weights, dict) or not weights:
        raise ConfigError("tai_score.weights must be a non-empty mapping")
    weight_sum = sum(float(value) for value in weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ConfigError(f"TAI-Score weights must sum to 1.0; got {weight_sum:.6f}")

    for key in ("data.test_size", "data.validation_size"):
        value = float(_get(config, key, 0.0))
        if not 0.0 < value < 0.5:
            raise ConfigError(f"{key} must be between 0 and 0.5")

    for key in ("fl.rounds", "fl.local_epochs", "fl.batch_size", "xai.sample_size"):
        if int(_get(config, key)) <= 0:
            raise ConfigError(f"{key} must be positive")


def resolve_path(config: dict[str, Any], dotted_key: str) -> Path:
    """Resolve a project-relative path from the configuration."""

    from .config import PROJECT_ROOT

    value = _get(config, dotted_key)
    if value is None:
        raise ConfigError(f"Missing path configuration: {dotted_key}")
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
