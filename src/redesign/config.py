"""Paths, fold definitions, and target configuration for the T2 redesign."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REDESIGN_DATA_DIR = PROJECT_ROOT / "data" / "redesign"
PILOT_RAW_DIR = REDESIGN_DATA_DIR / "raw_pilot"
PILOT_CLEAN_DIR = REDESIGN_DATA_DIR / "clean_pilot"
FOLDS_DIR = REDESIGN_DATA_DIR / "folds"
REDESIGN_RESULTS_DIR = PROJECT_ROOT / "results" / "redesign"

CITIES = ("nyc", "chicago", "boston", "los_angeles")

RESOLUTION_HOURS_COLUMN = "resolution_hours"
PRIMARY_TARGET_COLUMN = "y_log1p_resolution_hours"
SECONDARY_TARGET_COLUMN = "y_sla_breach"

CATEGORICAL_FEATURES = ("category", "descriptor", "agency", "area")
NUMERIC_FEATURES = (
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "latitude_grid",
    "longitude_grid",
    "area_request_count_7d",
    "category_request_count_7d",
    "agency_request_count_7d",
)
TEXT_FEATURE = "descriptor"

# Columns that must never reach the model: they encode the outcome.
OUTCOME_COLUMNS = (
    "closed_date",
    "status",
    RESOLUTION_HOURS_COLUMN,
    PRIMARY_TARGET_COLUMN,
    SECONDARY_TARGET_COLUMN,
    "sla_threshold_hours",
)
ID_COLUMNS = ("request_id", "created_date", "city")

QUANTILES = (0.1, 0.5, 0.9)


@dataclass(frozen=True)
class FoldSpec:
    """One leave-one-city-out fold."""

    held_out_city: str
    source_cities: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"holdout_{self.held_out_city}"


def build_folds(cities: tuple[str, ...] = CITIES) -> tuple[FoldSpec, ...]:
    """Return the four leave-one-city-out folds."""

    return tuple(
        FoldSpec(
            held_out_city=city,
            source_cities=tuple(other for other in cities if other != city),
        )
        for city in cities
    )


@dataclass(frozen=True)
class SplitConfig:
    """Chronological within-source-city split fractions."""

    train_frac: float = 0.70
    val_frac: float = 0.15

    def __post_init__(self) -> None:
        if not 0 < self.train_frac < 1:
            raise ValueError("train_frac must be in (0, 1)")
        if not 0 < self.val_frac < 1:
            raise ValueError("val_frac must be in (0, 1)")
        if self.train_frac + self.val_frac >= 1:
            raise ValueError("train_frac + val_frac must leave a non-empty internal test split")


@dataclass(frozen=True)
class TargetConfig:
    """Primary regression target plus the secondary policy threshold."""

    # Upper winsorization quantile for resolution_hours, estimated on source-train rows only.
    winsor_quantile: float = 0.995
    # Secondary SLA breach threshold quantile, estimated per comparable category on
    # source-train rows only. This is a policy definition, never fitted on val/test/held-out data.
    sla_quantile: float = 0.75
    min_category_rows_for_threshold: int = 50


@dataclass(frozen=True)
class PilotConfig:
    """Small pilot caps. Nothing here is a full-scale experiment."""

    max_rows_per_city: int = 20_000
    rounds: int = 5
    local_epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 1e-3
    hidden_sizes: tuple[int, ...] = (64, 32)
    proximal_mu: float = 0.01
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    seeds: tuple[int, ...] = (0, 1, 2)
    epsilons: tuple[float, ...] = (float("inf"), 5.0, 1.0)
    algorithms: tuple[str, ...] = ("fedavg", "fedprox")
    tfidf_max_features: int = 100
    secondary_loss_weight: float = 0.5
    subgroup_min_rows: int = 50
    quantiles: tuple[float, ...] = field(default=QUANTILES)
