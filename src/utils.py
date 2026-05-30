"""Small shared utilities."""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from .config import RANDOM_STATE, ensure_project_dirs


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Set common random seeds."""

    random.seed(seed)
    np.random.seed(seed)


def bootstrap_project(seed: int = RANDOM_STATE) -> None:
    """Create directories and initialize repeatability controls."""

    ensure_project_dirs()
    set_seed(seed)


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Write a JSON artifact with stable formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path

