"""Random seed helpers."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set common Python and NumPy random seeds."""

    random.seed(seed)
    np.random.seed(seed)
