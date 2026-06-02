"""Step 07: train centralized and local-only baselines."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("run_baselines.py")), run_name="__main__")
