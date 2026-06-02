"""Step 05: engineer creation-time features."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("engineer_step5_features.py")), run_name="__main__")
