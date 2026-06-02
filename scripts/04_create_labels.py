"""Step 04: create delayed-resolution and routing labels."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("label_step4_targets.py")), run_name="__main__")
