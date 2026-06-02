"""Step 06: create temporal train/validation/test splits."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("create_step6_splits.py")), run_name="__main__")
