"""Step 03: clean harmonized datasets."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("clean_step3_data.py")), run_name="__main__")
