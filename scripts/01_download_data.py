"""Step 01: download raw municipal open-data extracts."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("download_step1_raw.py")), run_name="__main__")
