"""Step 02: harmonize city schemas."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("harmonize_step2_schema.py")), run_name="__main__")
