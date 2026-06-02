"""Step 14: generate algorithmic transparency records."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("make_transparency_record.py")), run_name="__main__")
