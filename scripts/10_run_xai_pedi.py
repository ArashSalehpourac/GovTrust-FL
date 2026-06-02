"""Step 10: run XAI and PEDI evaluation."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("run_xai_pedi.py")), run_name="__main__")
