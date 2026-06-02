"""Step 13: compute TAI-Score ranking."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("run_trustworthiness_eval.py")), run_name="__main__")
