"""Run the synthetic smoke pipeline end to end.

This command is intended for code validation, not for paper-result claims.
Real paper experiments should use the official municipal datasets.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    """Run the reproducible synthetic smoke workflow."""

    commands = [
        ["scripts/make_synthetic_data.py", "--rows-per-city", "100"],
        ["scripts/prepare_data.py"],
        ["scripts/run_baselines.py"],
        ["scripts/run_federated.py", "--algorithm", "fedavg"],
        ["scripts/run_privacy_attack.py"],
        ["scripts/run_xai_pedi.py"],
        ["scripts/run_trustworthiness_eval.py"],
        ["scripts/make_transparency_record.py"],
    ]
    for command in commands:
        print("Running:", " ".join(command), flush=True)
        completed = subprocess.run([sys.executable, *command], cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
