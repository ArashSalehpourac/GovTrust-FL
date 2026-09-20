from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t2.config import diagnostic_plan, full_loco_plan


def main() -> int:
    import numpy
    import opacus
    import pandas
    import sklearn
    import torch

    print(json.dumps({
        "status": "preflight_ok",
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "opacus": opacus.__version__,
        "legacy_diagnostic_jobs_including_controls": len(diagnostic_plan()),
        "full_loco_primary_jobs": len(
            full_loco_plan(include_clipped_no_noise=False)
        ),
        "full_loco_jobs_including_controls": len(full_loco_plan()),
        "training_started": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
