from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t2.config import diagnostic_plan  # noqa: E402


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
        "diagnostic_jobs_including_controls": len(diagnostic_plan()),
        "training_started": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
