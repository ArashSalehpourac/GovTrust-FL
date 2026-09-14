"""Download the capped pilot extract for the redesigned pipeline.

Usage:
    python scripts/redesign/fetch_pilot_data.py --rows-per-city 20000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.redesign.pilot_data import build_pilot_extract  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-per-city", type=int, default=20_000)
    parser.add_argument("--min-category-samples", type=int, default=50)
    args = parser.parse_args()

    manifest = build_pilot_extract(
        rows_per_city=args.rows_per_city,
        min_category_samples=args.min_category_samples,
    )
    for output in manifest["outputs"]:
        print(
            f"{output['city']}: raw={output['raw_rows']:,} clean={output['clean_rows']:,} "
            f"categories={output['categories']} "
            f"[{output['created_date_min']} .. {output['created_date_max']}]",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
