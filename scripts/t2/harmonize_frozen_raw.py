from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t2.harmonize_frozen import harmonize_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and harmonize the 20 frozen T2 raw snapshots. "
            "This command never fetches live municipal APIs."
        )
    )
    parser.add_argument(
        "--raw-dir",
        required=True,
        help="Mounted Drive 01_Raw_Official directory containing the 20 canonical *_raw.csv files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Mounted Drive 02_Harmonized directory; existing outputs are never overwritten",
    )
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "configs" / "t2" / "frozen_raw_manifest.json"),
        help="Frozen raw manifest JSON committed with the execution code",
    )
    parser.add_argument(
        "--metadata-dir",
        default=None,
        help="Directory for harmonized manifest/checksum sidecars; defaults to output-dir",
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = harmonize_all(
        raw_dir=Path(args.raw_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        manifest_path=Path(args.manifest).expanduser().resolve(),
        metadata_dir=(
            Path(args.metadata_dir).expanduser().resolve()
            if args.metadata_dir is not None
            else None
        ),
        chunksize=args.chunksize,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
