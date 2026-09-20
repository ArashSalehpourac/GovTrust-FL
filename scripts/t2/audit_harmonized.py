from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t2.audit_harmonized import audit_harmonized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the 20 frozen-raw-derived T2 harmonized outputs without training"
    )
    parser.add_argument("--harmonized-dir", required=True)
    parser.add_argument(
        "--raw-manifest",
        default=str(ROOT / "configs" / "t2" / "frozen_raw_manifest.json"),
    )
    parser.add_argument("--harmonized-manifest", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = audit_harmonized(
        harmonized_dir=Path(args.harmonized_dir).expanduser().resolve(),
        raw_manifest_path=Path(args.raw_manifest).expanduser().resolve(),
        harmonized_manifest_path=Path(args.harmonized_manifest).expanduser().resolve(),
        report_path=Path(args.report).expanduser().resolve(),
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
