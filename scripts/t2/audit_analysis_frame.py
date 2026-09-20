from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t2.audit_analysis import audit_analysis_frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit T2 chronology, target-isolation contract, outcome semantics, "
            "and analysis-frame data quality without training"
        )
    )
    parser.add_argument("--harmonized-dir", required=True)
    parser.add_argument("--harmonized-manifest", required=True)
    parser.add_argument("--structural-audit", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--batch-size", type=int, default=100_000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = audit_analysis_frames(
        harmonized_dir=Path(args.harmonized_dir).expanduser().resolve(),
        harmonized_manifest_path=Path(args.harmonized_manifest).expanduser().resolve(),
        structural_audit_path=Path(args.structural_audit).expanduser().resolve(),
        report_path=Path(args.report).expanduser().resolve(),
        batch_size=args.batch_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
