"""Build the four leave-one-city-out folds with fold-specific preprocessing.

Usage:
    python scripts/redesign/build_folds.py --max-rows-per-city 20000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.redesign.config import (  # noqa: E402
    CITIES,
    FOLDS_DIR,
    PILOT_CLEAN_DIR,
    SplitConfig,
    TargetConfig,
    build_folds,
)
from src.redesign.folds import build_fold, write_fold  # noqa: E402
from src.redesign.pilot_data import load_clean_city  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rows-per-city", type=int, default=20_000)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--tfidf-max-features", type=int, default=100)
    parser.add_argument("--clean-dir", type=Path, default=PILOT_CLEAN_DIR)
    parser.add_argument("--folds-dir", type=Path, default=FOLDS_DIR)
    args = parser.parse_args()

    city_frames = {city: load_clean_city(city, args.clean_dir) for city in CITIES}
    split_config = SplitConfig(train_frac=args.train_frac, val_frac=args.val_frac)
    target_config = TargetConfig()

    summary = []
    for spec in build_folds():
        print(f"Building {spec.name} (source={spec.source_cities})", flush=True)
        fold = build_fold(
            spec,
            city_frames,
            split_config=split_config,
            target_config=target_config,
            max_rows_per_city=args.max_rows_per_city,
            tfidf_max_features=args.tfidf_max_features,
        )
        fold_dir = write_fold(fold, args.folds_dir)
        train_rows = sum(len(splits["train"]) for splits in fold.source_splits.values())
        print(
            f"  train rows={train_rows:,} external rows={len(fold.external):,} "
            f"features={fold.preprocessor.output_dim} -> {fold_dir}",
            flush=True,
        )
        summary.append(
            {
                "fold": spec.name,
                "held_out_city": spec.held_out_city,
                "source_cities": list(spec.source_cities),
                "train_rows": train_rows,
                "external_rows": int(len(fold.external)),
                "feature_dim": fold.preprocessor.output_dim,
                "preprocessor_fingerprint": fold.preprocessor.fingerprint(),
                "sla_threshold_global_hours": fold.policy.global_sla_threshold_hours,
                "fold_dir": str(fold_dir),
            }
        )

    args.folds_dir.mkdir(parents=True, exist_ok=True)
    (args.folds_dir / "folds_summary.json").write_text(
        json.dumps({"folds": summary}, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.folds_dir / 'folds_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
