"""Print the pre-experiment gate report from the manifests (no network, no training)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

CITY_ORDER = ("nyc", "chicago", "boston", "los_angeles")
YEARS = (2021, 2022, 2023, 2024, 2025)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", required=True, type=Path)
    args = parser.parse_args()
    man_dir = args.archive_dir / "03_Manifests_and_Checksums"
    raw = json.loads((man_dir / "RAW_MANIFEST.json").read_text())
    harm_path = man_dir / "HARMONIZATION_MANIFEST.json"
    harm = json.loads(harm_path.read_text()) if harm_path.exists() else {"cities": {}}

    lines: list[str] = []
    gates = raw["gates"]
    lines.append(f"RAW_DATA_GATE={gates['RAW_DATA_GATE']}")
    by_key = {(s["city"], s["year"]): s for s in raw["snapshots"]}
    for city in CITY_ORDER:
        for year in YEARS:
            s = by_key.get((city, year), {})
            lines.append(f"{city.upper()}_{year}_ROWS={s.get('raw_rows', 'MISSING')}")
    lines.append(f"RAW_FILES_COUNT={gates['RAW_FILES_COUNT']}")
    lines.append(f"RAW_SHA256_GATE={gates['RAW_SHA256_GATE']}")
    lines.append(f"SOURCE_DATE_RANGE_GATE={gates['SOURCE_DATE_RANGE_GATE']}")
    lines.append(f"DUPLICATE_AUDIT_GATE={gates['DUPLICATE_AUDIT_GATE']}")

    harm_ok = True
    hash_ok = True
    for city in CITY_ORDER:
        meta = harm["cities"].get(city)
        if meta is None:
            harm_ok = hash_ok = False
            continue
        path = args.archive_dir / "02_Harmonized" / meta["harmonized_file"]
        if not path.exists() or sha256_file(path) != meta["sha256"]:
            hash_ok = False
            continue
        sidecar = (args.archive_dir / "02_Harmonized" / f"{meta['harmonized_file']}.sha256")
        if not sidecar.exists() or sidecar.read_text().split()[0] != meta["sha256"]:
            hash_ok = False
        if pq.read_metadata(path).num_rows != meta["rows"] or not meta["row_accounting_balanced"]:
            harm_ok = False
        raw_total = sum(by_key[(city, y)].get("raw_rows", -1) for y in YEARS)
        if raw_total != meta["raw_rows_in_total"]:
            harm_ok = False
        for inp in meta["inputs"]:
            snap = next(s for s in raw["snapshots"] if s["snapshot_file"] == inp["snapshot_file"])
            if snap.get("sha256") != inp["sha256"]:
                harm_ok = False
    all_raw = all(g == "PASS" for k, g in gates.items() if k != "RAW_FILES_COUNT")
    lines.append(f"HARMONIZATION_GATE={'PASS' if harm_ok and all_raw else 'FAIL'}")
    lines.append(f"HARMONIZED_HASH_GATE={'PASS' if hash_ok and harm_ok else 'FAIL'}")
    lines.append("SCIENTIFIC_TRAINING_STARTED=NO")
    lines.append("REAL_RESULTS_GENERATED=NO")
    ready = all_raw and gates["RAW_FILES_COUNT"] == 20 and harm_ok and hash_ok
    lines.append(f"READY_FOR_EXPERIMENT_DESIGN_REVIEW={'YES' if ready else 'NO'}")
    report = "\n".join(lines)
    (man_dir / "GATE_REPORT.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
