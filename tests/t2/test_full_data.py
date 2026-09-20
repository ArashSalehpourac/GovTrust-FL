import json

import pandas as pd

from src.t2.full_data import (
    load_external_2025,
    load_harmonized_manifest,
    load_source_city_bundle,
)
from src.t2.preprocessing import build_fixed_preprocessor
from src.t2.provenance import sha256_file


def _write_year(tmp_path, year: int) -> dict[str, object]:
    created = [
        pd.Timestamp(f"{year}-06-01T00:00:00Z"),
        pd.Timestamp(f"{year}-12-31T00:00:00Z"),
    ]
    closed = [
        created[0] + pd.Timedelta(hours=1),
        created[1] + pd.Timedelta(days=2),
    ]
    frame = pd.DataFrame(
        {
            "request_id": [f"a-{year}", f"b-{year}"],
            "created_date": created,
            "closed_date": closed,
            "status": ["closed", "closed"],
            "category": ["Tree", "Pothole"],
            "descriptor": ["x", "y"],
            "agency": ["a", "a"],
            "latitude": [1.0, 1.0],
            "longitude": [2.0, 2.0],
            "area": ["q", "q"],
            "city": ["nyc", "nyc"],
        }
    )
    path = tmp_path / f"NYC_{year}_harmonized.parquet"
    frame.to_parquet(path, index=False)
    return {
        "city": "nyc",
        "year": year,
        "harmonized_filename": path.name,
        "harmonized_sha256": sha256_file(path),
        "harmonized_size_bytes": path.stat().st_size,
        "rows": len(frame),
        "coverage_note": None,
    }


def test_full_data_loader_builds_purged_compact_splits(tmp_path):
    outputs = [_write_year(tmp_path, year) for year in range(2021, 2026)]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "row_filtering": "none",
                "outputs": outputs
                + [
                    {
                        "city": city,
                        "year": year,
                        "harmonized_filename": f"unused-{city}-{year}",
                        "harmonized_sha256": "x",
                        "harmonized_size_bytes": 1,
                        "rows": 1,
                        "coverage_note": None,
                    }
                    for city in ("chicago", "boston", "los_angeles")
                    for year in range(2021, 2026)
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = load_harmonized_manifest(manifest_path)
    pre = build_fixed_preprocessor(text_hash_dim=16, category_hash_dim=4)

    bundle = load_source_city_bundle(
        harmonized_dir=tmp_path,
        harmonized_manifest=manifest,
        city="nyc",
        preprocessor=pre,
    )

    # The Dec-31 row in 2023 closes in 2024 and is purged.
    assert len(bundle.splits["train"]) == 5
    # The Dec-31 row in 2024 closes in 2025 and is purged.
    assert len(bundle.splits["val"]) == 1
    assert len(bundle.splits["internal_test"]) == 2
    assert bundle.summaries["train"]["rows"] == 5
    assert set(bundle.input_sha256_by_year) == {
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
    }

    external, summary, sha, notes = load_external_2025(
        harmonized_dir=tmp_path,
        harmonized_manifest=manifest,
        city="nyc",
        preprocessor=pre,
    )
    assert len(external) == 2
    assert summary["rows"] == 2
    assert set(sha) == {"2025"}
    assert notes == []
