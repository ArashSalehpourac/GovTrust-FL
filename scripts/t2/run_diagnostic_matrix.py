from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.t2.config import CITIES, T2Config, diagnostic_plan
from src.t2.provenance import canonical_hash, git_state, sha256_file

INPUT_PROTOCOL = "t2_diagnostic_input_archive_v2"
LEGACY_DIAGNOSTIC_EXECUTION_RETIRED = True


def _epsilon(value: object) -> float:
    if isinstance(value, str) and value.lower() in {"inf", "infinity"}:
        return float("inf")
    return float(value)


def _job_key(row: dict[str, object]) -> tuple[str, int, str, float]:
    return (
        str(row["held_out_city"]),
        int(row["seed"]),
        str(row["mode"]),
        _epsilon(row["epsilon"]),
    )


def _config_for(row: dict[str, object], device: str) -> T2Config:
    mode = str(row["mode"])
    epsilon = _epsilon(row["epsilon"])
    return T2Config(
        mode=mode,  # type: ignore[arg-type]
        target_epsilon=epsilon if mode == "private" else float("inf"),
        seed=int(row["seed"]),
        device=device,  # type: ignore[arg-type]
    )


def _input_paths(input_dir: Path) -> dict[str, Path]:
    paths = {city: input_dir / f"{city}.parquet" for city in CITIES}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing T2 diagnostic input files: {missing}")
    manifest_path = input_dir / "t2_input_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing T2 input manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != INPUT_PROTOCOL:
        raise ValueError(
            f"unexpected T2 input manifest protocol: {manifest.get('protocol')!r}; "
            f"expected {INPUT_PROTOCOL!r}"
        )
    if manifest.get("historical_step3_cleaned_forbidden") is not True:
        raise ValueError("input manifest does not forbid historical Step-3 cleaned data")
    selection = manifest.get("selection", {})
    if selection.get("uses_outcome_or_closed_date") is not False:
        raise ValueError("input archive selection must be outcome-independent")
    if selection.get("uses_status") is not False:
        raise ValueError("input archive selection must not use status")
    for city, path in paths.items():
        expected = manifest["outputs"][city]["archive_sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"{city}: archived input SHA256 mismatch")
    return paths


def _input_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {city: sha256_file(path) for city, path in paths.items()}


def _completed_keys(
    output_dir: Path,
    *,
    git_sha: str,
    input_hashes: dict[str, str],
    device: str,
) -> set[tuple[str, int, str, float]]:
    completed: set[tuple[str, int, str, float]] = set()
    if not output_dir.exists():
        return completed

    expected_config_hashes = {
        _job_key(row): canonical_hash(_config_for(row, device).as_dict())
        for row in diagnostic_plan()
    }
    for run_path in output_dir.glob("*/run.json"):
        try:
            run = json.loads(run_path.read_text(encoding="utf-8"))
            manifest = run["manifest"]
            row = {
                "held_out_city": run["held_out_city"],
                "seed": run["seed"],
                "mode": run["mode"],
                "epsilon": run["target_epsilon"],
            }
            key = _job_key(row)
            if key not in expected_config_hashes:
                continue
            if manifest.get("status") != "completed":
                continue
            if manifest.get("git_sha") != git_sha:
                continue
            if manifest.get("config_hash") != expected_config_hashes[key]:
                continue
            if manifest.get("input_sha256") != input_hashes:
                continue
            completed.add(key)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return completed


def _command(
    row: dict[str, object],
    paths: dict[str, Path],
    output_dir: Path,
    device: str,
) -> list[str]:
    command = [sys.executable, "-m", "src.t2.cli", "run"]
    for city in CITIES:
        command.extend(["--city", f"{city}={paths[city]}"])
    command.extend(
        [
            "--heldout",
            str(row["held_out_city"]),
            "--mode",
            str(row["mode"]),
            "--epsilon",
            "inf" if _epsilon(row["epsilon"]) == float("inf") else str(row["epsilon"]),
            "--seed",
            str(row["seed"]),
            "--rounds",
            "20",
            "--local-epochs",
            "1",
            "--batch-size",
            "256",
            "--learning-rate",
            "0.02",
            "--max-grad-norm",
            "1.0",
            "--patience",
            "5",
            "--epsilon-tolerance",
            "0.05",
            "--device",
            device,
            "--output-dir",
            str(output_dir),
        ]
    )
    return command


def execute_matrix(input_dir: Path, output_dir: Path, device: str) -> None:
    if LEGACY_DIAGNOSTIC_EXECUTION_RETIRED:
        raise RuntimeError(
            "Legacy 20k-per-city Boston/LA diagnostic matrix is retired. "
            "Use the full-data four-fold LOCO protocol only after pre-training gates pass."
        )
    sha, dirty = git_state()
    if sha == "UNKNOWN" or dirty:
        raise RuntimeError("diagnostic matrix requires a clean pinned Git checkout")

    paths = _input_paths(input_dir)
    input_hashes = _input_hashes(paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = diagnostic_plan()
    planned_keys = {_job_key(row) for row in plan}
    completed = _completed_keys(
        output_dir,
        git_sha=sha,
        input_hashes=input_hashes,
        device=device,
    )

    control_path = output_dir / "diagnostic_matrix_control.json"
    control = {
        "protocol": "t2_boston_la_diagnostic_matrix_v1",
        "execution_git_sha": sha,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "device_request": device,
        "input_sha256": input_hashes,
        "planned_jobs": len(plan),
        "completed_before_start": len(completed),
        "status": "running",
    }
    control_path.write_text(
        json.dumps(control, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for row in plan:
        key = _job_key(row)
        if key in completed:
            print(f"SKIP completed {key}", flush=True)
            continue
        command = _command(row, paths, output_dir, device)
        print("RUN " + " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        completed = _completed_keys(
            output_dir,
            git_sha=sha,
            input_hashes=input_hashes,
            device=device,
        )
        if key not in completed:
            raise RuntimeError(
                f"job completed process-wise but failed provenance validation: {key}"
            )

    completed = _completed_keys(
        output_dir,
        git_sha=sha,
        input_hashes=input_hashes,
        device=device,
    )
    if completed != planned_keys:
        missing = sorted(planned_keys - completed, key=str)
        raise RuntimeError(f"diagnostic matrix incomplete; missing {missing}")

    control.update(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_jobs": len(completed),
            "status": "completed",
        }
    )
    control_path.write_text(
        json.dumps(control, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"COMPLETE: {len(completed)} provenance-valid diagnostic jobs", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute/resume the frozen 24-job T2 Boston/LA diagnostic matrix"
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="required safety switch; without it only the frozen plan is printed",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.execute:
        print(json.dumps(diagnostic_plan(), indent=2, default=str))
        print("No training started. Re-run with --execute to authorize the frozen matrix.")
        return 0
    if LEGACY_DIAGNOSTIC_EXECUTION_RETIRED:
        raise SystemExit(
            "Legacy diagnostic execution is retired; no training started."
        )
    execute_matrix(
        Path(args.input_dir).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
        args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
