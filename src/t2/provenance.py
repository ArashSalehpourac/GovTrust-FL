from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

REQUIRED_MANIFEST_FIELDS = {
    "run_uuid", "git_sha", "git_dirty", "config_hash", "started_at_utc",
    "completed_at_utc", "status", "fold", "held_out_city", "source_cities",
    "input_sha256", "row_counts", "date_ranges", "preprocessor_fingerprint",
    "features", "seed", "training", "privacy", "runtime_seconds", "hardware",
    "packages", "command", "expected_outputs",
}


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        return sha, dirty
    except Exception:
        return "UNKNOWN", True


def package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "sklearn", "torch", "opacus"]
    out: dict[str, str] = {"python": platform.python_version()}
    for name in names:
        try:
            module = __import__(name)
            out[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            out[name] = "unavailable"
    return out


def new_manifest_skeleton(*, config: Mapping[str, object], fold: str, held_out_city: str, source_cities: list[str], command: str) -> dict[str, object]:
    sha, dirty = git_state()
    return {
        "run_uuid": str(uuid.uuid4()),
        "git_sha": sha,
        "git_dirty": dirty,
        "config_hash": canonical_hash(dict(config)),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_at_utc": None,
        "status": "running",
        "fold": fold,
        "held_out_city": held_out_city,
        "source_cities": source_cities,
        "input_sha256": {},
        "row_counts": {},
        "date_ranges": {},
        "preprocessor_fingerprint": None,
        "features": {},
        "seed": config.get("seed"),
        "training": {},
        "privacy": {},
        "runtime_seconds": None,
        "hardware": {"platform": platform.platform(), "processor": platform.processor(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "packages": package_versions(),
        "command": command,
        "expected_outputs": ["checkpoint.pt", "run.json", "manifest.json"],
    }


def validate_completed_manifest(manifest: Mapping[str, object]) -> None:
    missing = REQUIRED_MANIFEST_FIELDS - set(manifest)
    if missing:
        raise ValueError(f"manifest missing fields: {sorted(missing)}")
    if manifest.get("status") != "completed":
        raise ValueError("run is not completed")
    if manifest.get("git_sha") in {None, "", "UNKNOWN"}:
        raise ValueError("missing executable git SHA")
    if manifest.get("git_dirty") is not False:
        raise ValueError("real result must come from a clean Git worktree")
    if not manifest.get("config_hash") or not manifest.get("preprocessor_fingerprint"):
        raise ValueError("missing config/preprocessor fingerprint")
    privacy = manifest.get("privacy")
    if not isinstance(privacy, dict):
        raise ValueError("privacy ledger missing")
