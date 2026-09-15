from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

REQUIRED_MANIFEST_FIELDS = {
    "run_uuid",
    "git_sha",
    "git_dirty",
    "config_hash",
    "started_at_utc",
    "completed_at_utc",
    "status",
    "fold",
    "held_out_city",
    "source_cities",
    "input_sha256",
    "row_counts",
    "date_ranges",
    "preprocessor_fingerprint",
    "features",
    "seed",
    "training",
    "privacy",
    "runtime_seconds",
    "hardware",
    "packages",
    "command",
    "expected_outputs",
}


def canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> tuple[str, bool]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ).strip()
        )
        return sha, dirty
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return "UNKNOWN", True


def package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "pyarrow", "sklearn", "torch", "opacus"]
    out: dict[str, str] = {"python": platform.python_version()}
    for name in names:
        try:
            module = __import__(name)
            out[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            out[name] = "unavailable"
    return out


def hardware_info() -> dict[str, object]:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available": cuda_available,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_count": device_count,
        "cuda_device_name_0": (
            torch.cuda.get_device_name(0) if cuda_available and device_count > 0 else None
        ),
    }


def new_manifest_skeleton(
    *,
    config: Mapping[str, object],
    fold: str,
    held_out_city: str,
    source_cities: list[str],
    command: str,
) -> dict[str, object]:
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
        "hardware": hardware_info(),
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
        raise TypeError("privacy ledger must be a mapping")
