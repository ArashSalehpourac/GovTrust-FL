"""Runtime and memory measurement helpers."""

from __future__ import annotations

from pathlib import Path

import psutil


def measure_peak_memory_mb() -> float:
    """Return current process resident memory in megabytes."""

    process = psutil.Process()
    return float(process.memory_info().rss / (1024 * 1024))


def model_file_size_mb(path: str | Path) -> float:
    """Return a serialized model file size in megabytes."""

    return float(Path(path).stat().st_size / (1024 * 1024))
