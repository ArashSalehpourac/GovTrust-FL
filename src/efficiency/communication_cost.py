"""Federated communication-cost estimates."""

from __future__ import annotations


def communication_summary(model_size_bytes: int, num_clients: int, num_rounds: int) -> dict[str, float]:
    """Estimate upload, download, and total communication costs."""

    upload = model_size_bytes * num_clients * num_rounds
    download = model_size_bytes * num_clients * num_rounds
    total = upload + download
    return {
        "upload_bytes": float(upload),
        "download_bytes": float(download),
        "total_communication_bytes": float(total),
        "total_communication_mb": float(total / (1024 * 1024)),
    }
