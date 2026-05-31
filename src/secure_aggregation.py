"""Secure-aggregation simulation hooks.

This module intentionally implements a simulation-layer abstraction only. It is
not production cryptography and does not claim to provide deployed secure
aggregation. The goal is to estimate aggregation behavior and communication
overhead for paper experiments until a real cryptographic protocol is wired in.
"""

from __future__ import annotations

import numpy as np


def simulated_secure_aggregate(
    client_updates: list[tuple[list[np.ndarray], int]],
    *,
    overhead_factor: float = 0.25,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Aggregate client updates and return simulated communication overhead."""

    total_examples = sum(num_examples for _, num_examples in client_updates)
    if total_examples <= 0:
        raise ValueError("Cannot aggregate zero client examples")

    aggregated = []
    for values in zip(*(params for params, _ in client_updates), strict=True):
        weighted = sum(
            value * (num_examples / total_examples)
            for value, (_, num_examples) in zip(values, client_updates, strict=True)
        )
        aggregated.append(weighted.astype(np.float32, copy=False))

    update_bytes = sum(value.nbytes for value in aggregated)
    overhead_bytes = update_bytes * len(client_updates) * overhead_factor
    return aggregated, {
        "secure_aggregation_simulated": True,
        "secure_aggregation_overhead_bytes": float(overhead_bytes),
    }


def plain_fedavg(client_updates: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    """Weighted FedAvg aggregation without privacy-layer overhead."""

    aggregated, _ = simulated_secure_aggregate(client_updates, overhead_factor=0.0)
    return aggregated
