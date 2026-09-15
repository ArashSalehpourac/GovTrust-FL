from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

CITIES = ("nyc", "chicago", "boston", "los_angeles")
EPSILONS = (float("inf"), 5.0, 1.0)
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class T2Config:
    algorithm: Literal["fedavg"] = "fedavg"
    mode: Literal["nonprivate", "private", "clipped_no_noise"] = "nonprivate"
    target_epsilon: float = float("inf")
    rounds: int = 20
    local_epochs: int = 1
    batch_size: int = 256
    learning_rate: float = 0.02
    hidden_sizes: tuple[int, ...] = (64, 32)
    seed: int = 0
    max_grad_norm: float = 1.0
    checkpoint_patience: int = 5
    stop_on_patience: bool = False
    epsilon_tolerance: float = 0.05

    def as_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["hidden_sizes"] = list(self.hidden_sizes)
        return out

    def validate(self) -> None:
        if self.algorithm != "fedavg":
            raise ValueError("T2 inferential runs currently permit FedAvg only")
        if self.mode == "private" and not (self.target_epsilon > 0 and self.target_epsilon < float("inf")):
            raise ValueError("private mode requires finite positive target_epsilon")
        if self.mode != "private" and self.target_epsilon != float("inf"):
            raise ValueError("non-private controls must use epsilon=infinity")
        if self.rounds < 1 or self.local_epochs < 1 or self.batch_size < 1:
            raise ValueError("invalid training schedule")


def diagnostic_plan() -> list[dict[str, object]]:
    """Frozen first diagnostic: Boston/LA, eps inf/5/1, seeds 0/1/2."""
    rows: list[dict[str, object]] = []
    for held_out in ("boston", "los_angeles"):
        for seed in SEEDS:
            rows.append({"held_out_city": held_out, "seed": seed, "mode": "nonprivate", "epsilon": float("inf")})
            rows.append({"held_out_city": held_out, "seed": seed, "mode": "private", "epsilon": 5.0})
            rows.append({"held_out_city": held_out, "seed": seed, "mode": "private", "epsilon": 1.0})
            rows.append({"held_out_city": held_out, "seed": seed, "mode": "clipped_no_noise", "epsilon": float("inf")})
    return rows


def delta_for_client(n_rows: int) -> float:
    if n_rows < 2:
        raise ValueError("client requires at least 2 rows")
    delta = min(1e-5, 0.1 / float(n_rows))
    if not 0.0 < delta < 1.0 / n_rows:
        raise AssertionError("delta rule must satisfy delta < 1/N")
    return delta
