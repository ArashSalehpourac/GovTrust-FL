"""Privacy accounting and attack-evaluation helpers."""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class DPConfig:
    """Differential privacy knobs for later Opacus experiments."""

    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5


def membership_inference_auc(member_scores, nonmember_scores) -> float:
    """Estimate membership-inference leakage from confidence scores."""

    labels = np.concatenate(
        [
            np.ones(len(member_scores), dtype=int),
            np.zeros(len(nonmember_scores), dtype=int),
        ]
    )
    scores = np.concatenate([member_scores, nonmember_scores])
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def attack_advantage(attack_auc: float) -> float:
    """Convert attack AUC to a simple advantage score."""

    return float(abs(attack_auc - 0.5) * 2)

