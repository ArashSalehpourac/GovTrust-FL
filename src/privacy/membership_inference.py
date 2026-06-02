"""Membership-inference attack utilities."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def membership_inference_auc(member_scores, nonmember_scores) -> float:
    """Estimate membership-inference leakage from confidence or loss scores."""

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
