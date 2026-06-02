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


def membership_inference_metrics(member_scores, nonmember_scores) -> dict[str, float | int]:
    """Return a complete confidence-threshold membership-inference result row."""

    member_scores = np.asarray(member_scores, dtype=float)
    nonmember_scores = np.asarray(nonmember_scores, dtype=float)
    attack_auc = membership_inference_auc(member_scores, nonmember_scores)
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    scores = np.concatenate([member_scores, nonmember_scores])
    if scores.size == 0:
        attack_accuracy = float("nan")
    else:
        thresholds = np.quantile(scores, np.linspace(0.05, 0.95, 19))
        attack_accuracy = max(float(((scores >= threshold).astype(int) == labels).mean()) for threshold in thresholds)
    return {
        "attack_accuracy": attack_accuracy,
        "attack_auc": attack_auc,
        "attack_advantage": attack_advantage(attack_auc),
        "member_rows": int(len(member_scores)),
        "nonmember_rows": int(len(nonmember_scores)),
    }
