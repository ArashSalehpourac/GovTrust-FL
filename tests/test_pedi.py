import numpy as np

from src.pedi import privacy_explanation_drift_index, spearman_rank_stability, top_k_overlap


def test_pedi_identical_explanations_are_stable():
    reference = np.array([0.3, 0.2, 0.1, 0.0])
    candidate = reference.copy()

    assert spearman_rank_stability(reference, candidate) == 1.0
    assert privacy_explanation_drift_index(reference, candidate) == 0.0
    assert top_k_overlap(reference, candidate, k=2) == 1.0


def test_pedi_reversed_explanations_show_drift():
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    candidate = reference[::-1]

    assert spearman_rank_stability(reference, candidate) < 0.0
    assert privacy_explanation_drift_index(reference, candidate) > 1.0
