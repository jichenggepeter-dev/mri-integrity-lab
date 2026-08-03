import numpy as np

from mri_integrity_lab.evaluation import select_balanced_accuracy_threshold
from mri_integrity_lab.inference import ReliabilityState, reliability_state


def test_high_integrity_risk_requires_review() -> None:
    assert reliability_state(integrity_probability=0.81) == ReliabilityState.NEEDS_REVIEW


def test_low_integrity_risk_is_usable_for_research_review() -> None:
    assert reliability_state(integrity_probability=0.19) == ReliabilityState.RESEARCH_READY


def test_borderline_integrity_risk_is_uncertain() -> None:
    assert reliability_state(integrity_probability=0.50) == ReliabilityState.UNCERTAIN


def test_threshold_selection_separates_validation_scores() -> None:
    threshold = select_balanced_accuracy_threshold(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    )

    assert 0.2 < threshold <= 0.8
