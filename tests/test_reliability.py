from mri_integrity_lab.inference import ReliabilityState, reliability_state


def test_high_integrity_risk_requires_review() -> None:
    assert reliability_state(integrity_probability=0.81) == ReliabilityState.NEEDS_REVIEW


def test_low_integrity_risk_is_usable_for_research_review() -> None:
    assert reliability_state(integrity_probability=0.19) == ReliabilityState.RESEARCH_READY


def test_borderline_integrity_risk_is_uncertain() -> None:
    assert reliability_state(integrity_probability=0.50) == ReliabilityState.UNCERTAIN
