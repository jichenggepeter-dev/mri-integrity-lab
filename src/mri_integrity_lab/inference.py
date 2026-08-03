from __future__ import annotations

from enum import StrEnum


class ReliabilityState(StrEnum):
    RESEARCH_READY = "research_ready"
    UNCERTAIN = "uncertain"
    NEEDS_REVIEW = "needs_review"


def reliability_state(
    integrity_probability: float,
    *,
    lower_threshold: float = 0.35,
    upper_threshold: float = 0.65,
) -> ReliabilityState:
    if not 0.0 <= integrity_probability <= 1.0:
        raise ValueError("integrity_probability must be between 0 and 1.")
    if not 0.0 <= lower_threshold < upper_threshold <= 1.0:
        raise ValueError("Reliability thresholds must be ordered within [0, 1].")
    if integrity_probability <= lower_threshold:
        return ReliabilityState.RESEARCH_READY
    if integrity_probability >= upper_threshold:
        return ReliabilityState.NEEDS_REVIEW
    return ReliabilityState.UNCERTAIN
