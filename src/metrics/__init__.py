"""Metrics module for KosmoHackathon 2026."""

from src.metrics.score import (
    calculate_competition_score,
    calculate_q_score,
    calculate_spec_base,
    FLOOD_PAIRS,
    BASELINE_PAIRS,
    ALL_PAIRS,
)
from src.metrics.validator import (
    validate_submission_csv,
    validate_raster_masks,
    verify_consistency,
)

__all__ = [
    "calculate_competition_score",
    "calculate_q_score",
    "calculate_spec_base",
    "validate_submission_csv",
    "validate_raster_masks",
    "verify_consistency",
    "FLOOD_PAIRS",
    "BASELINE_PAIRS",
    "ALL_PAIRS",
]
