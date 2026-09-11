"""Tests for the competition evaluation metric engine (src/metrics/score.py)."""

import pytest
import pandas as pd
import numpy as np

from src.metrics.score import (
    calculate_q_score,
    calculate_spec_base,
    calculate_competition_score,
    ALL_PAIRS,
    FLOOD_PAIRS,
    BASELINE_PAIRS,
    THRESHOLD_FLOOD_HA,
    THRESHOLD_WATER_HA,
)


def test_q_score_exact_match():
    """Identical prediction and ground truth must yield q = 1.0."""
    assert calculate_q_score(150.0, 150.0, THRESHOLD_FLOOD_HA) == 1.0
    assert calculate_q_score(0.0, 0.0, THRESHOLD_FLOOD_HA) == 1.0
    assert calculate_q_score(1200.0, 1200.0, THRESHOLD_WATER_HA) == 1.0


def test_q_score_under_threshold():
    """When true area is below threshold, threshold is used in the denominator."""
    # true = 20, threshold = 50 -> denom = 50. Error = 15 -> q = 1 - 15/50 = 0.70
    assert pytest.approx(calculate_q_score(35.0, 20.0, 50.0), rel=1e-4) == 0.70
    # true = 0, threshold = 50 -> denom = 50. Error = 25 -> q = 1 - 25/50 = 0.50
    assert pytest.approx(calculate_q_score(25.0, 0.0, 50.0), rel=1e-4) == 0.50


def test_q_score_over_threshold():
    """When true area is above threshold, true area is used in the denominator."""
    # true = 500, threshold = 50 -> denom = 500. Error = 50 -> q = 1 - 50/500 = 0.90
    assert pytest.approx(calculate_q_score(450.0, 500.0, 50.0), rel=1e-4) == 0.90
    assert pytest.approx(calculate_q_score(550.0, 500.0, 50.0), rel=1e-4) == 0.90


def test_q_score_clipping():
    """Score must never drop below 0.0 or exceed 1.0."""
    # Error huge -> q should clip to 0.0
    assert calculate_q_score(1000.0, 10.0, 50.0) == 0.0
    assert calculate_q_score(0.0, 1000.0, 50.0) == 0.0


def test_spec_base_zero_excess():
    """When baseline flood prediction has zero excess, Spec_base = 1.0."""
    df_true = pd.DataFrame({"pair_id": BASELINE_PAIRS, "flood_ha": [0.0, 0.0, 0.0]})
    df_pred = pd.DataFrame({"pair_id": BASELINE_PAIRS, "flood_ha": [0.0, 0.0, 0.0]})

    spec_base, specs = calculate_spec_base(df_pred, df_true)
    assert spec_base == 1.0
    for pair in BASELINE_PAIRS:
        assert specs[pair] == 1.0


def test_spec_base_zeroed_by_excess():
    """When baseline false flood exceeds 0.5% (0.005) of AOI area, Spec_base zeroes out."""
    # AOI = 100,000 ha -> 0.5% is 500 ha. Excess of 600 ha must zero spec.
    aoi_areas = {p: 100000.0 for p in BASELINE_PAIRS}
    df_true = pd.DataFrame({"pair_id": BASELINE_PAIRS, "flood_ha": [0.0, 0.0, 0.0]})
    df_pred = pd.DataFrame({"pair_id": BASELINE_PAIRS, "flood_ha": [600.0, 600.0, 600.0]})

    spec_base, specs = calculate_spec_base(df_pred, df_true, aoi_areas_ha=aoi_areas)
    assert spec_base == 0.0
    for pair in BASELINE_PAIRS:
        assert specs[pair] == 0.0


def test_competition_score_perfect():
    """Perfect submission matching ground truth achieves a theoretical 1.00000."""
    gt_records = []
    for pair in ALL_PAIRS:
        is_flood = pair in FLOOD_PAIRS
        gt_records.append({
            "pair_id": pair,
            "flood_ha": 350.0 if is_flood else 0.0,
            "water_pre_ha": 800.0,
            "water_peak_ha": 1150.0 if is_flood else 800.0,
        })
    df_gt = pd.DataFrame(gt_records)
    df_sub = df_gt.copy()

    res = calculate_competition_score(df_sub, df_gt)
    assert res["score"] == 1.0
    assert res["Q_flood"] == 1.0
    assert res["Q_water_peak"] == 1.0
    assert res["Q_water_pre"] == 1.0
    assert res["Spec_base"] == 1.0


def test_competition_score_weights():
    """Verifies the exact weighting formula: 0.45, 0.25, 0.15, 0.15."""
    # Synthetic test where components are set to known distinct values
    gt_records = []
    sub_records = []
    for pair in ALL_PAIRS:
        gt_records.append({
            "pair_id": pair,
            "flood_ha": 100.0 if pair in FLOOD_PAIRS else 0.0,
            "water_pre_ha": 500.0,
            "water_peak_ha": 600.0,
        })
        # flood error = 20 -> q_f = 1 - 20/100 = 0.8
        # peak error = 60 -> q_pk = 1 - 60/600 = 0.9
        # pre error = 100 -> q_pr = 1 - 100/500 = 0.8
        sub_records.append({
            "pair_id": pair,
            "flood_ha": 120.0 if pair in FLOOD_PAIRS else 0.0,
            "water_pre_ha": 600.0,
            "water_peak_ha": 660.0,
        })

    df_gt = pd.DataFrame(gt_records)
    df_sub = pd.DataFrame(sub_records)

    res = calculate_competition_score(df_sub, df_gt)
    expected_score = 0.45 * 0.8 + 0.25 * 0.9 + 0.15 * 0.8 + 0.15 * 1.0
    assert pytest.approx(res["score"], abs=1e-4) == expected_score
