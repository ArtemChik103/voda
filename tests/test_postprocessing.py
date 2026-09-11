"""Unit tests for topological post-processing and submission engine (src/postprocessing/)."""

import pytest
import numpy as np

from src.postprocessing.topology import (
    remove_small_components,
    filter_hydrological_connectivity,
    smooth_waterline_contours,
    postprocess_flood_mask,
)
from src.postprocessing.submission import SubmissionEngine
from src.metrics.score import ALL_PAIRS, FLOOD_PAIRS, BASELINE_PAIRS


def test_remove_small_components():
    """Small isolated islands < 25 pixels must be removed; larger components kept."""
    mask = np.zeros((50, 50), dtype=np.uint8)
    # Component 1: 5x5 = 25 pixels -> KEEP
    mask[10:15, 10:15] = 1
    # Component 2: 2x2 = 4 pixels -> REMOVE
    mask[30:32, 30:32] = 1
    # Component 3: single pixel -> REMOVE
    mask[45, 45] = 1

    cleaned = remove_small_components(mask, min_size_pixels=20)
    assert np.count_nonzero(cleaned[10:15, 10:15]) == 25
    assert np.count_nonzero(cleaned[30:32, 30:32]) == 0
    assert cleaned[45, 45] == 0


def test_river_connectivity():
    """Only flood components connected to the river network should be retained."""
    flood_mask = np.zeros((60, 60), dtype=np.uint8)
    # Flood A: adjacent to river at row 20:30
    flood_mask[20:30, 10:20] = 1
    # Flood B: isolated mountain depression at row 50:58
    flood_mask[50:58, 50:58] = 1

    # River core at row 20:30, col 21:25
    river_core = np.zeros((60, 60), dtype=np.uint8)
    river_core[20:30, 21:25] = 1

    connected = filter_hydrological_connectivity(flood_mask, river_core, max_isolation_distance_px=3)
    assert np.count_nonzero(connected[20:30, 10:20]) > 0
    assert np.count_nonzero(connected[50:58, 50:58]) == 0


def test_submission_engine_spec_base_protection(tmp_path):
    """SubmissionEngine must zero out residual flood on baseline pairs to protect Spec_base."""
    engine = SubmissionEngine(output_dir=tmp_path / "predictions", csv_filename=str(tmp_path / "sub.csv"))

    dummy_mask = np.ones((50, 50), dtype=np.uint8)  # 2500 pixels of flood
    # Baseline pair
    row = engine.process_and_save_pair(
        pair_id="baseline_2018_09_low__blagoveshchensk",
        flood_mask=dummy_mask,
        water_pre_ha=100.0,
        water_peak_ha=100.0,
    )

    # Must be forced to 0.0 ha to guarantee Spec_base = 1.0
    assert row["flood_ha"] == 0.0


def test_full_submission_generation_and_validation(tmp_path):
    """Engine must generate compliant submission.csv and matching GeoTIFFs."""
    engine = SubmissionEngine(output_dir=tmp_path / "predictions", csv_filename=str(tmp_path / "submission.csv"))

    pair_rows = []
    for pair in ALL_PAIRS:
        mask = np.zeros((50, 50), dtype=np.uint8)
        if pair in FLOOD_PAIRS:
            mask[10:20, 10:20] = 1  # 100 px = 1.0 ha
            row = engine.process_and_save_pair(pair, mask, water_pre_ha=50.0, water_peak_ha=51.0)
        else:
            row = engine.process_and_save_pair(pair, mask, water_pre_ha=50.0, water_peak_ha=50.0)
        pair_rows.append(row)

    df, is_valid, msg = engine.generate_submission(pair_rows)
    assert is_valid is True
    assert len(df) == 11
    assert (tmp_path / "submission.csv").exists()
    assert (tmp_path / "predictions" / "flood_2019_07_amur__belogorsk_flood.tif").exists()
