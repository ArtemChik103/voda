"""Tests for submission and raster mask validator (src/metrics/validator.py)."""

import pytest
import pandas as pd
import numpy as np
import tifffile

from src.metrics.validator import (
    validate_submission_csv,
    validate_raster_mask,
    ValidationError,
)
from src.metrics.score import ALL_PAIRS, FLOOD_PAIRS


def test_valid_csv():
    """Valid dataframe passes with no errors."""
    data = []
    for pair in ALL_PAIRS:
        data.append({
            "pair_id": pair,
            "flood_ha": 50.0 if pair in FLOOD_PAIRS else 0.0,
            "water_pre_ha": 300.0,
            "water_peak_ha": 350.0 if pair in FLOOD_PAIRS else 300.0,
        })
    df = pd.DataFrame(data)
    validated = validate_submission_csv(df)
    assert len(validated) == 11
    assert "flood_ha" in validated.columns


def test_missing_column():
    """Missing a required column raises ValidationError."""
    df = pd.DataFrame({"pair_id": ALL_PAIRS, "flood_ha": [0.0] * 11})
    with pytest.raises(ValidationError, match="Missing required column"):
        validate_submission_csv(df)


def test_negative_values():
    """Negative values raise ValidationError."""
    data = [{"pair_id": p, "flood_ha": -5.0, "water_pre_ha": 10.0, "water_peak_ha": 10.0} for p in ALL_PAIRS]
    with pytest.raises(ValidationError, match="Negative values detected"):
        validate_submission_csv(pd.DataFrame(data))


def test_physical_consistency_violation():
    """flood_ha > water_peak_ha violates hydrology physics and raises ValidationError."""
    data = []
    for pair in ALL_PAIRS:
        data.append({
            "pair_id": pair,
            "flood_ha": 500.0,
            "water_pre_ha": 100.0,
            "water_peak_ha": 400.0,  # 500 > 400 violates flood <= peak!
        })
    with pytest.raises(ValidationError, match="Physical violation"):
        validate_submission_csv(pd.DataFrame(data))


def test_raster_mask_validation_passing(tmp_path):
    """GeoTIFF with exact matching area passes validation."""
    # 1000 pixels of 1 = 10 ha (at 10m res, 1 px = 0.01 ha)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:10, 0:100] = 1  # exactly 1000 pixels = 10.0 ha

    tif_file = tmp_path / "test_flood.tif"
    tifffile.imwrite(str(tif_file), mask)

    # Expected area is 10.0 ha -> 0% discrepancy
    res = validate_raster_mask(tif_file, expected_csv_flood_ha=10.0)
    assert res["valid"] is True
    assert res["flood_pixels"] == 1000
    assert res["raster_area_ha"] == 10.0
    assert res["discrepancy_ratio"] == 0.0


def test_raster_mask_discrepancy_tolerance(tmp_path):
    """Discrepancy within 2% passes, discrepancy > 2% fails."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:10, 0:100] = 1  # 1000 pixels = 10.0 ha

    tif_file = tmp_path / "test_flood.tif"
    tifffile.imwrite(str(tif_file), mask)

    # 10.0 vs 10.1 ha -> 1% difference -> passes (< 2%)
    res = validate_raster_mask(tif_file, expected_csv_flood_ha=10.1)
    assert res["valid"] is True

    # 10.0 vs 11.0 ha -> 10% difference -> FAILS (> 2%)
    with pytest.raises(ValidationError, match="exceeding the allowable 2% limit"):
        validate_raster_mask(tif_file, expected_csv_flood_ha=11.0)
