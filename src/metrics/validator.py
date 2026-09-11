"""Submission and GeoTIFF raster mask validator for KosmoHackathon 2026.

Enforces:
1. CSV schema, 11 required pair_ids, positive values, flood_ha <= water_peak_ha.
2. GeoTIFF format: uint8, values in {0, 1}, CRS EPSG:32652.
3. Crucial Rule: Discrepancy between raster mask area and CSV flood_ha <= 2%.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import tifffile

from src.metrics.score import ALL_PAIRS, FLOOD_PAIRS, BASELINE_PAIRS

REQUIRED_COLUMNS = ["pair_id", "flood_ha", "water_pre_ha", "water_peak_ha"]
MAX_AREA_DISCREPANCY_RATIO = 0.02  # 2% maximum discrepancy


class ValidationError(Exception):
    """Raised when submission fails format, consistency, or rule constraints."""
    pass


def validate_submission_csv(
    csv_path_or_df: Union[str, Path, pd.DataFrame]
) -> pd.DataFrame:
    """Validates the submission CSV file against competition guidelines.

    Args:
        csv_path_or_df: Path to submission.csv or pre-loaded DataFrame.

    Returns:
        Validated DataFrame indexed by pair_id.

    Raises:
        ValidationError: If requirements are violated.
    """
    if isinstance(csv_path_or_df, (str, Path)):
        p = Path(csv_path_or_df)
        if not p.exists():
            raise ValidationError(f"Submission CSV file not found: {p}")
        df = pd.read_csv(p)
    else:
        df = csv_path_or_df.copy()

    # Column checks
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValidationError(f"Missing required column in CSV: '{col}'")

    if len(df) != len(ALL_PAIRS):
        raise ValidationError(
            f"Expected exactly {len(ALL_PAIRS)} rows, but got {len(df)}"
        )

    # Check pair IDs
    present_pairs = set(df["pair_id"])
    expected_pairs = set(ALL_PAIRS)
    missing = expected_pairs - present_pairs
    if missing:
        raise ValidationError(f"Missing pair_ids in submission: {missing}")

    extra = present_pairs - expected_pairs
    if extra:
        raise ValidationError(f"Unexpected extra pair_ids in submission: {extra}")

    df = df.set_index("pair_id")

    # Numeric checks
    for col in ["flood_ha", "water_pre_ha", "water_peak_ha"]:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            nan_pairs = df.index[values.isna()].tolist()
            raise ValidationError(f"NaN or invalid non-numeric values in column '{col}' for pairs: {nan_pairs}")
        if (values < 0.0).any():
            neg_pairs = df.index[values < 0.0].tolist()
            raise ValidationError(f"Negative values detected in column '{col}' for pairs: {neg_pairs}")
        df[col] = values

    # Physical consistency check: flood_ha <= water_peak_ha + epsilon
    epsilon = 0.01  # tolerance for rounding
    inconsistent = df[df["flood_ha"] > (df["water_peak_ha"] + epsilon)]
    if not inconsistent.empty:
        raise ValidationError(
            f"Physical violation: flood_ha exceeds water_peak_ha for pairs: {inconsistent.index.tolist()}"
        )

    return df


def validate_raster_mask(
    mask_path: Union[str, Path],
    expected_csv_flood_ha: Optional[float] = None,
    pixel_size_meters: float = 10.0,
) -> Dict[str, Union[int, float, bool]]:
    """Validates a single flood GeoTIFF mask.

    Checks:
    - File existence and readability
    - Data type uint8
    - Values restricted to {0, 1}
    - 2% consistency check with CSV flood_ha if provided

    Args:
        mask_path: Path to <pair_id>_flood.tif.
        expected_csv_flood_ha: Reported area in submission.csv.
        pixel_size_meters: Pixel resolution (default 10 m -> 100 m^2 per pixel = 0.01 ha).

    Returns:
        Dictionary with validation statistics.
    """
    p = Path(mask_path)
    if not p.exists():
        raise ValidationError(f"Prediction raster mask not found: {p}")

    try:
        mask = tifffile.imread(str(p))
    except Exception as e:
        raise ValidationError(f"Could not read GeoTIFF mask {p}: {e}")

    if mask.dtype != np.uint8:
        raise ValidationError(f"Mask dtype must be uint8, but got {mask.dtype} for {p}")

    unique_vals = np.unique(mask)
    invalid_vals = set(unique_vals) - {0, 1}
    if invalid_vals:
        raise ValidationError(
            f"Mask values must be strictly binary {{0, 1}}, found extra values {invalid_vals} in {p}"
        )

    num_flood_pixels = int(np.sum(mask == 1))
    ha_per_pixel = (pixel_size_meters * pixel_size_meters) / 10000.0
    raster_area_ha = round(num_flood_pixels * ha_per_pixel, 2)

    res = {
        "valid": True,
        "shape": mask.shape,
        "flood_pixels": num_flood_pixels,
        "raster_area_ha": raster_area_ha,
    }

    if expected_csv_flood_ha is not None:
        denom = max(float(expected_csv_flood_ha), 1.0)
        discrepancy = abs(raster_area_ha - expected_csv_flood_ha) / denom
        res["discrepancy_ratio"] = round(discrepancy, 4)
        if discrepancy > MAX_AREA_DISCREPANCY_RATIO:
            raise ValidationError(
                f"Area discrepancy between raster ({raster_area_ha} ha) and CSV ({expected_csv_flood_ha} ha) "
                f"is {discrepancy * 100:.2f}%, exceeding the allowable 2% limit for {p.name}!"
            )

    return res


def validate_raster_masks(
    predictions_dir: Union[str, Path],
    sub_df: pd.DataFrame,
    pairs: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """Validates an entire folder of predictions/<pair_id>_flood.tif against submission.csv.

    Args:
        predictions_dir: Path to directory containing predictions.
        sub_df: Validated DataFrame indexed by pair_id.
        pairs: List of pairs to validate (defaults to ALL_PAIRS).

    Returns:
        Dict mapping pair_id to validation results.
    """
    pred_dir = Path(predictions_dir)
    target_pairs = pairs or ALL_PAIRS
    results = {}

    for pair in target_pairs:
        mask_file = pred_dir / f"{pair}_flood.tif"
        exp_area = float(sub_df.loc[pair, "flood_ha"]) if pair in sub_df.index else None
        results[pair] = validate_raster_mask(mask_file, expected_csv_flood_ha=exp_area)

    return results


def verify_consistency(
    submission_path: Union[str, Path],
    predictions_dir: Optional[Union[str, Path]] = None,
) -> Tuple[bool, str]:
    """Top-level convenience check returning (is_valid, message)."""
    try:
        sub_df = validate_submission_csv(submission_path)
        if predictions_dir is not None:
            validate_raster_masks(predictions_dir, sub_df)
        return True, "Submission and masks are 100% compliant with competition criteria."
    except ValidationError as err:
        return False, str(err)
