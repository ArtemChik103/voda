"""Geomorphological and hydrological auxiliary feature module (AUX stack).

Implements:
1. Continuous Logistic HAND Prior (avoids step artifacts on alluvial terraces).
2. DEM-based Slope gradient computation (Sobel operator).
3. Permanent water masking from JRC Global Surface Water (GSW).
4. Urban built-up and road infrastructure suppression.
"""

from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import sobel


def compute_logistic_hand_prior(
    hand_meters: np.ndarray,
    h0_center_meters: float = 12.0,
    temperature_tau: float = 3.0,
) -> np.ndarray:
    """Computes a continuous physical prior probability of flood inundation based on HAND.

    Formula:
        P(flood | HAND) = 1 / (1 + exp((HAND - H0) / tau))

    - At HAND = 0 m (riverbed/floodplain): P ~ 0.98
    - At HAND = 12 m (terrace transition): P = 0.50
    - At HAND > 22 m (valley hillside): P < 0.03

    Prevents contour-line staircase artifacts caused by hard cutoffs at 25 m.
    """
    clipped_hand = np.clip(hand_meters.astype(np.float32), 0.0, 100.0)
    z = (clipped_hand - float(h0_center_meters)) / float(temperature_tau)
    # Clip z to prevent float overflow
    z_clipped = np.clip(z, -20.0, 20.0)
    prob = 1.0 / (1.0 + np.exp(z_clipped))
    return prob.astype(np.float32)


def compute_slope_degrees(
    dem_meters: np.ndarray,
    pixel_size_meters: float = 10.0,
) -> np.ndarray:
    """Computes terrain slope in degrees using 2D spatial gradients (Sobel operator).

    Args:
        dem_meters: 2D array of Digital Elevation Model in meters.
        pixel_size_meters: Horizontal grid resolution (default 10 m).

    Returns:
        2D array of slope in degrees [0, 90].
    """
    dem = dem_meters.astype(np.float32)
    # Sobel operator has distance factor of 8 * pixel_size
    scale = 8.0 * float(pixel_size_meters)
    dz_dx = sobel(dem, axis=1) / scale
    dz_dy = sobel(dem, axis=0) / scale

    gradient = np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy)
    slope_rad = np.arctan(gradient)
    slope_deg = np.degrees(slope_rad)
    return slope_deg.astype(np.float32)


def create_permanent_water_mask(
    gsw_occurrence_pct: np.ndarray,
    threshold_pct: float = 80.0,
) -> np.ndarray:
    """Identifies permanent water bodies that must be excluded from 'flood' class.

    Args:
        gsw_occurrence_pct: JRC Global Surface Water occurrence [0, 100].
        threshold_pct: Occurrence threshold (default >= 80%).

    Returns:
        Binary mask (uint8) where 1 indicates permanent water.
    """
    perm = gsw_occurrence_pct.astype(np.float32) >= float(threshold_pct)
    return perm.astype(np.uint8)


def create_builtup_mask(
    worldcover_classes: np.ndarray,
    builtup_class_code: int = 50,
) -> np.ndarray:
    """Flags artificial built-up and paved surfaces that cause specular SAR reflection.

    Args:
        worldcover_classes: ESA WorldCover raster (50 = Built-up).
        builtup_class_code: Code representing urban/paved land (default 50).

    Returns:
        Binary mask (uint8) where 1 indicates built-up land.
    """
    is_builtup = worldcover_classes == builtup_class_code
    return is_builtup.astype(np.uint8)


def extract_aux_feature_stack(
    dem_or_slope: np.ndarray,
    hand_meters: np.ndarray,
    gsw_occurrence_pct: np.ndarray,
    builtup_layer: Optional[np.ndarray] = None,
    is_slope_already: bool = False,
) -> np.ndarray:
    """Builds a standardized 4-channel Auxiliary feature stack.

    Channels:
    0: Slope (degrees)
    1: Logistic HAND flood prior [0, 1]
    2: GSW permanent water mask (0 or 1)
    3: Built-up mask (0 or 1)

    Returns:
        Tensor of shape (4, H, W) as float32.
    """
    if is_slope_already:
        slope_deg = dem_or_slope.astype(np.float32)
    else:
        slope_deg = compute_slope_degrees(dem_or_slope)

    hand_prior = compute_logistic_hand_prior(hand_meters)
    perm_water = create_permanent_water_mask(gsw_occurrence_pct)

    if builtup_layer is not None:
        builtup = create_builtup_mask(builtup_layer)
    else:
        builtup = np.zeros_like(perm_water, dtype=np.uint8)

    stack = np.stack([
        slope_deg,
        hand_prior,
        perm_water.astype(np.float32),
        builtup.astype(np.float32),
    ], axis=0).astype(np.float32)

    return stack
