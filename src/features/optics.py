"""Optical (Sentinel-2 MSI) spectral indices and cloud-screening module.

Implements:
1. MNDWI, NDWI, NDVI, and AWEIsh spectral indices.
2. Scene Classification Layer (SCL) cloud/shadow masking.
3. Adaptive optical validity and cloud-fraction estimator.
4. Optical feature stack assembly.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np


def compute_normalized_diff(
    band_a: np.ndarray,
    band_b: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Computes a normalized difference index: (A - B) / (A + B).

    Guarantees output in [-1.0, 1.0] with numerical stability.
    """
    a = band_a.astype(np.float32)
    b = band_b.astype(np.float32)
    denom = a + b
    # Avoid zero division
    denom = np.where(np.abs(denom) < eps, eps, denom)
    diff = (a - b) / denom
    return np.clip(diff, -1.0, 1.0)


def compute_mndwi(b03_green: np.ndarray, b11_swir: np.ndarray) -> np.ndarray:
    """Computes Modified Normalized Difference Water Index (MNDWI).

    Formula: (B03 - B11) / (B03 + B11)
    Strongly suppresses asphalt, concrete, and built-up land compared to NDWI.
    Maintains positive contrast even in turbid sediment-laden waters.
    """
    return compute_normalized_diff(b03_green, b11_swir)


def compute_ndwi(b03_green: np.ndarray, b08_nir: np.ndarray) -> np.ndarray:
    """Computes McFeeters Normalized Difference Water Index (NDWI).

    Formula: (B03 - B08) / (B03 + B08)
    """
    return compute_normalized_diff(b03_green, b08_nir)


def compute_ndvi(b08_nir: np.ndarray, b04_red: np.ndarray) -> np.ndarray:
    """Computes Normalized Difference Vegetation Index (NDVI).

    Formula: (B08 - B04) / (B08 + B04)
    Used to screen high biomass vegetation and separate wet soil from flooded reeds.
    """
    return compute_normalized_diff(b08_nir, b04_red)


def compute_aweish(
    b03_green: np.ndarray,
    b08_nir: np.ndarray,
    b11_swir: np.ndarray,
    b02_blue: Optional[np.ndarray] = None,
    b12_swir2: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Computes Automated Water Extraction Index with shadow suppression (AWEIsh).

    Standard formula:
      AWEIsh = B02 + 2.5 * B03 - 1.5 * (B08 + B11) - 0.25 * B12
    Fallback if B02/B12 omitted:
      AWEIsh = 2.5 * B03 - 1.5 * (B08 + B11)
    """
    g = b03_green.astype(np.float32)
    nir = b08_nir.astype(np.float32)
    swir = b11_swir.astype(np.float32)

    if b02_blue is not None and b12_swir2 is not None:
        blue = b02_blue.astype(np.float32)
        swir2 = b12_swir2.astype(np.float32)
        awei = blue + 2.5 * g - 1.5 * (nir + swir) - 0.25 * swir2
    else:
        awei = 2.5 * g - 1.5 * (nir + swir)

    return awei


def create_cloud_mask_from_scl(scl_layer: np.ndarray) -> np.ndarray:
    """Extracts binary invalid cloud/shadow mask from Sentinel-2 SCL layer.

    SCL Classes to mask:
      3: Cloud shadows
      8: Cloud medium probability
      9: Cloud high probability
      10: Thin cirrus

    Returns:
        Binary mask (uint8) where 1 indicates cloud or cloud shadow.
    """
    scl = scl_layer.astype(np.uint8)
    is_cloud_or_shadow = (
        (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10)
    )
    return is_cloud_or_shadow.astype(np.uint8)


def evaluate_optical_quality(
    cloud_mask: np.ndarray,
    cloud_threshold_ratio: float = 0.70,
) -> Tuple[bool, float]:
    """Evaluates whether the optical scene is usable or completely obscured by cyclone clouds.

    Returns:
        Tuple of (is_optical_valid, cloud_fraction).
    """
    total_pixels = cloud_mask.size
    cloud_pixels = int(np.count_nonzero(cloud_mask))
    cloud_fraction = float(cloud_pixels / max(total_pixels, 1))

    is_valid = cloud_fraction <= cloud_threshold_ratio
    return is_valid, round(cloud_fraction, 4)


def extract_optical_feature_stack(
    b03_green: np.ndarray,
    b04_red: np.ndarray,
    b08_nir: np.ndarray,
    b11_swir: np.ndarray,
    scl_layer: Optional[np.ndarray] = None,
    b02_blue: Optional[np.ndarray] = None,
    b12_swir2: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool, float]:
    """Builds a standardized 5-channel Optical feature stack.

    Channels:
    0: MNDWI
    1: NDWI
    2: NDVI
    3: AWEIsh
    4: Cloud_mask (1 = invalid/cloud, 0 = clear sky)

    Returns:
        Tuple of (feature_stack of shape (5, H, W), is_optical_valid, cloud_fraction).
    """
    mndwi = compute_mndwi(b03_green, b11_swir)
    ndwi = compute_ndwi(b03_green, b08_nir)
    ndvi = compute_ndvi(b08_nir, b04_red)
    aweish = compute_aweish(b03_green, b08_nir, b11_swir, b02_blue, b12_swir2)

    if scl_layer is not None:
        cloud_mask = create_cloud_mask_from_scl(scl_layer)
    else:
        # Heuristic cloud mask based on high reflectance across visible and NIR
        high_refl = (b04_red.astype(np.float32) > 0.35) & (b08_nir.astype(np.float32) > 0.35)
        cloud_mask = high_refl.astype(np.uint8)

    is_valid, cloud_frac = evaluate_optical_quality(cloud_mask)

    stack = np.stack([
        mndwi,
        ndwi,
        ndvi,
        aweish,
        cloud_mask.astype(np.float32),
    ], axis=0).astype(np.float32)

    return stack, is_valid, cloud_frac
