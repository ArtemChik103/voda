"""SAR (Sentinel-1) feature engineering and speckle reduction module.

Implements:
1. Adaptive Lee speckle filter (operating in linear intensity domain).
2. Temporal differential backscatter: Delta sigma0 = sigma0_peak - sigma0_pre.
3. Cross-polarization ratio: Ratio_VH/VV = sigma0_VH - sigma0_VV (in dB).
4. Radar shadow and terrain layover screening.
"""

from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import uniform_filter


def db_to_linear(sigma0_db: np.ndarray) -> np.ndarray:
    """Converts backscatter from decibels (dB) to linear intensity scale."""
    return np.power(10.0, sigma0_db / 10.0)


def linear_to_db(linear_intensity: np.ndarray, min_val: float = 1e-5) -> np.ndarray:
    """Converts linear intensity back to decibels (dB) with numerical stabilization."""
    clipped = np.clip(linear_intensity, min_val, None)
    return 10.0 * np.log10(clipped)


def apply_lee_filter(
    image_db: np.ndarray,
    window_size: int = 7,
    equivalent_looks: float = 4.4,
) -> np.ndarray:
    """Applies the adaptive Lee speckle filter to SAR imagery.

    The filter operates in the physical linear intensity domain to prevent
    logarithmic transformation bias, then converts back to decibels.

    Args:
        image_db: 2D array of backscatter in dB.
        window_size: Odd kernel window size (e.g., 5 or 7).
        equivalent_looks: Equivalent number of independent looks (ENL) for Sentinel-1 IW GRD (~4.4).

    Returns:
        Filtered 2D array in dB with preserved edges and smoothed speckle.
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(f"window_size must be an odd integer >= 3, got {window_size}")

    # Convert dB to linear intensity
    intensity = db_to_linear(image_db.astype(np.float32))

    # Noise variance estimate for multiplicative speckle
    noise_variance = 1.0 / float(equivalent_looks)

    # Local statistics
    local_mean = uniform_filter(intensity, size=window_size)
    local_sq_mean = uniform_filter(intensity * intensity, size=window_size)
    local_variance = np.maximum(0.0, local_sq_mean - local_mean * local_mean)

    # Lee weighting factor: W = max(0, (var_I - mean_I^2 * var_v) / (var_I * (1 + var_v)))
    denominator = local_variance * (1.0 + noise_variance) + 1e-8
    numerator = local_variance - (local_mean * local_mean * noise_variance)
    weight = np.clip(numerator / denominator, 0.0, 1.0)

    # Filtered estimate
    filtered_intensity = local_mean + weight * (intensity - local_mean)

    # Convert back to decibels
    return linear_to_db(filtered_intensity).astype(image_db.dtype)


def compute_temporal_delta(
    sigma0_peak_db: np.ndarray,
    sigma0_pre_db: np.ndarray,
) -> np.ndarray:
    """Computes temporal backscatter drop: Delta sigma0 = sigma0_peak - sigma0_pre (in dB).

    Negative values (e.g. <= -3 dB) indicate a sudden transition from dry rough land
    to smooth specular open water mirror.
    """
    return (sigma0_peak_db.astype(np.float32) - sigma0_pre_db.astype(np.float32))


def compute_polarization_ratio(
    sigma0_vh_db: np.ndarray,
    sigma0_vv_db: np.ndarray,
) -> np.ndarray:
    """Computes cross-polarization ratio: Ratio_VH/VV = sigma0_VH - sigma0_VV (in dB).

    Differentiates flooded vegetation (strong double-bounce in VV) from open calm water.
    """
    return (sigma0_vh_db.astype(np.float32) - sigma0_vv_db.astype(np.float32))


def compute_radar_shadow_mask(
    sigma0_vv_db: np.ndarray,
    slope_deg: Optional[np.ndarray] = None,
    noise_floor_db: float = -24.0,
    slope_threshold_deg: float = 15.0,
) -> np.ndarray:
    """Flags probable radar shadow pixels that mimic open water.

    A pixel is flagged as shadow if its backscatter drops below the sensor noise floor
    on steep terrain slopes.

    Returns:
        Binary mask (uint8) where 1 indicates shadow/layover artifact.
    """
    is_noise_floor = sigma0_vv_db <= noise_floor_db
    if slope_deg is not None:
        is_steep = slope_deg >= slope_threshold_deg
        shadow = is_noise_floor & is_steep
    else:
        shadow = is_noise_floor
    return shadow.astype(np.uint8)


def extract_sar_feature_stack(
    s1_pre_vv: np.ndarray,
    s1_pre_vh: np.ndarray,
    s1_peak_vv: np.ndarray,
    s1_peak_vh: np.ndarray,
    slope_deg: Optional[np.ndarray] = None,
    filter_speckle: bool = True,
    window_size: int = 7,
) -> np.ndarray:
    """Builds a standardized 6-channel SAR feature tensor.

    Channels:
    0: sigma0_vv_pre (filtered, dB)
    1: sigma0_vh_pre (filtered, dB)
    2: sigma0_vv_peak (filtered, dB)
    3: sigma0_vh_peak (filtered, dB)
    4: delta_sigma0_vv = peak - pre (dB)
    5: vh_vv_ratio_peak = vh_peak - vv_peak (dB)

    Returns:
        Tensor of shape (6, H, W) as float32.
    """
    if filter_speckle:
        vv_pre = apply_lee_filter(s1_pre_vv, window_size=window_size)
        vh_pre = apply_lee_filter(s1_pre_vh, window_size=window_size)
        vv_peak = apply_lee_filter(s1_peak_vv, window_size=window_size)
        vh_peak = apply_lee_filter(s1_peak_vh, window_size=window_size)
    else:
        vv_pre = s1_pre_vv.astype(np.float32)
        vh_pre = s1_pre_vh.astype(np.float32)
        vv_peak = s1_peak_vv.astype(np.float32)
        vh_peak = s1_peak_vh.astype(np.float32)

    delta_vv = compute_temporal_delta(vv_peak, vv_pre)
    ratio_peak = compute_polarization_ratio(vh_peak, vv_peak)

    stack = np.stack([
        vv_pre,
        vh_pre,
        vv_peak,
        vh_peak,
        delta_vv,
        ratio_peak,
    ], axis=0).astype(np.float32)

    return stack
