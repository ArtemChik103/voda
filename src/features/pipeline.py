"""Unified Multi-Modal Feature Extraction Pipeline.

Assembles the 15-channel feature tensor combining:
- Sentinel-1 SAR (6 channels: VV_pre, VH_pre, VV_peak, VH_peak, Delta_VV, Ratio_VH/VV)
- Sentinel-2 MSI (5 channels: MNDWI, NDWI, NDVI, AWEIsh, Cloud_mask)
- Auxiliary layers (4 channels: Slope, Logistic HAND Prior, Permanent Water, Built-up)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np

from src.features.sar import extract_sar_feature_stack
from src.features.optics import extract_optical_feature_stack
from src.features.terrain import extract_aux_feature_stack


FEATURE_CHANNEL_NAMES = [
    "sar_vv_pre_filtered_db",
    "sar_vh_pre_filtered_db",
    "sar_vv_peak_filtered_db",
    "sar_vh_peak_filtered_db",
    "sar_delta_vv_db",
    "sar_ratio_vh_vv_db",
    "optics_mndwi",
    "optics_ndwi",
    "optics_ndvi",
    "optics_aweish",
    "optics_cloud_mask",
    "terrain_slope_deg",
    "terrain_hand_prior",
    "terrain_perm_water_mask",
    "terrain_builtup_mask",
]


def assemble_multimodal_tensor(
    # SAR inputs
    s1_pre_vv: np.ndarray,
    s1_pre_vh: np.ndarray,
    s1_peak_vv: np.ndarray,
    s1_peak_vh: np.ndarray,
    # Optical inputs (optional for 100% cloudy scenes)
    s2_peak_b03: Optional[np.ndarray] = None,
    s2_peak_b04: Optional[np.ndarray] = None,
    s2_peak_b08: Optional[np.ndarray] = None,
    s2_peak_b11: Optional[np.ndarray] = None,
    s2_scl: Optional[np.ndarray] = None,
    # Auxiliary inputs
    hand_meters: Optional[np.ndarray] = None,
    dem_or_slope: Optional[np.ndarray] = None,
    gsw_occurrence_pct: Optional[np.ndarray] = None,
    builtup_layer: Optional[np.ndarray] = None,
    # Processing options
    filter_sar_speckle: bool = True,
    speckle_window_size: int = 7,
    is_slope_already: bool = False,
) -> Tuple[np.ndarray, Dict[str, Union[bool, float, int]]]:
    """Assembles the complete 15-channel multi-modal feature tensor for a scene.

    Args:
        s1_pre_vv, s1_pre_vh: Pre-event SAR VV and VH rasters (dB).
        s1_peak_vv, s1_peak_vh: Peak-event SAR VV and VH rasters (dB).
        s2_peak_b03 ... s2_peak_b11: Sentinel-2 optical bands (Green, Red, NIR, SWIR).
        s2_scl: Sentinel-2 Scene Classification Layer for cloud detection.
        hand_meters: Height Above Nearest Drainage raster in meters.
        dem_or_slope: Copernicus DEM in meters or pre-computed slope.
        gsw_occurrence_pct: JRC Global Surface Water occurrence raster (0-100%).
        builtup_layer: ESA WorldCover or built-up layer.
        filter_sar_speckle: Whether to apply the adaptive Lee filter.
        speckle_window_size: Window dimension for the Lee filter.
        is_slope_already: True if dem_or_slope is already slope in degrees.

    Returns:
        Tuple of (feature_tensor of shape (15, H, W) as float32, metadata dictionary).
    """
    h, w = s1_peak_vv.shape

    # 1. SAR Branch (6 channels)
    sar_stack = extract_sar_feature_stack(
        s1_pre_vv=s1_pre_vv,
        s1_pre_vh=s1_pre_vh,
        s1_peak_vv=s1_peak_vv,
        s1_peak_vh=s1_peak_vh,
        filter_speckle=filter_sar_speckle,
        window_size=speckle_window_size,
    )

    # 2. Optical Branch (5 channels)
    has_optics = all(b is not None for b in [s2_peak_b03, s2_peak_b04, s2_peak_b08, s2_peak_b11])
    if has_optics:
        optics_stack, is_optical_valid, cloud_frac = extract_optical_feature_stack(
            b03_green=s2_peak_b03,
            b04_red=s2_peak_b04,
            b08_nir=s2_peak_b08,
            b11_swir=s2_peak_b11,
            scl_layer=s2_scl,
        )
    else:
        # Graceful fallback: zero optical features, optical marked as invalid (100% cloud)
        optics_stack = np.zeros((5, h, w), dtype=np.float32)
        optics_stack[4, :, :] = 1.0  # entire scene marked as cloud
        is_optical_valid = False
        cloud_frac = 1.0

    # 3. Auxiliary Branch (4 channels)
    hand = hand_meters if hand_meters is not None else np.zeros((h, w), dtype=np.float32)
    dem = dem_or_slope if dem_or_slope is not None else np.zeros((h, w), dtype=np.float32)
    gsw = gsw_occurrence_pct if gsw_occurrence_pct is not None else np.zeros((h, w), dtype=np.float32)

    aux_stack = extract_aux_feature_stack(
        dem_or_slope=dem,
        hand_meters=hand,
        gsw_occurrence_pct=gsw,
        builtup_layer=builtup_layer,
        is_slope_already=is_slope_already,
    )

    # 4. Concatenate along channel axis (6 + 5 + 4 = 15 channels)
    full_tensor = np.concatenate([sar_stack, optics_stack, aux_stack], axis=0).astype(np.float32)

    meta = {
        "num_channels": 15,
        "channel_names": FEATURE_CHANNEL_NAMES,
        "height": h,
        "width": w,
        "is_optical_valid": is_optical_valid,
        "cloud_fraction": cloud_frac,
        "sar_speckle_filtered": filter_sar_speckle,
    }

    return full_tensor, meta
