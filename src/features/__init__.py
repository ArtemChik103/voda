"""Features engineering package for KosmoHackathon 2026."""

from src.features.sar import (
    apply_lee_filter,
    compute_temporal_delta,
    compute_polarization_ratio,
    compute_radar_shadow_mask,
    extract_sar_feature_stack,
    db_to_linear,
    linear_to_db,
)
from src.features.optics import (
    compute_mndwi,
    compute_ndwi,
    compute_ndvi,
    compute_aweish,
    create_cloud_mask_from_scl,
    evaluate_optical_quality,
    extract_optical_feature_stack,
)
from src.features.terrain import (
    compute_logistic_hand_prior,
    compute_slope_degrees,
    create_permanent_water_mask,
    create_builtup_mask,
    extract_aux_feature_stack,
)
from src.features.tiling import (
    create_2d_hann_window,
    TileStitcher,
    extract_tiles,
)
from src.features.pipeline import (
    assemble_multimodal_tensor,
    FEATURE_CHANNEL_NAMES,
)

__all__ = [
    "apply_lee_filter",
    "compute_temporal_delta",
    "compute_polarization_ratio",
    "compute_radar_shadow_mask",
    "extract_sar_feature_stack",
    "db_to_linear",
    "linear_to_db",
    "compute_mndwi",
    "compute_ndwi",
    "compute_ndvi",
    "compute_aweish",
    "create_cloud_mask_from_scl",
    "evaluate_optical_quality",
    "extract_optical_feature_stack",
    "compute_logistic_hand_prior",
    "compute_slope_degrees",
    "create_permanent_water_mask",
    "create_builtup_mask",
    "extract_aux_feature_stack",
    "create_2d_hann_window",
    "TileStitcher",
    "extract_tiles",
    "assemble_multimodal_tensor",
    "FEATURE_CHANNEL_NAMES",
]
