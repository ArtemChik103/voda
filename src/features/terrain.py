"""Geomorphological and hydrological auxiliary feature module (AUX stack).

Implements:
1. Continuous Logistic HAND Prior (avoids step artifacts on alluvial terraces).
2. DEM-based Slope gradient computation (Sobel operator).
3. Permanent water masking from JRC Global Surface Water (GSW).
4. Urban built-up and road infrastructure suppression.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import sobel, zoom


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


def load_and_resample_aux(
    aux_source: Union[str, Path, np.ndarray],
    target_shape: Tuple[int, int],
    target_transform: Optional[Any] = None,
    target_crs: Optional[Any] = None,
) -> Dict[str, np.ndarray]:
    """Reads 6-channel AUX_terrain_gsw stack (30m) and resamples to target 10m grid.

    Bands:
        1: slope (Copernicus DEM GLO-30) - continuous -> bilinear resampling
        2: hand (MERIT Hydro) - continuous -> bilinear resampling
        3: occurrence (JRC GSW v1.4) - percentage [0, 100] -> nearest neighbor
        4: seasonality (JRC GSW v1.4) - months [0, 12] -> nearest neighbor
        5: max_extent (JRC GSW v1.4) - binary [0, 1] -> nearest neighbor
        6: builtup (ESA WorldCover) - binary [0, 1] -> nearest neighbor

    Args:
        aux_source: Path to AUX_terrain_gsw.tif or in-memory 3D array of shape (6, H_src, W_src).
        target_shape: Target (H, W) grid dimensions.
        target_transform: Affine transform of target 10m grid. If None, derived from source bounds.
        target_crs: Target coordinate reference system.

    Returns:
        Dict mapping band names to 2D numpy arrays (H, W) of float32.
    """
    H_tgt, W_tgt = target_shape

    # Branch 1: In-memory numpy array (shape: 6, H_src, W_src)
    if isinstance(aux_source, np.ndarray):
        if aux_source.ndim != 3 or aux_source.shape[0] < 6:
            raise ValueError(f"Expected aux_source array with shape (6, H, W), got {aux_source.shape}")
        
        H_src, W_src = aux_source.shape[1], aux_source.shape[2]
        zoom_factors = (H_tgt / H_src, W_tgt / W_src)

        band_specs = [
            ("slope", 0, 1, 0.0, 90.0),
            ("hand", 1, 1, 0.0, 5000.0),
            ("occurrence", 2, 0, 0.0, 100.0),
            ("seasonality", 3, 0, 0.0, 12.0),
            ("max_extent", 4, 0, 0.0, 1.0),
            ("builtup", 5, 0, 0.0, 1.0),
        ]
        resampled = {}
        for name, c_idx, order, val_min, val_max in band_specs:
            arr_zoom = zoom(aux_source[c_idx].astype(np.float32), zoom_factors, order=order)
            if arr_zoom.shape != target_shape:
                cur_h, cur_w = arr_zoom.shape
                cropped = np.zeros(target_shape, dtype=np.float32)
                h_end = min(cur_h, H_tgt)
                w_end = min(cur_w, W_tgt)
                cropped[:h_end, :w_end] = arr_zoom[:h_end, :w_end]
                arr_zoom = cropped
            arr_zoom = np.nan_to_num(arr_zoom, nan=val_min, posinf=val_max, neginf=val_min)
            arr_zoom = np.clip(arr_zoom, val_min, val_max)
            resampled[name] = arr_zoom.astype(np.float32)
        return resampled

    # Branch 2: Filepath (GeoTIFF)
    aux_path = Path(aux_source)
    if not aux_path.exists():
        raise FileNotFoundError(f"AUX file not found: {aux_path}")

    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds

    with rasterio.open(str(aux_path)) as src:
        dst_crs = target_crs if target_crs is not None else src.crs
        dst_transform = target_transform
        if dst_transform is None:
            dst_transform = from_bounds(
                src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top,
                W_tgt, H_tgt
            )

        band_specs = [
            ("slope", 1, Resampling.bilinear, 0.0, 90.0),
            ("hand", 2, Resampling.bilinear, 0.0, 5000.0),
            ("occurrence", 3, Resampling.nearest, 0.0, 100.0),
            ("seasonality", 4, Resampling.nearest, 0.0, 12.0),
            ("max_extent", 5, Resampling.nearest, 0.0, 1.0),
            ("builtup", 6, Resampling.nearest, 0.0, 1.0),
        ]

        resampled = {}
        for name, b_idx, resampling_mode, val_min, val_max in band_specs:
            dst_arr = np.zeros(target_shape, dtype=np.float32)
            reproject(
                source=rasterio.band(src, b_idx),
                destination=dst_arr,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=resampling_mode,
            )
            dst_arr = np.nan_to_num(dst_arr, nan=val_min, posinf=val_max, neginf=val_min)
            dst_arr = np.clip(dst_arr, val_min, val_max)
            resampled[name] = dst_arr.astype(np.float32)

        return resampled
