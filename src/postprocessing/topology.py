"""Topological post-processing, noise suppression, and hydrological connectivity module.

Implements:
1. Small noise component removal: purges isolated specks < 25 pixels (0.25 ha) per official guidelines.
2. Geodesic connectivity filtering: connects flood inundation to core river drainage networks.
3. Morphological shoreline contour smoothing.
"""

from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import label, generate_binary_structure, binary_opening, binary_closing, binary_dilation


def remove_small_components(
    binary_mask: np.ndarray,
    min_size_pixels: int = 25,
) -> np.ndarray:
    """Removes disconnected speckle islands smaller than min_size_pixels (default 25 = 0.25 ha).

    Args:
        binary_mask: Binary 2D array (0 or 1).
        min_size_pixels: Minimum allowable component size in pixels (2500 m^2 at 10m).

    Returns:
        Filtered binary mask with small isolated noise patches removed.
    """
    if np.count_nonzero(binary_mask) == 0:
        return binary_mask.copy()

    structure = generate_binary_structure(2, 2)  # 8-connectivity
    labeled_array, num_features = label(binary_mask, structure=structure)

    if num_features == 0:
        return binary_mask.copy()

    # Count pixel area for each connected component
    counts = np.bincount(labeled_array.ravel())
    # Identify labels exceeding min_size_pixels (ignoring background label 0)
    keep_mask = counts >= min_size_pixels
    keep_mask[0] = False  # background is always excluded

    # Reconstruct filtered binary mask
    filtered = keep_mask[labeled_array].astype(np.uint8)
    return filtered


def filter_hydrological_connectivity(
    flood_mask: np.ndarray,
    river_core_mask: np.ndarray,
    max_isolation_distance_px: int = 5,
) -> np.ndarray:
    """Retains only flood polygons hydrologically connected or adjacent to the river network.

    Prevents spurious mountain depressions from being classified as river flood inundation.

    Args:
        flood_mask: Candidate flood mask.
        river_core_mask: Permanent river channel / water pre mask.
        max_isolation_distance_px: Maximum gap (in 10m pixels) allowed across embankments.

    Returns:
        Hydrologically connected flood mask.
    """
    if np.count_nonzero(flood_mask) == 0 or np.count_nonzero(river_core_mask) == 0:
        return flood_mask.copy()

    structure = generate_binary_structure(2, 2)
    labeled_flood, num_features = label(flood_mask, structure=structure)

    if num_features == 0:
        return flood_mask.copy()

    # Dilation of river core to bridge narrow dikes/embankments
    selem = generate_binary_structure(2, 1)
    expanded_river = binary_dilation(river_core_mask, structure=selem, iterations=max_isolation_distance_px)

    # Find connected components that intersect the expanded river network
    touching_labels = np.unique(labeled_flood[expanded_river == 1])
    touching_labels = touching_labels[touching_labels != 0]

    if len(touching_labels) == 0:
        return np.zeros_like(flood_mask, dtype=np.uint8)

    keep_table = np.zeros(num_features + 1, dtype=bool)
    keep_table[touching_labels] = True

    connected_flood = keep_table[labeled_flood].astype(np.uint8)
    return connected_flood


def smooth_waterline_contours(
    binary_mask: np.ndarray,
) -> np.ndarray:
    """Applies morphological opening/closing to smooth jagged staircases along shorelines."""
    if np.count_nonzero(binary_mask) == 0:
        return binary_mask.copy()

    selem = generate_binary_structure(2, 1)  # 4-connectivity cross
    opened = binary_opening(binary_mask, structure=selem)
    closed = binary_closing(opened, structure=selem)
    return closed.astype(np.uint8)


def postprocess_flood_mask(
    raw_flood_mask: np.ndarray,
    river_core_mask: Optional[np.ndarray] = None,
    min_size_pixels: int = 25,
    enforce_connectivity: bool = True,
) -> np.ndarray:
    """Complete post-processing pipeline:

    1. Remove small isolated speckle (< 25 pixels).
    2. Enforce hydrological river network connectivity.
    3. Smooth shoreline contours.
    """
    cleaned = remove_small_components(raw_flood_mask, min_size_pixels=min_size_pixels)

    if enforce_connectivity and river_core_mask is not None:
        cleaned = filter_hydrological_connectivity(cleaned, river_core_mask=river_core_mask)

    smoothed = smooth_waterline_contours(cleaned)
    return smoothed
