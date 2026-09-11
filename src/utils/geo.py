"""Geospatial utilities for KosmoHackathon 2026."""

from typing import Tuple
import numpy as np

PIXEL_RESOLUTION_M = 10.0
HA_PER_PIXEL = (PIXEL_RESOLUTION_M * PIXEL_RESOLUTION_M) / 10000.0  # 0.01 ha per 10x10m pixel


def pixels_to_hectares(num_pixels: int) -> float:
    """Converts a count of 10m x 10m pixels into hectares."""
    return round(float(num_pixels * HA_PER_PIXEL), 2)


def hectares_to_pixels(hectares: float) -> int:
    """Converts area in hectares to equivalent number of 10m x 10m pixels."""
    return int(round(hectares / HA_PER_PIXEL))


def compute_binary_mask_area_ha(mask: np.ndarray) -> float:
    """Calculates area in hectares of non-zero pixels in a binary mask."""
    count = int(np.count_nonzero(mask))
    return pixels_to_hectares(count)
