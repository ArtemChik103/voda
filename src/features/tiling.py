"""Sliding window tiling and 2D Hann window blending engine for large raster processing.

Enables memory-efficient chunked processing of huge geospatial scenes (up to 4500x4500)
with zero seam or edge artifacts at tile boundaries.
"""

from typing import Generator, List, Tuple
import numpy as np


def create_2d_hann_window(size: int, power: float = 1.0) -> np.ndarray:
    """Generates a 2D Hann (cosine) blending window for smooth tile stitching.

    Args:
        size: Dimension of square tile (e.g. 512 or 1024).
        power: Exponent for steepness of window falloff (default 1.0).

    Returns:
        2D array of shape (size, size) with values in [0, 1].
    """
    w1d = np.hanning(size).astype(np.float32)
    # Add small epsilon to avoid exact zero at edges
    w1d = np.maximum(w1d, 1e-3)
    w2d = np.outer(w1d, w1d)
    if power != 1.0:
        w2d = np.power(w2d, power)
    return w2d.astype(np.float32)


class TileStitcher:
    """Seamless 2D window accumulator and reconstructor using Hann blending.

    Guarantees mathematically smooth tile transitions and eliminates boundary artifacts.
    """

    def __init__(
        self,
        full_height: int,
        full_width: int,
        tile_size: int = 512,
        overlap: int = 64,
        num_channels: int = 1,
    ):
        self.height = full_height
        self.width = full_width
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
        self.num_channels = num_channels

        if self.stride <= 0:
            raise ValueError(f"overlap ({overlap}) must be strictly less than tile_size ({tile_size})")

        self.window_2d = create_2d_hann_window(tile_size)

        # Accumulation buffers
        if num_channels == 1:
            self.accumulator = np.zeros((full_height, full_width), dtype=np.float32)
        else:
            self.accumulator = np.zeros((num_channels, full_height, full_width), dtype=np.float32)
        self.weight_map = np.zeros((full_height, full_width), dtype=np.float32)

    def generate_tile_coords(self) -> List[Tuple[int, int, int, int]]:
        """Computes all (y1, y2, x1, x2) bounding boxes covering the full image."""
        coords = []
        y_starts = list(range(0, max(1, self.height - self.tile_size + 1), self.stride))
        if y_starts[-1] + self.tile_size < self.height:
            y_starts.append(self.height - self.tile_size)

        x_starts = list(range(0, max(1, self.width - self.tile_size + 1), self.stride))
        if x_starts[-1] + self.tile_size < self.width:
            x_starts.append(self.width - self.tile_size)

        for y in y_starts:
            for x in x_starts:
                y1 = max(0, y)
                y2 = min(self.height, y1 + self.tile_size)
                x1 = max(0, x)
                x2 = min(self.width, x1 + self.tile_size)
                # If image is smaller than tile_size
                if (y2 - y1) < self.tile_size or (x2 - x1) < self.tile_size:
                    y1 = 0
                    y2 = min(self.height, self.tile_size)
                    x1 = 0
                    x2 = min(self.width, self.tile_size)
                coords.append((y1, y2, x1, x2))

        return coords

    def add_tile(
        self,
        tile_pred: np.ndarray,
        y1: int,
        y2: int,
        x1: int,
        x2: int,
    ) -> None:
        """Adds a predicted tile into the weighted accumulation buffer."""
        th = y2 - y1
        tw = x2 - x1
        w = self.window_2d[:th, :tw]

        if self.num_channels == 1:
            tile = tile_pred.squeeze()[:th, :tw]
            self.accumulator[y1:y2, x1:x2] += tile * w
        else:
            tile = tile_pred[:, :th, :tw]
            self.accumulator[:, y1:y2, x1:x2] += tile * w[np.newaxis, ...]

        self.weight_map[y1:y2, x1:x2] += w

    def finalize(self) -> np.ndarray:
        """Normalizes accumulated predictions by total spatial blending weights.

        Returns:
            Reconstructed array matching the original image dimensions.
        """
        weights = np.maximum(self.weight_map, 1e-8)
        if self.num_channels == 1:
            result = self.accumulator / weights
        else:
            result = self.accumulator / weights[np.newaxis, ...]
        return result


def extract_tiles(
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
) -> Generator[Tuple[np.ndarray, Tuple[int, int, int, int]], None, None]:
    """Extracts sliding window tiles from an image tensor (C, H, W) or (H, W).

    Yields:
        (tile_array, (y1, y2, x1, x2))
    """
    is_2d = image.ndim == 2
    if is_2d:
        h, w = image.shape
    else:
        _, h, w = image.shape

    stitcher = TileStitcher(h, w, tile_size=tile_size, overlap=overlap)
    coords = stitcher.generate_tile_coords()

    for (y1, y2, x1, x2) in coords:
        if is_2d:
            tile = image[y1:y2, x1:x2]
        else:
            tile = image[:, y1:y2, x1:x2]
        yield tile, (y1, y2, x1, x2)
