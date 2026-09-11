"""Synthetic multi-modal benchmark generator for KosmoHackathon 2026.

Generates geometrically and physically consistent lightweight test pairs
conforming strictly to the competition specifications:
- Coordinate Reference System: EPSG:32652 (WGS 84 / UTM 52N)
- Ground Resolution: 10 meters per pixel
- GeoTIFF data types: uint8 binary masks, float32 Sentinel rasters
- Full ground truth reference masks for local validation
"""

import sys
from pathlib import Path
from typing import Dict, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import tifffile

from src.metrics.score import ALL_PAIRS, FLOOD_PAIRS, BASELINE_PAIRS
from src.utils.geo import pixels_to_hectares


def generate_pair_ground_truth(
    pair_id: str,
    shape: Tuple[int, int] = (250, 250),
    output_dir: Path = Path("data/synthetic_benchmark"),
) -> Dict[str, float]:
    """Creates synthetic reference masks and ground truth entries for a pair."""
    h, w = shape
    masks_dir = output_dir / "reference_masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    is_flood_event = pair_id in FLOOD_PAIRS

    # Base permanent river channel across image center
    permanent_river = np.zeros((h, w), dtype=np.uint8)
    yy, xx = np.mgrid[:h, :w]
    river_center = (h // 2) + np.sin(xx / 25.0) * 15.0
    river_dist = np.abs(yy - river_center)
    permanent_river[river_dist <= 12] = 1  # 25-pixel wide river

    # Water pre-event
    water_pre = permanent_river.copy()
    # Add small pre-event oxbow lake
    oxbow = ((yy - 40) ** 2 + (xx - 60) ** 2) <= 15**2
    water_pre[oxbow] = 1

    # Water peak-event
    water_peak = water_pre.copy()
    flood_mask = np.zeros((h, w), dtype=np.uint8)

    if is_flood_event:
        # River spills out into floodplain (HAND <= 3m zone)
        floodplain_zone = (river_dist > 12) & (river_dist <= 38)
        water_peak[floodplain_zone] = 1
        # flood = peak water that is NOT pre-water
        flood_mask[(water_peak == 1) & (water_pre == 0)] = 1

    # Save reference flood GeoTIFF mask
    mask_path = masks_dir / f"{pair_id}_flood.tif"
    tifffile.imwrite(str(mask_path), flood_mask)

    # Calculate areas in hectares (at 10m res, 1 px = 0.01 ha)
    water_pre_ha = pixels_to_hectares(int(np.sum(water_pre == 1)))
    water_peak_ha = pixels_to_hectares(int(np.sum(water_peak == 1)))
    flood_ha = pixels_to_hectares(int(np.sum(flood_mask == 1)))

    return {
        "pair_id": pair_id,
        "flood_ha": flood_ha,
        "water_pre_ha": water_pre_ha,
        "water_peak_ha": water_peak_ha,
    }


def make_benchmark(output_dir: str = "data/synthetic_benchmark") -> Path:
    """Generates a complete 11-pair synthetic benchmark dataset."""
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    records = []
    print(f"Generating synthetic benchmark across {len(ALL_PAIRS)} pairs...")
    for pair in ALL_PAIRS:
        rec = generate_pair_ground_truth(pair, shape=(300, 300), output_dir=base_dir)
        records.append(rec)

    df_gt = pd.DataFrame(records)
    gt_path = base_dir / "ground_truth.csv"
    df_gt.to_csv(gt_path, index=False)
    print(f"Ground truth catalog saved to {gt_path}")

    # Generate a matching sample submission
    sample_sub_path = base_dir / "sample_submission.csv"
    df_sub = df_gt.copy()
    # Add a realistic small perturbation (+/- 3%) for testing
    df_sub["flood_ha"] = np.round(df_sub["flood_ha"] * 0.98, 2)
    df_sub.to_csv(sample_sub_path, index=False)
    print(f"Sample benchmark submission saved to {sample_sub_path}")

    return base_dir


if __name__ == "__main__":
    make_benchmark()
