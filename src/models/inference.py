"""Unified Dual-Track Inference Engine.

Integrates Track A (Deep Learning) and Track B (Physics Baseline) with sliding-window
Hann tile stitching for memory-bounded high-resolution inference.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch

from src.features.pipeline import assemble_multimodal_tensor
from src.features.tiling import TileStitcher, extract_tiles
from src.models.physical import PhysicalHydrologyModel
from src.models.deep_learning import MultiModalHydrologyNet
from src.utils.geo import compute_binary_mask_area_ha


class HydrologyInferenceEngine:
    """Full-scene dual-track inference engine."""

    def __init__(
        self,
        deep_model: Optional[MultiModalHydrologyNet] = None,
        physical_model: Optional[PhysicalHydrologyModel] = None,
        tile_size: int = 512,
        overlap: int = 64,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        deep_weight: float = 0.65,
    ):
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.deep_weight = deep_weight

        self.physical_model = physical_model or PhysicalHydrologyModel()
        self.deep_model = deep_model
        if self.deep_model is not None:
            self.deep_model.to(self.device)
            self.deep_model.eval()

    def predict_scene(
        self,
        # SAR inputs
        s1_pre_vv: np.ndarray,
        s1_pre_vh: np.ndarray,
        s1_peak_vv: np.ndarray,
        s1_peak_vh: np.ndarray,
        # Auxiliary inputs
        hand_meters: np.ndarray,
        perm_water_mask: np.ndarray,
        dem_or_slope: Optional[np.ndarray] = None,
        builtup_layer: Optional[np.ndarray] = None,
        # Optical inputs
        s2_peak_b03: Optional[np.ndarray] = None,
        s2_peak_b04: Optional[np.ndarray] = None,
        s2_peak_b08: Optional[np.ndarray] = None,
        s2_peak_b11: Optional[np.ndarray] = None,
        s2_scl: Optional[np.ndarray] = None,
        wind_speed_ms: float = 2.0,
    ) -> Dict[str, Union[np.ndarray, float, str]]:
        """Executes full-scene prediction returning binary masks and areas in hectares."""
        h, w = s1_peak_vv.shape

        # 1. Physical Expert Track (Track B)
        phys_result = self.physical_model.detect_flood(
            s1_pre_vv_db=s1_pre_vv,
            s1_pre_vh_db=s1_pre_vh,
            s1_peak_vv_db=s1_peak_vv,
            s1_peak_vh_db=s1_peak_vh,
            hand_meters=hand_meters,
            perm_water_mask=perm_water_mask,
            wind_speed_ms=wind_speed_ms,
        )

        p_phys_flood = phys_result["flood_mask"].astype(np.float32)

        # 2. Deep Learning Track (Track A) if model provided
        if self.deep_model is not None:
            # Assemble multi-modal 15-channel tensor
            tensor_15ch, meta = assemble_multimodal_tensor(
                s1_pre_vv=s1_pre_vv,
                s1_pre_vh=s1_pre_vh,
                s1_peak_vv=s1_peak_vv,
                s1_peak_vh=s1_peak_vh,
                s2_peak_b03=s2_peak_b03,
                s2_peak_b04=s2_peak_b04,
                s2_peak_b08=s2_peak_b08,
                s2_peak_b11=s2_peak_b11,
                s2_scl=s2_scl,
                hand_meters=hand_meters,
                dem_or_slope=dem_or_slope,
                gsw_occurrence_pct=perm_water_mask * 100.0,
                builtup_layer=builtup_layer,
                filter_sar_speckle=True,
            )

            # Tiled inference using Hann window stitcher
            stitcher_water = TileStitcher(h, w, tile_size=self.tile_size, overlap=self.overlap)
            stitcher_flood = TileStitcher(h, w, tile_size=self.tile_size, overlap=self.overlap)

            with torch.no_grad():
                for tile_feat, (y1, y2, x1, x2) in extract_tiles(tensor_15ch, tile_size=self.tile_size, overlap=self.overlap):
                    th, tw = y2 - y1, x2 - x1
                    # Pad tile to exact tile_size if along boundary
                    if th < self.tile_size or tw < self.tile_size:
                        padded = np.zeros((15, self.tile_size, self.tile_size), dtype=np.float32)
                        padded[:, :th, :tw] = tile_feat
                        inp = torch.from_numpy(padded).unsqueeze(0).to(self.device)
                    else:
                        inp = torch.from_numpy(tile_feat).unsqueeze(0).to(self.device)

                    out = self.deep_model(inp)
                    p_w = out["prob_water"].squeeze().cpu().numpy()[:th, :tw]
                    p_f = out["prob_flood"].squeeze().cpu().numpy()[:th, :tw]

                    stitcher_water.add_tile(p_w, y1, y2, x1, x2)
                    stitcher_flood.add_tile(p_f, y1, y2, x1, x2)

            p_deep_water = stitcher_water.finalize()
            p_deep_flood = stitcher_flood.finalize()

            # Dynamic blending: if optical is cloudy, rely more on physics/SAR
            w_deep = self.deep_weight if meta["is_optical_valid"] else min(self.deep_weight, 0.45)
            fused_prob_flood = w_deep * p_deep_flood + (1.0 - w_deep) * p_phys_flood
            final_flood_mask = (fused_prob_flood >= 0.50).astype(np.uint8)
            final_water_peak = (p_deep_water >= 0.50).astype(np.uint8)
        else:
            # Fall back to pure Physical Track B
            final_flood_mask = phys_result["flood_mask"]
            final_water_peak = phys_result["water_peak_mask"]
            fused_prob_flood = p_phys_flood

        # Clean mask: permanent water must strictly not be in flood
        final_flood_mask[perm_water_mask == 1] = 0
        final_water_pre = phys_result["water_pre_mask"]

        # Calculate areas in hectares
        flood_ha = compute_binary_mask_area_ha(final_flood_mask)
        water_pre_ha = compute_binary_mask_area_ha(final_water_pre)
        water_peak_ha = compute_binary_mask_area_ha(final_water_peak)

        # Enforce physical rule: flood_ha <= water_peak_ha
        flood_ha = min(flood_ha, water_peak_ha)

        return {
            "flood_mask": final_flood_mask,
            "water_pre_mask": final_water_pre,
            "water_peak_mask": final_water_peak,
            "flood_probability": fused_prob_flood,
            "flood_ha": flood_ha,
            "water_pre_ha": water_pre_ha,
            "water_peak_ha": water_peak_ha,
        }
