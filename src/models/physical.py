"""Track B: Physical-Hydrological Expert Algorithm (Physics Baseline).

Implements:
1. Adaptive Otsu thresholding on SAR VV backscatter in [-24, -11] dB range.
2. Wind-induced roughness compensation (shifts threshold or relies on VH/optics when wind > 3.5 m/s).
3. Flooded vegetation (double-bounce) detection via cross-pol ratio (VH - VV) and low HAND.
4. Continuous logistic HAND prior filtering to suppress high-elevation false alarms.
5. Multi-spectral optical consensus (MNDWI > 0.15) where optical observations are clear.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure

from src.features.sar import apply_lee_filter, compute_polarization_ratio, compute_temporal_delta
from src.features.terrain import compute_logistic_hand_prior
from src.utils.geo import compute_binary_mask_area_ha


def calculate_otsu_threshold(
    values: np.ndarray,
    val_min: float = -24.0,
    val_max: float = -11.0,
    num_bins: int = 128,
) -> float:
    """Calculates optimal bimodal separation threshold using Otsu's method.

    Restricted to plausible radar backscatter range [val_min, val_max] dB.
    """
    valid = values[(values >= val_min) & (values <= val_max)]
    if len(valid) < 100:
        return -16.5  # Standard empirical water/land threshold in Amur basin

    counts, bin_edges = np.histogram(valid, bins=num_bins, range=(val_min, val_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total_pixels = float(np.sum(counts))
    if total_pixels == 0:
        return -16.5

    current_max = 0.0
    optimal_threshold = -16.5

    # Otsu between-class variance maximization
    weight_background = 0.0
    sum_background = 0.0
    total_mean = float(np.sum(counts * bin_centers)) / total_pixels

    for i in range(num_bins):
        w = counts[i] / total_pixels
        if w == 0:
            continue
        weight_background += w
        weight_foreground = 1.0 - weight_background
        if weight_foreground <= 0:
            break

        sum_background += w * bin_centers[i]
        mean_background = sum_background / weight_background
        mean_foreground = (total_mean - sum_background) / weight_foreground

        # Inter-class variance: w_bg * w_fg * (mean_bg - mean_fg)^2
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if variance > current_max:
            current_max = variance
            optimal_threshold = float(bin_edges[i + 1])

    return float(np.clip(optimal_threshold, val_min, val_max))


class PhysicalHydrologyModel:
    """Physical expert model integrating radar physics, wind compensation, and terrain priors."""

    def __init__(
        self,
        hand_h0: float = 12.0,
        hand_tau: float = 3.0,
        wind_speed_threshold_ms: float = 3.5,
        wind_backscatter_offset_db: float = 2.5,
        delta_sigma0_flood_threshold_db: float = -2.8,
        double_bounce_ratio_threshold_db: float = -6.5,
        mndwi_water_threshold: float = 0.15,
    ):
        self.hand_h0 = hand_h0
        self.hand_tau = hand_tau
        self.wind_speed_threshold = wind_speed_threshold_ms
        self.wind_offset = wind_backscatter_offset_db
        self.delta_sigma0_thresh = delta_sigma0_flood_threshold_db
        self.double_bounce_ratio_thresh = double_bounce_ratio_threshold_db
        self.mndwi_thresh = mndwi_water_threshold

    def segment_water_surface(
        self,
        s1_vv_db: np.ndarray,
        s1_vh_db: np.ndarray,
        hand_meters: np.ndarray,
        wind_speed_ms: float = 2.0,
        mndwi: Optional[np.ndarray] = None,
        cloud_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Segments open water surface and returns (binary_mask, probability_map, otsu_thresh)."""
        h, w = s1_vv_db.shape

        # 1. Compute dynamic Otsu threshold on VV
        base_threshold = calculate_otsu_threshold(s1_vv_db)

        # 2. Wind compensation: if wind > 3.5 m/s, surface capillary waves raise sigma0
        effective_threshold = base_threshold
        if wind_speed_ms > self.wind_speed_threshold:
            # Shift threshold upwards to prevent calm water from being dropped
            effective_threshold += self.wind_offset

        # 3. Radar water probability via sigmoid around threshold
        # Lower VV -> higher water probability
        vv_prob = 1.0 / (1.0 + np.exp((s1_vv_db - effective_threshold) / 1.5))

        # 4. Continuous HAND Prior
        hand_prior = compute_logistic_hand_prior(hand_meters, h0_center_meters=self.hand_h0, temperature_tau=self.hand_tau)

        # 5. Combine with Optical MNDWI if clear sky
        if mndwi is not None and cloud_mask is not None:
            clear_optics = (cloud_mask == 0)
            opt_water_prob = 1.0 / (1.0 + np.exp(-(mndwi - self.mndwi_thresh) / 0.1))
            # Where clear, blend SAR and Optics; where cloudy, use 100% SAR
            fused_water_prob = np.where(clear_optics, 0.55 * vv_prob + 0.45 * opt_water_prob, vv_prob)
        else:
            fused_water_prob = vv_prob

        # Multiply by terrain prior
        final_prob = fused_water_prob * hand_prior

        # Binary decision: high water probability and valid terrain prior
        binary_water = ((fused_water_prob >= 0.50) & (hand_prior >= 0.25)).astype(np.uint8)

        return binary_water, final_prob.astype(np.float32), effective_threshold

    def detect_flood(
        self,
        s1_pre_vv_db: np.ndarray,
        s1_pre_vh_db: np.ndarray,
        s1_peak_vv_db: np.ndarray,
        s1_peak_vh_db: np.ndarray,
        hand_meters: np.ndarray,
        perm_water_mask: np.ndarray,
        wind_speed_ms: float = 2.0,
        mndwi_peak: Optional[np.ndarray] = None,
        cloud_mask_peak: Optional[np.ndarray] = None,
    ) -> Dict[str, Union[np.ndarray, float]]:
        """Executes full multi-temporal physical flood detection for a pair."""
        # 1. Segment water at pre date
        water_pre, prob_pre, _ = self.segment_water_surface(
            s1_vv_db=s1_pre_vv_db,
            s1_vh_db=s1_pre_vh_db,
            hand_meters=hand_meters,
            wind_speed_ms=wind_speed_ms,
        )

        # 2. Segment water at peak date
        water_peak, prob_peak, thresh_peak = self.segment_water_surface(
            s1_vv_db=s1_peak_vv_db,
            s1_vh_db=s1_peak_vh_db,
            hand_meters=hand_meters,
            wind_speed_ms=wind_speed_ms,
            mndwi=mndwi_peak,
            cloud_mask=cloud_mask_peak,
        )

        # 3. Temporal delta backscatter
        delta_vv = compute_temporal_delta(s1_peak_vv_db, s1_pre_vv_db)

        # 4. Double-bounce detection in flooded vegetation:
        # In flooded forest: VV remains moderately high, VH/VV ratio drops, HAND is low (< 2.5m)
        ratio_peak = compute_polarization_ratio(s1_peak_vh_db, s1_peak_vv_db)
        is_flooded_veg = (
            (hand_meters <= 2.5)
            & (ratio_peak <= self.double_bounce_ratio_thresh)
            & (s1_peak_vv_db >= -13.0)
            & (s1_peak_vv_db <= -6.0)
        )

        # 5. Raw flood = (Water at peak) AND (NOT Water at pre) AND (NOT permanent water)
        raw_flood = (water_peak == 1) & (water_pre == 0) & (perm_water_mask == 0)

        # Verified flood requires either strong backscatter drop (delta_vv <= -2.8 dB)
        # or agreement with optical MNDWI or double-bounce indicator
        confirmed_by_delta = delta_vv <= self.delta_sigma0_thresh
        if mndwi_peak is not None:
            confirmed_by_optics = (mndwi_peak > self.mndwi_thresh)
        else:
            confirmed_by_optics = False

        flood_mask = raw_flood & (confirmed_by_delta | confirmed_by_optics | is_flooded_veg)

        # Morphological noise cleanup
        selem = generate_binary_structure(2, 1)
        cleaned_flood = binary_opening(flood_mask, structure=selem)
        cleaned_flood = binary_closing(cleaned_flood, structure=selem).astype(np.uint8)

        # Area calculations in hectares
        water_pre_ha = compute_binary_mask_area_ha(water_pre)
        water_peak_ha = compute_binary_mask_area_ha(water_peak)
        flood_ha = compute_binary_mask_area_ha(cleaned_flood)

        # Enforce physical consistency: flood_ha <= water_peak_ha
        flood_ha = min(flood_ha, water_peak_ha)

        return {
            "water_pre_mask": water_pre,
            "water_peak_mask": water_peak,
            "flood_mask": cleaned_flood,
            "water_pre_ha": water_pre_ha,
            "water_peak_ha": water_peak_ha,
            "flood_ha": flood_ha,
            "effective_otsu_threshold_db": round(thresh_peak, 2),
            "prob_peak": prob_peak,
        }
