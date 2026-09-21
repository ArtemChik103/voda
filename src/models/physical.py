"""Track B: Physical-Hydrological Expert Algorithm (Physics Baseline).

Implements:
1. Adaptive Otsu thresholding on SAR VV backscatter in [-24, -11] dB range.
2. Wind-induced roughness compensation (shifts threshold or relies on VH/optics when wind > 3.5 m/s).
3. Flooded vegetation (double-bounce) detection via cross-pol ratio (VH - VV) and low HAND.
4. Continuous logistic HAND prior filtering to suppress high-elevation false alarms.
5. Multi-spectral optical consensus (MNDWI > 0.15) where optical observations are clear.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure, label

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


def filter_urban_false_alarms(
    flood_mask: np.ndarray,
    builtup_mask: np.ndarray,
    hand_meters: np.ndarray,
    airport_hand_threshold_m: float = 12.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Suppresses artificial specular false alarms (airport runways, roads, large roofs).

    Trap context (Blagoveshchensk / urban):
    Smooth asphalt runways (e.g. Ignatyevo airport) and metal roofs cause specular radar reflection
    with low backscatter (-20 to -26 dB) identical to calm water. However, genuine river floods
    are constrained to low HAND elevations. Pixels with builtup > 0.5 and HAND > 12m are false alarms.
    """
    is_urban_trap = (builtup_mask > 0.5) & (hand_meters > float(airport_hand_threshold_m))
    filtered = flood_mask.copy()
    filtered[is_urban_trap] = 0
    return filtered, is_urban_trap.astype(np.uint8)


def filter_permanent_water_gsw(
    flood_mask: np.ndarray,
    gsw_occurrence_pct: np.ndarray,
    occurrence_threshold_pct: float = 80.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Excludes permanent oxbow lakes and perennial river channels from new flood mask.

    Trap context (Svobodny / Zeya river meanders):
    The Zeya floodplain contains numerous oxbow lakes (старицы) and perennial wetlands.
    These features have historical water occurrence >= 80% and must be masked as permanent
    water to avoid inflating flood_ha.
    """
    is_perm_trap = gsw_occurrence_pct >= float(occurrence_threshold_pct)
    filtered = flood_mask.copy()
    filtered[is_perm_trap] = 0
    return filtered, is_perm_trap.astype(np.uint8)


def filter_waterlogged_cropland(
    flood_mask: np.ndarray,
    mndwi_peak: np.ndarray,
    ndvi_peak: np.ndarray,
    delta_vv_db: np.ndarray,
    mndwi_water_threshold: float = 0.15,
    ndvi_cropland_threshold: float = 0.20,
    delta_vv_min_drop_db: float = -3.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Separates standing flood water from saturated / waterlogged agricultural soils.

    Trap context (Belogorsk / Tom river):
    Moist agricultural fields produce weak moisture signals (MNDWI 0.05..0.15) and moderate backscatter
    drop, but maintain vegetative structure (NDVI > 0.20). Open standing water completely extinguishes
    NIR (NDVI <= 0.10) and exhibits MNDWI >= 0.15 with deep backscatter drop.
    """
    is_waterlogged = (
        (mndwi_peak >= 0.05) & (mndwi_peak < float(mndwi_water_threshold))
        & (ndvi_peak > float(ndvi_cropland_threshold))
        & (delta_vv_db > float(delta_vv_min_drop_db))
    )
    filtered = flood_mask.copy()
    filtered[is_waterlogged] = 0
    return filtered, is_waterlogged.astype(np.uint8)


def filter_dry_sandbars(
    flood_mask: np.ndarray,
    gsw_max_extent: np.ndarray,
    hand_meters: np.ndarray,
    b04_red_peak: Optional[np.ndarray] = None,
    mndwi_peak: Optional[np.ndarray] = None,
    sandbar_hand_threshold_m: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Suppresses dry sand and gravel bars that mimic specular radar return.

    Trap context (Poyarkovo / Amur river sandbars):
    Dry quartz sand on elevated river bars has low backscatter. However, elevated sand has never been
    inundated in the 38-year GSW record (max_extent == 0 and HAND > 8m), and optically exhibits high
    red reflectance (B04 > 0.18) with negative MNDWI.
    """
    is_sand_trap = (gsw_max_extent == 0) & (hand_meters > float(sandbar_hand_threshold_m))
    if b04_red_peak is not None and mndwi_peak is not None:
        is_opt_sand = (b04_red_peak > 0.18) & (mndwi_peak < 0.0)
        is_sand_trap = is_sand_trap | is_opt_sand

    filtered = flood_mask.copy()
    filtered[is_sand_trap] = 0
    return filtered, is_sand_trap.astype(np.uint8)


def filter_speckle_noise_patches(
    flood_mask: np.ndarray,
    min_patch_pixels: int = 5,
) -> np.ndarray:
    """Removes isolated speckle artifacts with size below min_patch_pixels."""
    labeled, num_features = label(flood_mask)
    if num_features == 0:
        return flood_mask
    counts = np.bincount(labeled.ravel())
    small_indices = np.where(counts < min_patch_pixels)[0]
    cleaned = flood_mask.copy()
    cleaned[np.isin(labeled, small_indices)] = 0
    return cleaned.astype(np.uint8)


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
        airport_hand_threshold_m: float = 12.0,
        permanent_water_occurrence_pct: float = 80.0,
        cropland_ndvi_threshold: float = 0.20,
        cropland_delta_vv_drop_db: float = -3.5,
        sandbar_hand_threshold_m: float = 8.0,
        min_flood_patch_pixels: int = 5,
    ):
        self.hand_h0 = hand_h0
        self.hand_tau = hand_tau
        self.wind_speed_threshold = wind_speed_threshold_ms
        self.wind_offset = wind_backscatter_offset_db
        self.delta_sigma0_thresh = delta_sigma0_flood_threshold_db
        self.double_bounce_ratio_thresh = double_bounce_ratio_threshold_db
        self.mndwi_thresh = mndwi_water_threshold
        # Trap parameters
        self.airport_hand_thresh = airport_hand_threshold_m
        self.perm_gsw_thresh = permanent_water_occurrence_pct
        self.cropland_ndvi_thresh = cropland_ndvi_threshold
        self.cropland_delta_thresh = cropland_delta_vv_drop_db
        self.sandbar_hand_thresh = sandbar_hand_threshold_m
        self.min_patch_pixels = min_flood_patch_pixels

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
        builtup_mask: Optional[np.ndarray] = None,
        gsw_occurrence_pct: Optional[np.ndarray] = None,
        gsw_max_extent: Optional[np.ndarray] = None,
        ndvi_peak: Optional[np.ndarray] = None,
        b04_red_peak: Optional[np.ndarray] = None,
        weather_features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Union[np.ndarray, float, Dict[str, int]]]:
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

        # Pre-filter urban infrastructure from water masks
        if builtup_mask is not None:
            is_urban_false = (builtup_mask > 0.5) & (hand_meters > float(self.airport_hand_thresh))
            water_pre = np.where(is_urban_false, 0, water_pre).astype(np.uint8)
            water_peak = np.where(is_urban_false, 0, water_peak).astype(np.uint8)

        # Pre-filter GSW permanent water >= 80%
        effective_perm_water = perm_water_mask.copy()
        if gsw_occurrence_pct is not None:
            is_perm_gsw = gsw_occurrence_pct >= float(self.perm_gsw_thresh)
            effective_perm_water = (effective_perm_water | is_perm_gsw.astype(np.uint8)).astype(np.uint8)

        # 3. Temporal delta backscatter
        delta_vv = compute_temporal_delta(s1_peak_vv_db, s1_pre_vv_db)

        # 4. Double-bounce detection in flooded vegetation:
        ratio_peak = compute_polarization_ratio(s1_peak_vh_db, s1_peak_vv_db)
        is_flooded_veg = (
            (hand_meters <= 2.5)
            & (ratio_peak <= self.double_bounce_ratio_thresh)
            & (s1_peak_vv_db >= -13.0)
            & (s1_peak_vv_db <= -6.0)
        )

        # 5. Raw flood = (Water at peak) AND (NOT Water at pre) AND (NOT permanent water)
        raw_flood = (water_peak == 1) & (water_pre == 0) & (effective_perm_water == 0)

        # Verified flood requires either strong backscatter drop (delta_vv <= threshold)
        # or agreement with optical MNDWI or double-bounce indicator
        confirmed_by_delta = delta_vv <= self.delta_sigma0_thresh
        if mndwi_peak is not None:
            confirmed_by_optics = (mndwi_peak > self.mndwi_thresh)
        else:
            confirmed_by_optics = False

        flood_mask = raw_flood & (confirmed_by_delta | confirmed_by_optics | is_flooded_veg)

        # Trap Filter Tracking
        trap_stats: Dict[str, int] = {}

        # Trap 1: Airport & Built-up suppression (Blagoveshchensk)
        if builtup_mask is not None:
            flood_mask, urban_trap = filter_urban_false_alarms(
                flood_mask=flood_mask,
                builtup_mask=builtup_mask,
                hand_meters=hand_meters,
                airport_hand_threshold_m=self.airport_hand_thresh,
            )
            trap_stats["urban_pixels_filtered"] = int(np.sum(urban_trap))

        # Trap 2: Permanent water & Oxbow lakes suppression (Svobodny)
        if gsw_occurrence_pct is not None:
            flood_mask, oxbow_trap = filter_permanent_water_gsw(
                flood_mask=flood_mask,
                gsw_occurrence_pct=gsw_occurrence_pct,
                occurrence_threshold_pct=self.perm_gsw_thresh,
            )
            trap_stats["oxbow_pixels_filtered"] = int(np.sum(oxbow_trap))

        # Trap 3: Waterlogged cropland suppression (Belogorsk)
        if mndwi_peak is not None and ndvi_peak is not None:
            flood_mask, cropland_trap = filter_waterlogged_cropland(
                flood_mask=flood_mask,
                mndwi_peak=mndwi_peak,
                ndvi_peak=ndvi_peak,
                delta_vv_db=delta_vv,
                mndwi_water_threshold=self.mndwi_thresh,
                ndvi_cropland_threshold=self.cropland_ndvi_thresh,
                delta_vv_min_drop_db=self.cropland_delta_thresh,
            )
            trap_stats["cropland_pixels_filtered"] = int(np.sum(cropland_trap))

        # Trap 4: Dry sandbars suppression (Poyarkovo)
        if gsw_max_extent is not None:
            flood_mask, sand_trap = filter_dry_sandbars(
                flood_mask=flood_mask,
                gsw_max_extent=gsw_max_extent,
                hand_meters=hand_meters,
                b04_red_peak=b04_red_peak,
                mndwi_peak=mndwi_peak,
                sandbar_hand_threshold_m=self.sandbar_hand_thresh,
            )
            trap_stats["sandbar_pixels_filtered"] = int(np.sum(sand_trap))

        # Weather Modulation: adjust minimum patch size during dry baseline periods
        min_patch = self.min_patch_pixels
        if weather_features is not None:
            precip_7d = float(weather_features.get("precip_7d_sum_mm", 50.0))
            risk_level = str(weather_features.get("weather_risk_level", "HIGH"))
            if risk_level == "LOW" or precip_7d < 15.0:
                # Dry period: eliminate isolated noise to protect Spec_base = 1.000
                min_patch = max(12, self.min_patch_pixels * 2)

        # Morphological noise cleanup
        selem = generate_binary_structure(2, 1)
        cleaned_flood = binary_opening(flood_mask, structure=selem)
        cleaned_flood = binary_closing(cleaned_flood, structure=selem).astype(np.uint8)

        # Enforce hard physical domain boundaries post-morphology (guarantees no closing leakage)
        if builtup_mask is not None:
            is_urban_false = (builtup_mask > 0.5) & (hand_meters > float(self.airport_hand_thresh))
            cleaned_flood[is_urban_false] = 0
        if gsw_occurrence_pct is not None:
            is_perm_gsw = gsw_occurrence_pct >= float(self.perm_gsw_thresh)
            cleaned_flood[is_perm_gsw] = 0
        if gsw_max_extent is not None:
            is_sand_trap = (gsw_max_extent == 0) & (hand_meters > float(self.sandbar_hand_thresh))
            cleaned_flood[is_sand_trap] = 0
        cleaned_flood[effective_perm_water == 1] = 0

        # Speckle noise patch filtering
        cleaned_flood = filter_speckle_noise_patches(cleaned_flood, min_patch_pixels=min_patch)

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
            "trap_stats": trap_stats,
        }
