"""Unit tests for the geospatial feature engineering pipeline (src/features/)."""

import pytest
import numpy as np

from src.features.sar import (
    apply_lee_filter,
    compute_temporal_delta,
    compute_polarization_ratio,
    compute_radar_shadow_mask,
    extract_sar_feature_stack,
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


def test_lee_filter_variance_reduction():
    """Lee filter must reduce variance in homogeneous speckle-corrupted regions."""
    np.random.seed(42)
    # Synthetic flat region (-18 dB) with multiplicative Rayleigh speckle
    base_linear = 10.0 ** (-18.0 / 10.0)
    speckle_noise = np.random.exponential(scale=1.0, size=(100, 100))
    noisy_db = 10.0 * np.log10(np.maximum(base_linear * speckle_noise, 1e-5))

    initial_var = np.var(noisy_db)
    filtered_db = apply_lee_filter(noisy_db, window_size=7)
    filtered_var = np.var(filtered_db)

    # Variance must be significantly reduced
    assert filtered_var < initial_var * 0.45
    assert filtered_db.shape == noisy_db.shape


def test_lee_filter_edge_preservation():
    """Lee filter must preserve sharp high-contrast boundary transitions."""
    # Step edge from -22 dB (water) to -10 dB (land)
    step = np.full((50, 50), -22.0, dtype=np.float32)
    step[:, 25:] = -10.0

    filtered = apply_lee_filter(step, window_size=5)
    # Points well away from edge must remain near their true values
    assert pytest.approx(filtered[25, 5], abs=0.5) == -22.0
    assert pytest.approx(filtered[25, 45], abs=0.5) == -10.0


def test_temporal_delta_and_ratio():
    """Delta sigma0 and VH/VV ratio must strictly match math definitions."""
    peak = np.array([[-20.0, -15.0]], dtype=np.float32)
    pre = np.array([[-12.0, -15.0]], dtype=np.float32)
    delta = compute_temporal_delta(peak, pre)
    np.testing.assert_allclose(delta, [[-8.0, 0.0]])

    vh = np.array([[-26.0]], dtype=np.float32)
    vv = np.array([[-18.0]], dtype=np.float32)
    ratio = compute_polarization_ratio(vh, vv)
    assert ratio[0, 0] == -8.0


def test_radar_shadow_mask():
    """Shadows are flagged when backscatter is near noise floor on steep relief."""
    vv = np.array([[-26.0, -26.0], [-15.0, -10.0]], dtype=np.float32)
    slope = np.array([[25.0, 5.0], [30.0, 2.0]], dtype=np.float32)
    shadow = compute_radar_shadow_mask(vv, slope_deg=slope, noise_floor_db=-24.0, slope_threshold_deg=15.0)

    # [0, 0]: vv <= -24 and slope >= 15 -> 1 (shadow)
    # [0, 1]: vv <= -24 but slope < 15 -> 0 (calm flat water)
    # [1, 0]: vv > -24 and slope >= 15 -> 0 (bright steep land)
    assert shadow[0, 0] == 1
    assert shadow[0, 1] == 0
    assert shadow[1, 0] == 0


def test_optical_indices():
    """Water indices must produce high positive values for water and negative for dry land."""
    # Water spectral signature: high Green (0.25), low Red (0.05), very low NIR (0.02), zero SWIR (0.01)
    green = np.array([[0.25]], dtype=np.float32)
    red = np.array([[0.05]], dtype=np.float32)
    nir = np.array([[0.02]], dtype=np.float32)
    swir = np.array([[0.01]], dtype=np.float32)

    mndwi = compute_mndwi(green, swir)
    ndwi = compute_ndwi(green, nir)
    ndvi = compute_ndvi(nir, red)

    assert mndwi[0, 0] > 0.8  # Strong water response
    assert ndwi[0, 0] > 0.7   # Strong water response
    assert ndvi[0, 0] < 0.0   # Negative NDVI for water


def test_cloud_screening_and_quality():
    """SCL mask flags clouds and detects unviable optical scenes."""
    scl = np.array([[4, 6], [9, 3]], dtype=np.uint8)  # 4=Veg, 6=Water, 9=Cloud, 3=Shadow
    cloud_mask = create_cloud_mask_from_scl(scl)

    assert cloud_mask[0, 0] == 0  # Veg
    assert cloud_mask[0, 1] == 0  # Water
    assert cloud_mask[1, 0] == 1  # Cloud
    assert cloud_mask[1, 1] == 1  # Shadow

    # 2 out of 4 pixels are cloudy -> 50% cloud fraction
    is_valid, cloud_frac = evaluate_optical_quality(cloud_mask, cloud_threshold_ratio=0.7)
    assert is_valid is True
    assert cloud_frac == 0.5

    # 100% cloudy scene
    full_cloud = np.ones((10, 10), dtype=np.uint8)
    is_valid, cloud_frac = evaluate_optical_quality(full_cloud, cloud_threshold_ratio=0.7)
    assert is_valid is False
    assert cloud_frac == 1.0


def test_logistic_hand_prior():
    """Logistic HAND prior must decrease monotonically with elevation above drainage."""
    hand = np.array([[0.0, 6.0, 12.0, 18.0, 30.0]], dtype=np.float32)
    prior = compute_logistic_hand_prior(hand, h0_center_meters=12.0, temperature_tau=3.0)

    # In floodplain (HAND=0): very high prior
    assert prior[0, 0] > 0.95
    # At transition threshold (HAND=12): exactly 0.50
    assert pytest.approx(prior[0, 2], abs=1e-3) == 0.50
    # High hillside (HAND=30): virtually impossible to flood
    assert prior[0, 4] < 0.01
    # Strictly monotonically decreasing
    assert np.all(np.diff(prior[0]) < 0)


def test_slope_computation():
    """Flat terrain must yield 0 slope; steep slope must yield expected gradient."""
    flat_dem = np.full((50, 50), 100.0, dtype=np.float32)
    slope_flat = compute_slope_degrees(flat_dem, pixel_size_meters=10.0)
    assert np.all(slope_flat == 0.0)


def test_tile_stitching_hann_identity():
    """Stitching a constant 1.0 field across overlapping tiles with Hann window must reconstruct 1.0."""
    h, w = 300, 300
    tile_size = 128
    overlap = 32

    stitcher = TileStitcher(full_height=h, full_width=w, tile_size=tile_size, overlap=overlap)
    coords = stitcher.generate_tile_coords()
    assert len(coords) > 1

    for (y1, y2, x1, x2) in coords:
        th = y2 - y1
        tw = x2 - x1
        tile_pred = np.ones((th, tw), dtype=np.float32)
        stitcher.add_tile(tile_pred, y1, y2, x1, x2)

    reconstructed = stitcher.finalize()
    assert reconstructed.shape == (h, w)
    # The reconstructed values across the entire field should be 1.0 within numerical tolerance
    np.testing.assert_allclose(reconstructed, 1.0, rtol=1e-4, atol=1e-4)


def test_multimodal_tensor_assembly():
    """Pipeline must assemble a 15-channel feature tensor and metadata."""
    h, w = 80, 80
    # Synthetic SAR
    s1_pre_vv = np.full((h, w), -12.0, dtype=np.float32)
    s1_pre_vh = np.full((h, w), -19.0, dtype=np.float32)
    s1_peak_vv = np.full((h, w), -20.0, dtype=np.float32)
    s1_peak_vh = np.full((h, w), -26.0, dtype=np.float32)

    # Synthetic Optics
    s2_b03 = np.full((h, w), 0.2, dtype=np.float32)
    s2_b04 = np.full((h, w), 0.1, dtype=np.float32)
    s2_b08 = np.full((h, w), 0.05, dtype=np.float32)
    s2_b11 = np.full((h, w), 0.02, dtype=np.float32)
    s2_scl = np.full((h, w), 6, dtype=np.uint8)  # Water

    # Synthetic Aux
    hand = np.full((h, w), 1.5, dtype=np.float32)
    dem = np.full((h, w), 95.0, dtype=np.float32)
    gsw = np.full((h, w), 10.0, dtype=np.float32)

    tensor, meta = assemble_multimodal_tensor(
        s1_pre_vv=s1_pre_vv,
        s1_pre_vh=s1_pre_vh,
        s1_peak_vv=s1_peak_vv,
        s1_peak_vh=s1_peak_vh,
        s2_peak_b03=s2_b03,
        s2_peak_b04=s2_b04,
        s2_peak_b08=s2_b08,
        s2_peak_b11=s2_b11,
        s2_scl=s2_scl,
        hand_meters=hand,
        dem_or_slope=dem,
        gsw_occurrence_pct=gsw,
        filter_sar_speckle=True,
    )

    assert tensor.shape == (15, h, w)
    assert meta["num_channels"] == 15
    assert meta["is_optical_valid"] is True
    assert meta["cloud_fraction"] == 0.0
    assert len(FEATURE_CHANNEL_NAMES) == 15
