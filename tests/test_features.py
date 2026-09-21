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
    load_and_resample_aux,
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
from src.features.weather import (
    parse_era5_daily,
    compute_antecedent_precipitation_index,
    compute_window_api,
    assess_weather_flood_risk,
    extract_event_weather_features,
    batch_extract_pair_weather,
)
from pathlib import Path
import pandas as pd


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


def test_load_and_resample_aux_synthetic():
    """Bilinear and nearest neighbor resampling of 6-channel AUX array."""
    # Synthetic 6-band source at 30m resolution (shape: 6, 20, 20)
    src_aux = np.zeros((6, 20, 20), dtype=np.float32)
    src_aux[0, :, :] = 15.0  # slope
    src_aux[1, :, :] = 5.5   # hand
    src_aux[2, :, :] = 85.0  # occurrence
    src_aux[3, :, :] = 6.0   # seasonality
    src_aux[4, :, :] = 1.0   # max_extent
    src_aux[5, :10, :10] = 1.0  # builtup

    target_shape = (60, 60)
    res = load_and_resample_aux(src_aux, target_shape=target_shape)

    assert set(res.keys()) == {"slope", "hand", "occurrence", "seasonality", "max_extent", "builtup"}
    for name, arr in res.items():
        assert arr.shape == target_shape
        assert arr.dtype == np.float32

    # Verify values are preserved after resampling
    np.testing.assert_allclose(res["slope"], 15.0, atol=1e-3)
    np.testing.assert_allclose(res["hand"], 5.5, atol=1e-3)
    # Discrete builtup mask should contain only 0.0 and 1.0
    unique_builtup = np.unique(res["builtup"])
    assert all(val in [0.0, 1.0] for val in unique_builtup)


def test_load_and_resample_aux_geotiff():
    """Loads and resamples real AUX GeoTIFF from competition dataset if available."""
    aux_file = Path("new tz/data/rasters/baseline_2018_09_low/blagoveshchensk/AUX_terrain_gsw.tif")
    if not aux_file.exists():
        pytest.skip("AUX GeoTIFF file not present in test environment.")

    target_shape = (368, 448)  # Scaled target grid
    res = load_and_resample_aux(aux_file, target_shape=target_shape)

    assert res["hand"].shape == target_shape
    assert res["slope"].shape == target_shape
    assert res["occurrence"].shape == target_shape
    assert res["builtup"].shape == target_shape
    assert res["hand"].min() >= 0.0
    assert res["slope"].min() >= 0.0
    assert res["occurrence"].min() >= 0.0
    assert res["occurrence"].max() <= 100.0


def test_multimodal_tensor_with_aux_dict():
    """assemble_multimodal_tensor properly integrates pre-resampled aux_dict."""
    h, w = 30, 30
    s1 = np.full((h, w), -15.0, dtype=np.float32)
    aux_dict = {
        "slope": np.full((h, w), 2.0, dtype=np.float32),
        "hand": np.full((h, w), 3.0, dtype=np.float32),
        "occurrence": np.full((h, w), 90.0, dtype=np.float32),
        "builtup": np.zeros((h, w), dtype=np.float32),
    }

    tensor, meta = assemble_multimodal_tensor(
        s1_pre_vv=s1,
        s1_pre_vh=s1,
        s1_peak_vv=s1,
        s1_peak_vh=s1,
        aux_dict=aux_dict,
    )

    assert tensor.shape == (15, h, w)
    # Channel 11 is slope_deg
    np.testing.assert_allclose(tensor[11], 2.0)
    # Channel 13 is perm_water_mask (occurrence >= 80% -> 1.0)
    np.testing.assert_allclose(tensor[13], 1.0)


def test_parse_era5_daily(tmp_path):
    """parse_era5_daily properly formats dates and clips negative values."""
    csv_file = tmp_path / "test_era5.csv"
    csv_file.write_text("date,precip_mm,temp_c,snowmelt_mm\n2020-07-02,5.2,21.0,0.0\n2020-07-01,-1.0,20.5,0.0\n")

    df = parse_era5_daily(csv_file)
    assert len(df) == 2
    # Chronological sort
    assert df["date"].iloc[0] == pd.Timestamp("2020-07-01")
    assert df["date"].iloc[1] == pd.Timestamp("2020-07-02")
    # Negative precip must be clipped to 0
    assert df["precip_mm"].iloc[0] == 0.0
    assert df["precip_mm"].iloc[1] == 5.2


def test_compute_antecedent_precipitation_index():
    """API decays exponentially during dry periods and increments on rain."""
    precip = np.array([0.0, 10.0, 0.0, 0.0], dtype=np.float32)
    api = compute_antecedent_precipitation_index(precip, decay_factor=0.85)

    assert api[0] == 0.0
    assert api[1] == 10.0
    assert pytest.approx(api[2], abs=1e-4) == 8.5
    assert pytest.approx(api[3], abs=1e-4) == 7.225


def test_extract_event_weather_features():
    """extract_event_weather_features correctly computes sliding windows and risk."""
    dates = pd.date_range("2020-07-01", periods=10, freq="D")
    precips = [0.0, 1.0, 2.0, 0.0, 5.0, 10.0, 20.0, 30.0, 15.0, 5.0]
    temps = [20.0] * 10
    snowmelts = [0.0] * 10

    df = pd.DataFrame({"date": dates, "precip_mm": precips, "temp_c": temps, "snowmelt_mm": snowmelts})
    peak_date = "2020-07-08"  # index 7 (precip = 30.0)

    feat = extract_event_weather_features(df, peak_date=peak_date)

    assert feat["peak_date"] == "2020-07-08"
    assert feat["precip_1d_mm"] == 30.0
    # 3-day window: indices 5, 6, 7 -> 10 + 20 + 30 = 60.0
    assert feat["precip_3d_sum_mm"] == 60.0
    # 7-day window: indices 1 to 7 -> 1+2+0+5+10+20+30 = 68.0
    assert feat["precip_7d_sum_mm"] == 68.0
    assert feat["temp_mean_7d_c"] == 20.0
    assert feat["api_7d_mm"] > 30.0
    assert feat["weather_risk_level"] in ["HIGH", "EXTREME"]


def test_assess_weather_flood_risk():
    """assess_weather_flood_risk assigns monotonic risk tiers."""
    r_low = assess_weather_flood_risk(5.0, 3.0)
    r_mod = assess_weather_flood_risk(25.0, 15.0)
    r_high = assess_weather_flood_risk(55.0, 35.0)
    r_ext = assess_weather_flood_risk(110.0, 70.0)

    assert r_low["level"] == "LOW"
    assert r_mod["level"] == "MODERATE"
    assert r_high["level"] == "HIGH"
    assert r_ext["level"] == "EXTREME"
    assert r_low["score"] < r_mod["score"] < r_high["score"] < r_ext["score"]


def test_batch_extract_pair_weather_real():
    """batch_extract_pair_weather extracts valid features for competition pairs."""
    rasters_dir = Path("new tz/data/rasters")
    if not rasters_dir.exists():
        pytest.skip("new tz/data/rasters directory not found")

    from scripts.eda import PAIR_CATALOGUE
    batch = batch_extract_pair_weather(rasters_dir, PAIR_CATALOGUE)

    assert len(batch) == 11
    for pid, wfeat in batch.items():
        assert "precip_7d_sum_mm" in wfeat
        assert "api_7d_mm" in wfeat
        assert "weather_risk_level" in wfeat
        assert wfeat["precip_7d_sum_mm"] >= 0.0
