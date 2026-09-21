"""Unit tests for Track A and Track B modeling components (src/models/)."""

import pytest
import numpy as np
import torch

from src.models.physical import (
    calculate_otsu_threshold,
    PhysicalHydrologyModel,
    filter_urban_false_alarms,
    filter_permanent_water_gsw,
    filter_waterlogged_cropland,
    filter_dry_sandbars,
)
from src.models.deep_learning import (
    MultiModalHydrologyNet,
    CombinedHydrologicalLoss,
    FocalLoss,
    LovaszLoss,
)
from src.models.dataset import (
    get_loao_splits,
    HydrologyDataset,
)
from src.models.inference import (
    HydrologyInferenceEngine,
)
from src.metrics.score import ALL_PAIRS


def test_otsu_threshold_bimodal():
    """Otsu thresholding should separate two clear Gaussian backscatter modes."""
    np.random.seed(42)
    # Mode 1: Water around -21 dB
    water = np.random.normal(-21.0, 1.0, 1000)
    # Mode 2: Land around -13 dB
    land = np.random.normal(-13.0, 1.0, 3000)
    combined = np.concatenate([water, land])

    thresh = calculate_otsu_threshold(combined, val_min=-24.0, val_max=-11.0)
    # The optimal threshold should fall cleanly between the two modes
    assert -18.5 <= thresh <= -15.5


def test_physical_model_flood_detection():
    """Physical model must detect flood and enforce flood_ha <= water_peak_ha."""
    h, w = 100, 100
    model = PhysicalHydrologyModel()

    # Pre-event: channel in center
    s1_pre_vv = np.full((h, w), -12.0, dtype=np.float32)
    s1_pre_vh = np.full((h, w), -19.0, dtype=np.float32)
    s1_pre_vv[40:60, :] = -22.0
    s1_pre_vh[40:60, :] = -28.0

    # Peak-event: river overflows into rows 20:80
    s1_peak_vv = s1_pre_vv.copy()
    s1_peak_vh = s1_pre_vh.copy()
    s1_peak_vv[20:80, :] = -21.0
    s1_peak_vh[20:80, :] = -27.0

    hand = np.zeros((h, w), dtype=np.float32)
    perm_water = np.zeros((h, w), dtype=np.uint8)
    perm_water[45:55, :] = 1  # central core is permanent water

    res = model.detect_flood(
        s1_pre_vv_db=s1_pre_vv,
        s1_pre_vh_db=s1_pre_vh,
        s1_peak_vv_db=s1_peak_vv,
        s1_peak_vh_db=s1_peak_vh,
        hand_meters=hand,
        perm_water_mask=perm_water,
    )

    assert "flood_mask" in res
    assert "flood_ha" in res
    assert res["flood_ha"] <= res["water_peak_ha"]
    assert res["flood_ha"] > 0.0  # Detected the inundation zone


def test_deep_learning_forward_and_backward():
    """Net forward pass and combined loss backward pass must compute valid gradients."""
    net = MultiModalHydrologyNet(in_channels=15, base_filters=16)
    loss_fn = CombinedHydrologicalLoss()

    inp = torch.randn(2, 15, 64, 64, dtype=torch.float32)
    target_flood = torch.randint(0, 2, (2, 1, 64, 64), dtype=torch.float32)

    net.train()
    out = net(inp)

    assert out["logits_flood"].shape == (2, 1, 64, 64)
    assert out["prob_flood"].shape == (2, 1, 64, 64)

    loss_dict = loss_fn(out["logits_flood"], target_flood)
    loss = loss_dict["total_loss"]
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    loss.backward()
    # Check gradient exists on head weights
    assert net.head_flood.weight.grad is not None


def test_loao_splits():
    """Leave-One-AOI-Out split must produce disjoint sets covering all pairs."""
    train_pairs, val_pairs = get_loao_splits("blagoveshchensk")
    assert len(train_pairs) > 0
    assert len(val_pairs) > 0

    # Disjoint check
    assert len(set(train_pairs).intersection(set(val_pairs))) == 0
    # Complete coverage check
    assert set(train_pairs).union(set(val_pairs)) == set(ALL_PAIRS)


def test_inference_engine_end_to_end():
    """Inference engine with tiled stitching must return valid outputs."""
    net = MultiModalHydrologyNet(in_channels=15, base_filters=16)
    engine = HydrologyInferenceEngine(deep_model=net, tile_size=64, overlap=16)

    h, w = 100, 100
    s1_pre_vv = np.full((h, w), -12.0, dtype=np.float32)
    s1_pre_vh = np.full((h, w), -19.0, dtype=np.float32)
    s1_peak_vv = np.full((h, w), -20.0, dtype=np.float32)
    s1_peak_vh = np.full((h, w), -27.0, dtype=np.float32)
    hand = np.full((h, w), 2.0, dtype=np.float32)
    perm_water = np.zeros((h, w), dtype=np.uint8)

    res = engine.predict_scene(
        s1_pre_vv=s1_pre_vv,
        s1_pre_vh=s1_pre_vh,
        s1_peak_vv=s1_peak_vv,
        s1_peak_vh=s1_peak_vh,
        hand_meters=hand,
        perm_water_mask=perm_water,
    )

    assert res["flood_mask"].shape == (h, w)
    assert res["flood_ha"] <= res["water_peak_ha"]
    assert res["flood_mask"].dtype == np.uint8


def test_trap_airport_builtup_filter():
    """Airport runway with low backscatter and high HAND is eliminated."""
    h, w = 40, 40
    raw_flood = np.ones((h, w), dtype=np.uint8)
    builtup = np.zeros((h, w), dtype=np.float32)
    hand = np.zeros((h, w), dtype=np.float32)

    # Place airport runway in top half (builtup = 1, HAND = 18m)
    builtup[:20, :] = 1.0
    hand[:20, :] = 18.0

    # Low floodplain in bottom half (builtup = 0, HAND = 2m)
    hand[20:, :] = 2.0

    filtered, trap = filter_urban_false_alarms(raw_flood, builtup, hand, airport_hand_threshold_m=12.0)
    assert np.all(filtered[:20, :] == 0)   # Airport eliminated
    assert np.all(filtered[20:, :] == 1)   # Floodplain preserved
    assert np.sum(trap) == 20 * 40


def test_trap_oxbow_lake_permanent_water():
    """Oxbow lake with GSW occurrence >= 80% is excluded from new flood."""
    h, w = 30, 30
    raw_flood = np.ones((h, w), dtype=np.uint8)
    occurrence = np.zeros((h, w), dtype=np.float32)

    # Oxbow lake patch
    occurrence[10:20, 10:20] = 88.0

    filtered, trap = filter_permanent_water_gsw(raw_flood, occurrence, occurrence_threshold_pct=80.0)
    assert np.all(filtered[10:20, 10:20] == 0)
    assert np.sum(filtered) == (30 * 30) - (10 * 10)
    assert np.sum(trap) == 100


def test_trap_waterlogged_cropland():
    """Saturated agricultural soil (moderate MNDWI, high NDVI) is filtered out."""
    h, w = 20, 20
    raw_flood = np.ones((h, w), dtype=np.uint8)

    # Saturated field: MNDWI=0.10, NDVI=0.35, small drop delta_vv=-1.5 dB
    mndwi = np.full((h, w), 0.10, dtype=np.float32)
    ndvi = np.full((h, w), 0.35, dtype=np.float32)
    delta_vv = np.full((h, w), -1.5, dtype=np.float32)

    # Genuine flood in lower half: MNDWI=0.25, NDVI=0.05, delta_vv=-5.0 dB
    mndwi[10:, :] = 0.25
    ndvi[10:, :] = 0.05
    delta_vv[10:, :] = -5.0

    filtered, trap = filter_waterlogged_cropland(
        raw_flood, mndwi, ndvi, delta_vv,
        mndwi_water_threshold=0.15,
        ndvi_cropland_threshold=0.20,
        delta_vv_min_drop_db=-3.5,
    )
    assert np.all(filtered[:10, :] == 0)  # Cropland excluded
    assert np.all(filtered[10:, :] == 1)  # Genuine flood preserved


def test_trap_dry_sandbars():
    """Dry quartz sand (max_extent=0, HAND>8m, or high B04 with negative MNDWI) is filtered."""
    h, w = 20, 20
    raw_flood = np.ones((h, w), dtype=np.uint8)
    max_extent = np.zeros((h, w), dtype=np.float32)
    hand = np.full((h, w), 10.0, dtype=np.float32)

    # Historical flood zone in bottom half
    max_extent[10:, :] = 1.0

    filtered, trap = filter_dry_sandbars(raw_flood, max_extent, hand, sandbar_hand_threshold_m=8.0)
    assert np.all(filtered[:10, :] == 0)  # Dry sand excluded
    assert np.all(filtered[10:, :] == 1)  # Real flood zone preserved


def test_physical_model_with_traps_integrated():
    """PhysicalHydrologyModel properly runs all trap filters and returns trap_stats."""
    h, w = 60, 60
    model = PhysicalHydrologyModel()

    s1_pre = np.full((h, w), -12.0, dtype=np.float32)
    s1_peak = np.full((h, w), -22.0, dtype=np.float32)
    hand = np.full((h, w), 2.0, dtype=np.float32)
    perm_water = np.zeros((h, w), dtype=np.uint8)

    # Add airport trap in upper-left (builtup=1, HAND=15)
    builtup = np.zeros((h, w), dtype=np.float32)
    builtup[:15, :15] = 1.0
    hand[:15, :15] = 15.0

    # Add oxbow lake in upper-right (GSW occurrence = 95%)
    gsw_occ = np.zeros((h, w), dtype=np.float32)
    gsw_occ[:15, 45:] = 95.0

    res = model.detect_flood(
        s1_pre_vv_db=s1_pre,
        s1_pre_vh_db=s1_pre - 6.0,
        s1_peak_vv_db=s1_peak,
        s1_peak_vh_db=s1_peak - 6.0,
        hand_meters=hand,
        perm_water_mask=perm_water,
        builtup_mask=builtup,
        gsw_occurrence_pct=gsw_occ,
        weather_features={"weather_risk_level": "HIGH", "precip_7d_sum_mm": 65.0},
    )

    assert "trap_stats" in res
    assert res["flood_ha"] > 0.0
    # Airport runway pixels must be 0 in flood mask
    assert np.all(res["flood_mask"][:15, :15] == 0)
    # Oxbow lake pixels must be 0 in flood mask
    assert np.all(res["flood_mask"][:15, 45:] == 0)
