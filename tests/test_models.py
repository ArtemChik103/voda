"""Unit tests for Track A and Track B modeling components (src/models/)."""

import pytest
import numpy as np
import torch

from src.models.physical import (
    calculate_otsu_threshold,
    PhysicalHydrologyModel,
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
