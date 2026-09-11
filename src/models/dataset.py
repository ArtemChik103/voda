"""Spatial Leave-One-AOI-Out (LOAO) Dataset and Data Loader Module.

Ensures zero spatial data leakage across geographic regions:
- 5 AOIs: Belogorsk, Blagoveshchensk, Konstantinovka, Svobodny, Poyarkovo.
- Strict LOAO splits and baseline specificity tracking.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from src.metrics.score import ALL_PAIRS, FLOOD_PAIRS, BASELINE_PAIRS

AOI_MAPPING: Dict[str, str] = {
    "baseline_2018_09_low__blagoveshchensk": "blagoveshchensk",
    "baseline_2018_09_low__konstantinovka": "konstantinovka",
    "baseline_2018_09_low__svobodny": "svobodny",
    "flood_2019_07_amur__belogorsk": "belogorsk",
    "flood_2019_07_amur__blagoveshchensk": "blagoveshchensk",
    "flood_2019_07_amur__konstantinovka": "konstantinovka",
    "flood_2019_07_amur__svobodny": "svobodny",
    "flood_2021_06_amur__blagoveshchensk": "blagoveshchensk",
    "flood_2021_06_amur__konstantinovka": "konstantinovka",
    "flood_2021_06_amur__poyarkovo": "poyarkovo",
    "flood_2021_08_zeya__svobodny": "svobodny",
}


def get_loao_splits(val_aoi: str) -> Tuple[List[str], List[str]]:
    """Generates Leave-One-AOI-Out (LOAO) train and validation split of pair IDs.

    Args:
        val_aoi: AOI name to hold out for validation (e.g., 'svobodny').

    Returns:
        Tuple of (train_pair_ids, val_pair_ids).
    """
    val_pairs = [p for p, aoi in AOI_MAPPING.items() if aoi == val_aoi]
    train_pairs = [p for p in ALL_PAIRS if p not in val_pairs]
    return train_pairs, val_pairs


class HydrologyDataset(Dataset):
    """PyTorch Dataset yielding 15-channel feature tensors and binary ground truth targets."""

    def __init__(
        self,
        samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],  # list of (features_15ch, water_target, flood_target)
        augment: bool = True,
    ):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, target_water, target_flood = self.samples[idx]

        # Convert to float32 copies
        feat = features.copy().astype(np.float32)
        w_tgt = target_water.copy().astype(np.float32)
        f_tgt = target_flood.copy().astype(np.float32)

        # Spatial augmentations: random flips and 90-degree rotations
        if self.augment:
            if np.random.rand() > 0.5:
                # Horizontal flip
                feat = np.flip(feat, axis=2).copy()
                w_tgt = np.flip(w_tgt, axis=1).copy()
                f_tgt = np.flip(f_tgt, axis=1).copy()

            if np.random.rand() > 0.5:
                # Vertical flip
                feat = np.flip(feat, axis=1).copy()
                w_tgt = np.flip(w_tgt, axis=0).copy()
                f_tgt = np.flip(f_tgt, axis=0).copy()

            k_rot = np.random.randint(0, 4)
            if k_rot > 0:
                feat = np.rot90(feat, k=k_rot, axes=(1, 2)).copy()
                w_tgt = np.rot90(w_tgt, k=k_rot, axes=(0, 1)).copy()
                f_tgt = np.rot90(f_tgt, k=k_rot, axes=(0, 1)).copy()

        tensor_feat = torch.from_numpy(feat)
        tensor_water = torch.from_numpy(w_tgt).unsqueeze(0)  # (1, H, W)
        tensor_flood = torch.from_numpy(f_tgt).unsqueeze(0)  # (1, H, W)

        return tensor_feat, tensor_water, tensor_flood
