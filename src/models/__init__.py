"""Models package for KosmoHackathon 2026."""

from src.models.physical import (
    PhysicalHydrologyModel,
    calculate_otsu_threshold,
)
from src.models.deep_learning import (
    MultiModalHydrologyNet,
    CombinedHydrologicalLoss,
    FocalLoss,
    LovaszLoss,
    BoundaryLoss,
)
from src.models.dataset import (
    HydrologyDataset,
    get_loao_splits,
    AOI_MAPPING,
)
from src.models.inference import (
    HydrologyInferenceEngine,
)

__all__ = [
    "PhysicalHydrologyModel",
    "calculate_otsu_threshold",
    "MultiModalHydrologyNet",
    "CombinedHydrologicalLoss",
    "FocalLoss",
    "LovaszLoss",
    "BoundaryLoss",
    "HydrologyDataset",
    "get_loao_splits",
    "AOI_MAPPING",
    "HydrologyInferenceEngine",
]
