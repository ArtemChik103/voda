"""Competition evaluation metric engine for KosmoHackathon 2026.

Implements the exact evaluation criteria specified in Section 12 of the case:
Score = 0.45 * Q_flood + 0.25 * Q_water_peak + 0.15 * Q_water_pre + 0.15 * Spec_base

where:
  q = max(0.0, 1.0 - |X_pred - X_true| / max(X_true, threshold))
  - threshold for flood_ha: 50.0 ha
  - threshold for water_peak_ha / water_pre_ha: 200.0 ha
  - Q is the mean of q across the 8 flood pairs.
  - Spec_base is evaluated across the 3 dry season baseline pairs:
      fraction = max(0.0, flood_pred - flood_true) / AOI_area
      spec_j = 1.0 - min(1.0, fraction / 0.005)
      Spec_base = mean(spec_j)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# 8 Official Flood Pairs
FLOOD_PAIRS: List[str] = [
    "flood_2019_07_amur__belogorsk",
    "flood_2019_07_amur__blagoveshchensk",
    "flood_2019_07_amur__konstantinovka",
    "flood_2019_07_amur__svobodny",
    "flood_2021_06_amur__blagoveshchensk",
    "flood_2021_06_amur__konstantinovka",
    "flood_2021_06_amur__poyarkovo",
    "flood_2021_08_zeya__svobodny",
]

# 3 Official Dry Season Baseline Pairs
BASELINE_PAIRS: List[str] = [
    "baseline_2018_09_low__blagoveshchensk",
    "baseline_2018_09_low__konstantinovka",
    "baseline_2018_09_low__svobodny",
]

ALL_PAIRS: List[str] = BASELINE_PAIRS + FLOOD_PAIRS

# Thresholds in hectares per official formula
THRESHOLD_FLOOD_HA: float = 50.0
THRESHOLD_WATER_HA: float = 200.0

# Tolerance fraction for baseline false positives (0.5% of AOI area)
BASELINE_MAX_FRACTION: float = 0.005

# Official AOI areas in hectares from dataset reference masks
# (Exact values derived from 10 m resolution masks in EPSG:32652)
DEFAULT_AOI_AREAS_HA: Dict[str, float] = {
    "baseline_2018_09_low__blagoveshchensk": 164916.78,
    "baseline_2018_09_low__konstantinovka": 123544.96,
    "baseline_2018_09_low__svobodny": 132896.50,
    "flood_2019_07_amur__belogorsk": 91385.13,
    "flood_2019_07_amur__blagoveshchensk": 164916.78,
    "flood_2019_07_amur__konstantinovka": 123544.96,
    "flood_2019_07_amur__svobodny": 132896.50,
    "flood_2021_06_amur__blagoveshchensk": 164916.78,
    "flood_2021_06_amur__konstantinovka": 123544.96,
    "flood_2021_06_amur__poyarkovo": 109070.60,
    "flood_2021_08_zeya__svobodny": 129790.03,
}


def calculate_q_score(x_pred: float, x_true: float, threshold: float) -> float:
    """Calculates the convergence score q for a single pair and target quantity.

    Formula:
        q = max(0.0, 1.0 - |x_pred - x_true| / max(x_true, threshold))

    Args:
        x_pred: Predicted area in hectares.
        x_true: True reference area in hectares.
        threshold: Absolute tolerance threshold in hectares (50 ha for flood, 200 ha for water).

    Returns:
        q score bounded within [0.0, 1.0].
    """
    denom = max(float(x_true), float(threshold))
    error = abs(float(x_pred) - float(x_true))
    q = max(0.0, 1.0 - (error / denom))
    return float(np.clip(q, 0.0, 1.0))


def calculate_spec_base(
    pred_df: pd.DataFrame,
    true_df: pd.DataFrame,
    aoi_areas_ha: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Calculates Spec_base across the 3 baseline pairs.

    Formula:
        fraction = max(0.0, flood_pred - flood_true) / AOI_area
        spec_j = 1.0 - min(1.0, fraction / 0.005)
        Spec_base = mean(spec_j)

    A false flood exceeding 0.5% (0.005) of the AOI area yields spec_j = 0.0.

    Args:
        pred_df: DataFrame containing predictions indexed or with column 'pair_id'.
        true_df: DataFrame containing ground truth indexed or with column 'pair_id'.
        aoi_areas_ha: Optional dictionary mapping pair_id to AOI area in hectares.

    Returns:
        Tuple of (Spec_base aggregate score, dictionary of per-pair spec scores).
    """
    areas = aoi_areas_ha or DEFAULT_AOI_AREAS_HA
    p_df = pred_df.set_index("pair_id") if "pair_id" in pred_df.columns else pred_df
    t_df = true_df.set_index("pair_id") if "pair_id" in true_df.columns else true_df

    specs: Dict[str, float] = {}
    for pair in BASELINE_PAIRS:
        if pair not in p_df.index or pair not in t_df.index:
            raise KeyError(f"Baseline pair '{pair}' is missing from submission or ground truth.")

        flood_pred = float(p_df.loc[pair, "flood_ha"])
        flood_true = float(t_df.loc[pair, "flood_ha"])
        aoi_area = float(areas.get(pair, 122500.0))

        excess = max(0.0, flood_pred - flood_true)
        fraction = excess / aoi_area
        spec_val = 1.0 - min(1.0, fraction / BASELINE_MAX_FRACTION)
        specs[pair] = float(np.clip(spec_val, 0.0, 1.0))

    spec_base = float(np.mean(list(specs.values())))
    return spec_base, specs


def calculate_competition_score(
    submission: Union[pd.DataFrame, str],
    ground_truth: Union[pd.DataFrame, str],
    aoi_areas_ha: Optional[Dict[str, float]] = None,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Calculates the complete competition Score and all intermediate sub-metrics.

    Score = 0.45 * Q_flood + 0.25 * Q_water_peak + 0.15 * Q_water_pre + 0.15 * Spec_base

    Args:
        submission: DataFrame or path to submission.csv.
        ground_truth: DataFrame or path to ground_truth.csv.
        aoi_areas_ha: Optional mapping from pair_id to AOI area in hectares.

    Returns:
        Dictionary with keys:
            - 'score': Final weighted Score in [0.0, 1.0]
            - 'Q_flood': Mean q score for flood_ha across 8 flood pairs
            - 'Q_water_peak': Mean q score for water_peak_ha across 8 flood pairs
            - 'Q_water_pre': Mean q score for water_pre_ha across 8 flood pairs
            - 'Spec_base': Aggregate baseline specificity across 3 baseline pairs
            - 'q_flood_by_pair': Dict of per-pair q scores for flood
            - 'q_peak_by_pair': Dict of per-pair q scores for water_peak
            - 'q_pre_by_pair': Dict of per-pair q scores for water_pre
            - 'spec_by_pair': Dict of per-pair spec scores for baseline pairs
    """
    if isinstance(submission, str):
        sub_df = pd.read_csv(submission)
    else:
        sub_df = submission.copy()

    if isinstance(ground_truth, str):
        gt_df = pd.read_csv(ground_truth)
    else:
        gt_df = ground_truth.copy()

    if "pair_id" in sub_df.columns:
        sub_df = sub_df.set_index("pair_id")
    if "pair_id" in gt_df.columns:
        gt_df = gt_df.set_index("pair_id")

    # Verify presence of all 11 pairs
    for pair in ALL_PAIRS:
        if pair not in sub_df.index:
            raise ValueError(f"Submission is missing required pair_id: '{pair}'")
        if pair not in gt_df.index:
            raise ValueError(f"Ground truth is missing required pair_id: '{pair}'")

    # 1. Compute q for the 8 flood pairs
    q_flood_dict: Dict[str, float] = {}
    q_peak_dict: Dict[str, float] = {}
    q_pre_dict: Dict[str, float] = {}

    for pair in FLOOD_PAIRS:
        # flood_ha
        f_pred = float(sub_df.loc[pair, "flood_ha"])
        f_true = float(gt_df.loc[pair, "flood_ha"])
        q_f = calculate_q_score(f_pred, f_true, THRESHOLD_FLOOD_HA)
        q_flood_dict[pair] = q_f

        # water_peak_ha
        pk_pred = float(sub_df.loc[pair, "water_peak_ha"])
        pk_true = float(gt_df.loc[pair, "water_peak_ha"])
        q_pk = calculate_q_score(pk_pred, pk_true, THRESHOLD_WATER_HA)
        q_peak_dict[pair] = q_pk

        # water_pre_ha
        pr_pred = float(sub_df.loc[pair, "water_pre_ha"])
        pr_true = float(gt_df.loc[pair, "water_pre_ha"])
        q_pr = calculate_q_score(pr_pred, pr_true, THRESHOLD_WATER_HA)
        q_pre_dict[pair] = q_pr

    q_flood_mean = float(np.mean(list(q_flood_dict.values())))
    q_peak_mean = float(np.mean(list(q_peak_dict.values())))
    q_pre_mean = float(np.mean(list(q_pre_dict.values())))

    # 2. Compute Spec_base on the 3 baseline pairs
    spec_base, spec_dict = calculate_spec_base(sub_df, gt_df, aoi_areas_ha)

    # 3. Overall Weighted Score
    final_score = (
        0.45 * q_flood_mean
        + 0.25 * q_peak_mean
        + 0.15 * q_pre_mean
        + 0.15 * spec_base
    )

    return {
        "score": round(float(final_score), 5),
        "Q_flood": round(q_flood_mean, 5),
        "Q_water_peak": round(q_peak_mean, 5),
        "Q_water_pre": round(q_pre_mean, 5),
        "Spec_base": round(spec_base, 5),
        "q_flood_by_pair": {k: round(v, 5) for k, v in q_flood_dict.items()},
        "q_peak_by_pair": {k: round(v, 5) for k, v in q_peak_dict.items()},
        "q_pre_by_pair": {k: round(v, 5) for k, v in q_pre_dict.items()},
        "spec_by_pair": {k: round(v, 5) for k, v in spec_dict.items()},
    }
