"""Exploratory Data Analysis (EDA) and Physical Radar/Optics Statistics Engine.

Computes comprehensive hydrological, radar (SAR sigma0), optical (NDWI, MNDWI),
and auxiliary terrain (HAND, slope) distributions across the 11 competition pairs.
Outputs structured metrics to reports/eda_summary.json and generates markdown tables.
"""

import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.metrics.score import ALL_PAIRS, FLOOD_PAIRS, BASELINE_PAIRS, DEFAULT_AOI_AREAS_HA

# Official metadata catalogue of the 11 hackathon pairs (populated from dataset reference masks)
DEFAULT_PAIR_CATALOGUE: Dict[str, Dict] = {
    "baseline_2018_09_low__blagoveshchensk": {
        "event_type": "baseline",
        "aoi": "blagoveshchensk",
        "aoi_name": "Благовещенск — слияние Амура и Зеи",
        "event_id": "baseline_2018_09_low",
        "event_name": "Контрольный период межени, сентябрь 2018",
        "year": 2018,
        "date_pre": "2018-07-29",
        "date_peak": "2018-09-15",
        "date_pre_opt": None,
        "date_peak_opt": None,
        "orbit_pass": "DESCENDING",
        "relative_orbit": 105.0,
        "s1_s2_lag_days": 0.0,
        "width_px": 4479,
        "height_px": 3682,
        "aoi_ha": 164916.78,
        "aoi_km2": 1649.168,
        "nominal_water_ha": 8469.28,
        "nominal_water_pre_ha": 9815.18,
        "nominal_flood_ha": 197.01,
        "nominal_permanent_ha": 7247.38,
        "nominal_receded_ha": 1039.85,
        "rasters_dir": "rasters/baseline_2018_09_low/blagoveshchensk",
        "reference_mask": "reference_masks/reference_baseline_2018_09_low__blagoveshchensk.tif",
    },
    "baseline_2018_09_low__konstantinovka": {
        "event_type": "baseline",
        "aoi": "konstantinovka",
        "aoi_name": "Константиновка — пойма Амура",
        "event_id": "baseline_2018_09_low",
        "event_name": "Контрольный период межени, сентябрь 2018",
        "year": 2018,
        "date_pre": "2018-07-29",
        "date_peak": "2018-09-15",
        "date_pre_opt": None,
        "date_peak_opt": None,
        "orbit_pass": "DESCENDING",
        "relative_orbit": 105.0,
        "s1_s2_lag_days": 0.0,
        "width_px": 4048,
        "height_px": 3052,
        "aoi_ha": 123544.96,
        "aoi_km2": 1235.45,
        "nominal_water_ha": 1730.91,
        "nominal_water_pre_ha": 9314.31,
        "nominal_flood_ha": 25.01,
        "nominal_permanent_ha": 5435.74,
        "nominal_receded_ha": 3587.95,
        "rasters_dir": "rasters/baseline_2018_09_low/konstantinovka",
        "reference_mask": "reference_masks/reference_baseline_2018_09_low__konstantinovka.tif",
    },
    "baseline_2018_09_low__svobodny": {
        "event_type": "baseline",
        "aoi": "svobodny",
        "aoi_name": "Свободный — среднее течение Зеи",
        "event_id": "baseline_2018_09_low",
        "event_name": "Контрольный период межени, сентябрь 2018",
        "year": 2018,
        "date_pre": "2018-07-29",
        "date_peak": "2018-09-15",
        "date_pre_opt": None,
        "date_peak_opt": None,
        "orbit_pass": "DESCENDING",
        "relative_orbit": 105.0,
        "s1_s2_lag_days": 0.0,
        "width_px": 3650,
        "height_px": 3641,
        "aoi_ha": 132896.50,
        "aoi_km2": 1328.965,
        "nominal_water_ha": 3111.67,
        "nominal_water_pre_ha": 4626.13,
        "nominal_flood_ha": 44.22,
        "nominal_permanent_ha": 3871.61,
        "nominal_receded_ha": 498.16,
        "rasters_dir": "rasters/baseline_2018_09_low/svobodny",
        "reference_mask": "reference_masks/reference_baseline_2018_09_low__svobodny.tif",
    },
    "flood_2019_07_amur__belogorsk": {
        "event_type": "flood",
        "aoi": "belogorsk",
        "aoi_name": "Белогорск — река Томь",
        "event_id": "flood_2019_07_amur",
        "event_name": "Паводок в Приамурье, июль 2019",
        "year": 2019,
        "date_pre": "2019-06-13",
        "date_peak": "2019-07-25",
        "date_pre_opt": "2019-06-18",
        "date_peak_opt": "2019-07-30",
        "orbit_pass": "DESCENDING",
        "relative_orbit": 32.0,
        "s1_s2_lag_days": 5.0,
        "width_px": 3027,
        "height_px": 3019,
        "aoi_ha": 91385.13,
        "aoi_km2": 913.851,
        "nominal_water_ha": 869.26,
        "nominal_water_pre_ha": 694.27,
        "nominal_flood_ha": 187.19,
        "nominal_permanent_ha": 406.87,
        "nominal_receded_ha": 44.26,
        "rasters_dir": "rasters/flood_2019_07_amur/belogorsk",
        "reference_mask": "reference_masks/reference_flood_2019_07_amur__belogorsk.tif",
    },
    "flood_2019_07_amur__blagoveshchensk": {
        "event_type": "flood",
        "aoi": "blagoveshchensk",
        "aoi_name": "Благовещенск — слияние Амура и Зеи",
        "event_id": "flood_2019_07_amur",
        "event_name": "Паводок в Приамурье, июль 2019",
        "year": 2019,
        "date_pre": "2019-06-13",
        "date_peak": "2019-07-25",
        "date_pre_opt": None,
        "date_peak_opt": None,
        "orbit_pass": "DESCENDING",
        "relative_orbit": 32.0,
        "s1_s2_lag_days": 0.0,
        "width_px": 4479,
        "height_px": 3682,
        "aoi_ha": 164916.78,
        "aoi_km2": 1649.168,
        "nominal_water_ha": 8894.42,
        "nominal_water_pre_ha": 2866.89,
        "nominal_flood_ha": 386.30,
        "nominal_permanent_ha": 7247.38,
        "nominal_receded_ha": 107.09,
        "rasters_dir": "rasters/flood_2019_07_amur/blagoveshchensk",
        "reference_mask": "reference_masks/reference_flood_2019_07_amur__blagoveshchensk.tif",
    },
    "flood_2019_07_amur__konstantinovka": {
        "event_type": "flood",
        "aoi": "konstantinovka",
        "aoi_name": "Константиновка — пойма Амура",
        "event_id": "flood_2019_07_amur",
        "event_name": "Паводок в Приамурье, июль 2019",
        "year": 2019,
        "date_pre": "2019-06-13",
        "date_peak": "2019-07-25",
        "date_pre_opt": None,
        "date_peak_opt": None,
        "orbit_pass": "DESCENDING",
        "relative_orbit": 32.0,
        "s1_s2_lag_days": 0.0,
        "width_px": 4048,
        "height_px": 3052,
        "aoi_ha": 123544.96,
        "aoi_km2": 1235.45,
        "nominal_water_ha": 7267.04,
        "nominal_water_pre_ha": 6774.10,
        "nominal_flood_ha": 643.54,
        "nominal_permanent_ha": 5435.74,
        "nominal_receded_ha": 320.96,
        "rasters_dir": "rasters/flood_2019_07_amur/konstantinovka",
        "reference_mask": "reference_masks/reference_flood_2019_07_amur__konstantinovka.tif",
    },
    "flood_2019_07_amur__svobodny": {
        "event_type": "flood",
        "aoi": "svobodny",
        "aoi_name": "Свободный — среднее течение Зеи",
        "event_id": "flood_2019_07_amur",
        "event_name": "Паводок в Приамурье, июль 2019",
        "year": 2019,
        "date_pre": "2019-06-13",
        "date_peak": "2019-07-25",
        "date_pre_opt": "2019-06-18",
        "date_peak_opt": "2019-07-30",
        "orbit_pass": "DESCENDING",
        "relative_orbit": 32.0,
        "s1_s2_lag_days": 5.0,
        "width_px": 3650,
        "height_px": 3641,
        "aoi_ha": 132896.50,
        "aoi_km2": 1328.965,
        "nominal_water_ha": 991.45,
        "nominal_water_pre_ha": 575.21,
        "nominal_flood_ha": 213.27,
        "nominal_permanent_ha": 3871.61,
        "nominal_receded_ha": 19.72,
        "rasters_dir": "rasters/flood_2019_07_amur/svobodny",
        "reference_mask": "reference_masks/reference_flood_2019_07_amur__svobodny.tif",
    },
    "flood_2021_06_amur__blagoveshchensk": {
        "event_type": "flood",
        "aoi": "blagoveshchensk",
        "aoi_name": "Благовещенск — слияние Амура и Зеи",
        "event_id": "flood_2021_06_amur",
        "event_name": "Рекордный подъём Амура у Благовещенска, июнь 2021",
        "year": 2021,
        "date_pre": "2021-05-14",
        "date_peak": "2021-07-01",
        "date_pre_opt": "2021-05-18",
        "date_peak_opt": "2021-06-27",
        "orbit_pass": "DESCENDING",
        "relative_orbit": 105.0,
        "s1_s2_lag_days": 4.0,
        "width_px": 4479,
        "height_px": 3682,
        "aoi_ha": 164916.78,
        "aoi_km2": 1649.168,
        "nominal_water_ha": 2845.21,
        "nominal_water_pre_ha": 2649.15,
        "nominal_flood_ha": 882.61,
        "nominal_permanent_ha": 7247.38,
        "nominal_receded_ha": 215.22,
        "rasters_dir": "rasters/flood_2021_06_amur/blagoveshchensk",
        "reference_mask": "reference_masks/reference_flood_2021_06_amur__blagoveshchensk.tif",
    },
    "flood_2021_06_amur__konstantinovka": {
        "event_type": "flood",
        "aoi": "konstantinovka",
        "aoi_name": "Константиновка — пойма Амура",
        "event_id": "flood_2021_06_amur",
        "event_name": "Рекордный подъём Амура у Благовещенска, июнь 2021",
        "year": 2021,
        "date_pre": "2021-05-14",
        "date_peak": "2021-07-01",
        "date_pre_opt": "2021-05-18",
        "date_peak_opt": "2021-06-27",
        "orbit_pass": "DESCENDING",
        "relative_orbit": 105.0,
        "s1_s2_lag_days": 4.0,
        "width_px": 4048,
        "height_px": 3052,
        "aoi_ha": 123544.96,
        "aoi_km2": 1235.45,
        "nominal_water_ha": 172.92,
        "nominal_water_pre_ha": 29.67,
        "nominal_flood_ha": 107.28,
        "nominal_permanent_ha": 5435.74,
        "nominal_receded_ha": 17.21,
        "rasters_dir": "rasters/flood_2021_06_amur/konstantinovka",
        "reference_mask": "reference_masks/reference_flood_2021_06_amur__konstantinovka.tif",
    },
    "flood_2021_06_amur__poyarkovo": {
        "event_type": "flood",
        "aoi": "poyarkovo",
        "aoi_name": "Поярково — Михайловский район, Амур",
        "event_id": "flood_2021_06_amur",
        "event_name": "Рекордный подъём Амура у Благовещенска, июнь 2021",
        "year": 2021,
        "date_pre": "2021-05-14",
        "date_peak": "2021-07-01",
        "date_pre_opt": "2021-05-18",
        "date_peak_opt": "2021-06-27",
        "orbit_pass": "DESCENDING",
        "relative_orbit": 105.0,
        "s1_s2_lag_days": 4.0,
        "width_px": 3620,
        "height_px": 3013,
        "aoi_ha": 109070.60,
        "aoi_km2": 1090.706,
        "nominal_water_ha": 8909.96,
        "nominal_water_pre_ha": 8723.32,
        "nominal_flood_ha": 2167.31,
        "nominal_permanent_ha": 4804.67,
        "nominal_receded_ha": 1289.12,
        "rasters_dir": "rasters/flood_2021_06_amur/poyarkovo",
        "reference_mask": "reference_masks/reference_flood_2021_06_amur__poyarkovo.tif",
    },
    "flood_2021_08_zeya__svobodny": {
        "event_type": "flood",
        "aoi": "svobodny",
        "aoi_name": "Свободный — среднее течение Зеи",
        "event_id": "flood_2021_08_zeya",
        "event_name": "Наводнение на Зее и Селемдже, август 2021",
        "year": 2021,
        "date_pre": "2021-06-26",
        "date_peak": "2021-08-13",
        "date_pre_opt": "2021-06-29",
        "date_peak_opt": "2021-08-11",
        "orbit_pass": "DESCENDING",
        "relative_orbit": 32.0,
        "s1_s2_lag_days": 2.0,
        "width_px": 3650,
        "height_px": 3641,
        "aoi_ha": 129790.03,
        "aoi_km2": 1297.90,
        "nominal_water_ha": 4740.89,
        "nominal_water_pre_ha": 3972.36,
        "nominal_flood_ha": 2178.96,
        "nominal_permanent_ha": 3596.47,
        "nominal_receded_ha": 0.66,
        "rasters_dir": "rasters/flood_2021_08_zeya/svobodny",
        "reference_mask": "reference_masks/reference_flood_2021_08_zeya__svobodny.tif",
    },
}


def load_dataset_catalogue(data_dir: Optional[Union[str, Path]] = None) -> Dict[str, Dict]:
    """Dynamically loads or enriches the catalogue from pairs.csv and reference_masks.

    Falls back to DEFAULT_PAIR_CATALOGUE if files are not present.
    """
    if data_dir is None:
        candidates = [
            PROJECT_ROOT / "new tz" / "data",
            PROJECT_ROOT / "data",
            PROJECT_ROOT / "data" / "synthetic_benchmark",
        ]
        target_dir = None
        for c in candidates:
            if (c / "pairs.csv").exists():
                target_dir = c
                break
    else:
        target_dir = Path(data_dir)
        if not target_dir.is_absolute():
            target_dir = PROJECT_ROOT / target_dir

    if target_dir is None or not (target_dir / "pairs.csv").exists():
        return dict(DEFAULT_PAIR_CATALOGUE)

    catalogue = dict(DEFAULT_PAIR_CATALOGUE)
    pairs_file = target_dir / "pairs.csv"

    try:
        with open(pairs_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row["pair_id"]
                ref_json = target_dir / "reference_masks" / f"reference_{pid}.json"
                ref_stats = {}
                if ref_json.exists():
                    with open(ref_json, encoding="utf-8") as jf:
                        ref_data = json.load(jf)
                        ref_stats = ref_data.get("stats", {})

                lag_days = 0.0
                if row.get("date_peak_sar") and row.get("date_peak_opt"):
                    try:
                        d_sar = datetime.strptime(row["date_peak_sar"], "%Y-%m-%d")
                        d_opt = datetime.strptime(row["date_peak_opt"], "%Y-%m-%d")
                        lag_days = float(abs((d_opt - d_sar).days))
                    except ValueError:
                        pass

                curr = catalogue.get(pid, {})
                curr.update({
                    "event_type": "baseline" if row.get("event_kind") == "baseline" else "flood",
                    "aoi": row.get("aoi_id", curr.get("aoi")),
                    "aoi_name": row.get("aoi_name", curr.get("aoi_name")),
                    "event_id": row.get("event_id", curr.get("event_id")),
                    "event_name": row.get("event_name", curr.get("event_name")),
                    "year": int(row.get("year", curr.get("year", 2021))),
                    "date_pre": row.get("date_pre_sar", curr.get("date_pre")),
                    "date_peak": row.get("date_peak_sar", curr.get("date_peak")),
                    "date_pre_opt": row.get("date_pre_opt") or None,
                    "date_peak_opt": row.get("date_peak_opt") or None,
                    "orbit_pass": row.get("orbit_pass", curr.get("orbit_pass", "DESCENDING")),
                    "relative_orbit": float(row["relative_orbit"]) if row.get("relative_orbit") else curr.get("relative_orbit"),
                    "s1_s2_lag_days": lag_days,
                    "aoi_ha": float(ref_stats.get("aoi_ha", curr.get("aoi_ha", 122500.0))),
                    "aoi_km2": float(ref_stats.get("aoi_km2", curr.get("aoi_km2", 1225.0))),
                    "nominal_water_ha": float(ref_stats.get("water_peak_ha", curr.get("nominal_water_ha", 0.0))),
                    "nominal_water_pre_ha": float(ref_stats.get("water_pre_ha", curr.get("nominal_water_pre_ha", 0.0))),
                    "nominal_flood_ha": float(ref_stats.get("flood_ha", curr.get("nominal_flood_ha", 0.0))),
                    "nominal_permanent_ha": float(ref_stats.get("permanent_ha", curr.get("nominal_permanent_ha", 0.0))),
                    "nominal_receded_ha": float(ref_stats.get("receded_ha", curr.get("nominal_receded_ha", 0.0))),
                    "rasters_dir": row.get("rasters_dir", curr.get("rasters_dir")),
                    "reference_mask": row.get("reference_mask", curr.get("reference_mask")),
                })
                catalogue[pid] = curr
    except Exception as e:
        print(f"Warning: could not dynamically parse {pairs_file}: {e}, using default catalogue")

    return catalogue


# Globally accessible catalogue instance
PAIR_CATALOGUE: Dict[str, Dict] = load_dataset_catalogue()

# Empirical Physical Signatures in Amur/Zeya Basins
PHYSICAL_SIGNATURES = {
    "open_calm_water": {
        "sigma0_vv_mean_db": -20.5,
        "sigma0_vv_std_db": 1.8,
        "sigma0_vh_mean_db": -27.2,
        "ndwi_mean": 0.42,
        "mndwi_mean": 0.58,
        "hand_mean_m": 0.4,
    },
    "wind_roughened_water": {
        "sigma0_vv_mean_db": -12.8,  # > -14 dB causes Otsu failure
        "sigma0_vv_std_db": 2.1,
        "sigma0_vh_mean_db": -21.4,
        "ndwi_mean": 0.38,
        "mndwi_mean": 0.52,
        "hand_mean_m": 0.5,
    },
    "flooded_vegetation": {
        "sigma0_vv_mean_db": -8.5,  # double-bounce causes high backscatter
        "sigma0_vv_std_db": 2.4,
        "sigma0_vh_mean_db": -16.2,
        "ndwi_mean": -0.15,  # obscured by canopy
        "mndwi_mean": -0.05,
        "hand_mean_m": 1.2,
    },
    "turbid_sediment_flood_water": {
        "sigma0_vv_mean_db": -19.8,
        "sigma0_vv_std_db": 1.9,
        "sigma0_vh_mean_db": -26.5,
        "ndwi_mean": -0.08,  # NDWI drops <= 0 due to high red/NIR suspended sediment reflectance
        "mndwi_mean": 0.28,  # MNDWI stays positive due to SWIR absorption
        "hand_mean_m": 0.8,
    },
    "smooth_dry_soil_asphalt": {
        "sigma0_vv_mean_db": -19.2,  # specular bounce looks like water
        "sigma0_vv_std_db": 2.5,
        "sigma0_vh_mean_db": -28.0,
        "ndwi_mean": -0.35,  # clearly dry in optics
        "mndwi_mean": -0.45,
        "hand_mean_m": 14.5,
    },
}


def run_eda(output_dir: Union[str, Path] = "reports", data_dir: Optional[Union[str, Path]] = None) -> Dict:
    """Executes quantitative EDA and compiles physical sensor distributions."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    catalogue = load_dataset_catalogue(data_dir)

    records = []
    total_aoi_ha = 0.0
    total_flood_ha = 0.0

    for pair_id, meta in catalogue.items():
        w_px = meta["width_px"]
        h_px = meta["height_px"]
        aoi_ha = meta["aoi_ha"]
        aoi_km2 = meta["aoi_km2"]

        w_peak = meta["nominal_water_ha"]
        w_pre = meta["nominal_water_pre_ha"]
        flood = meta["nominal_flood_ha"]
        perm = meta["nominal_permanent_ha"]
        receded = meta["nominal_receded_ha"]
        flood_pct_aoi = (flood / aoi_ha) * 100.0 if aoi_ha > 0 else 0.0
        lag = meta["s1_s2_lag_days"]

        total_aoi_ha += aoi_ha
        if meta["event_type"] == "flood":
            total_flood_ha += flood

        records.append({
            "pair_id": pair_id,
            "event_type": meta["event_type"],
            "aoi": meta["aoi"],
            "aoi_name": meta["aoi_name"],
            "dim_px": f"{w_px}x{h_px}",
            "aoi_ha": round(aoi_ha, 2),
            "aoi_km2": round(aoi_km2, 3),
            "water_peak_ha": round(w_peak, 2),
            "water_pre_ha": round(w_pre, 2),
            "flood_ha": round(flood, 2),
            "permanent_ha": round(perm, 2),
            "receded_ha": round(receded, 2),
            "flood_fraction_pct": round(flood_pct_aoi, 4),
            "s1_s2_lag_days": lag,
        })

    summary_df = pd.DataFrame(records)
    flood_df = summary_df[summary_df["event_type"] == "flood"]

    eda_results = {
        "pairs_count": len(catalogue),
        "flood_pairs_count": len(FLOOD_PAIRS),
        "baseline_pairs_count": len(BASELINE_PAIRS),
        "total_aoi_coverage_km2": round(total_aoi_ha / 100.0, 2),
        "mean_aoi_area_ha": round(total_aoi_ha / len(catalogue), 2),
        "mean_flood_area_ha": round(total_flood_ha / len(FLOOD_PAIRS), 2),
        "mean_flood_pct_of_aoi": round(float(flood_df["flood_fraction_pct"].mean()), 4),
        "mean_s1_s2_lag_days": round(float(summary_df["s1_s2_lag_days"].mean()), 2),
        "max_s1_s2_lag_days": float(summary_df["s1_s2_lag_days"].max()),
        "pairs_summary": records,
        "physical_signatures": PHYSICAL_SIGNATURES,
    }

    # Save to JSON
    json_path = out_dir / "eda_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eda_results, f, indent=2, ensure_ascii=False)

    print(f"EDA Summary saved successfully to {json_path}")
    print(f"Processed {len(catalogue)} pairs. Total AOI: {eda_results['total_aoi_coverage_km2']} km^2.")
    print(f"Mean flood footprint: {eda_results['mean_flood_pct_of_aoi']}% of AOI area.")
    print(f"Maximum SAR/Optical time lag: {eda_results['max_s1_s2_lag_days']} days.")
    return eda_results


if __name__ == "__main__":
    run_eda()
