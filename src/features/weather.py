"""ERA5 Atmospheric and Hydrometeorological Feature Module.

Extracts hydrometeorological features from ERA5 daily reanalysis series:
- Cumulative precipitation across sliding temporal windows (1d, 3d, 7d, 14d)
- Thermal regime and snowmelt dynamics
- Antecedent Precipitation Index (API) with exponential recession decay
- Hydrometeorological flood potential assessment
"""

from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


def parse_era5_daily(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Reads and standardizes daily ERA5 reanalysis data.

    Expected columns:
        - date: YYYY-MM-DD
        - precip_mm: daily total precipitation (mm)
        - temp_c: 2m mean air temperature (deg C)
        - snowmelt_mm: daily snowmelt water equivalent (mm)

    Returns:
        Cleaned, chronologically sorted DataFrame with datetime index.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"ERA5 daily file not found: {path}")

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"Required 'date' column missing from {path}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure numeric columns are well-formed and non-negative where required
    for col in ["precip_mm", "snowmelt_mm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            df[col] = df[col].clip(lower=0.0)
        else:
            df[col] = 0.0

    if "temp_c" in df.columns:
        df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce").fillna(15.0)
    else:
        df["temp_c"] = 15.0

    return df


def compute_antecedent_precipitation_index(
    daily_precip: Union[pd.Series, np.ndarray],
    decay_factor: float = 0.85,
) -> np.ndarray:
    """Computes the continuous Antecedent Precipitation Index (API).

    Formula:
        API_t = API_{t-1} * k + P_t

    where k is the recession constant (default 0.85 for daily time steps).

    Args:
        daily_precip: 1D array or Series of daily precipitation (mm).
        decay_factor: Daily decay recession factor k in [0.70, 0.98].

    Returns:
        1D numpy array of API values of same length.
    """
    p = np.asarray(daily_precip, dtype=np.float64)
    api = np.zeros_like(p)
    if len(p) == 0:
        return api

    curr = 0.0
    for i in range(len(p)):
        curr = curr * float(decay_factor) + float(p[i])
        api[i] = curr

    return api.astype(np.float32)


def compute_window_api(
    daily_precip_window: Union[pd.Series, np.ndarray],
    decay_factor: float = 0.85,
) -> float:
    """Computes API backwards from the target peak date across a window of length N.

    Formula:
        API = sum_{i=0}^{N-1} P_{t-i} * k^i

    Args:
        daily_precip_window: Daily precipitation array ending at target date (index -1 is peak date).
        decay_factor: Recession factor k.

    Returns:
        Scalar API value in mm.
    """
    p = np.asarray(daily_precip_window, dtype=np.float64)
    if len(p) == 0:
        return 0.0
    # Reverse so index 0 is target date, index 1 is day before, etc.
    reversed_p = p[::-1]
    weights = np.power(decay_factor, np.arange(len(reversed_p)))
    return float(np.sum(reversed_p * weights))


def assess_weather_flood_risk(
    precip_7d_mm: float,
    api_7d_mm: float,
    snowmelt_7d_mm: float = 0.0,
) -> Dict[str, Union[str, float]]:
    """Assesses meteorological flood hazard potential based on precipitation and antecedent moisture.

    Classification thresholds (based on Far East Amur basin hydrological regime):
    - LOW (< 15 mm / 7d): Dry/normal baseflow conditions.
    - MODERATE (15 - 40 mm / 7d): Significant rainfall, floodplain saturation.
    - HIGH (40 - 80 mm / 7d): Severe storm, dangerous freshet / river rise.
    - EXTREME (> 80 mm / 7d): Catastrophic monsoon / cyclonic inundation.

    Returns:
        Dictionary with hazard level, numerical score [0, 1], and description.
    """
    total_forcing = float(precip_7d_mm + snowmelt_7d_mm)
    combined_signal = 0.6 * total_forcing + 0.4 * float(api_7d_mm)

    if combined_signal < 15.0:
        level = "LOW"
        level_ru = "Низкий (Межень / Слабые осадки)"
        score = float(np.clip(combined_signal / 15.0 * 0.25, 0.0, 0.25))
    elif combined_signal < 40.0:
        level = "MODERATE"
        level_ru = "Умеренный (Локальное увлажнение)"
        score = float(np.clip(0.25 + (combined_signal - 15.0) / 25.0 * 0.25, 0.25, 0.50))
    elif combined_signal < 80.0:
        level = "HIGH"
        level_ru = "Высокий (Паводкообразующие ливни)"
        score = float(np.clip(0.50 + (combined_signal - 40.0) / 40.0 * 0.30, 0.50, 0.80))
    else:
        level = "EXTREME"
        level_ru = "Экстремальный (Катастрофический паводок)"
        score = float(np.clip(0.80 + (combined_signal - 80.0) / 40.0 * 0.20, 0.80, 1.0))

    return {
        "level": level,
        "level_ru": level_ru,
        "score": round(score, 4),
        "total_forcing_mm": round(total_forcing, 2),
    }


def extract_event_weather_features(
    era5_source: Union[str, Path, pd.DataFrame],
    peak_date: Union[str, pd.Timestamp, datetime, date],
    window_3d: int = 3,
    window_7d: int = 7,
    window_14d: int = 14,
    decay_factor: float = 0.85,
) -> Dict[str, Union[float, str]]:
    """Extracts hydrometeorological feature vector for a specific event peak date.

    Args:
        era5_source: Path to ERA5 CSV or loaded DataFrame.
        peak_date: Target peak observation date.
        window_3d: Short-term window length (days).
        window_7d: Medium-term window length (days).
        window_14d: Long-term window length (days).
        decay_factor: API decay coefficient.

    Returns:
        Dictionary of standardized scalar weather features.
    """
    if isinstance(era5_source, (str, Path)):
        df = parse_era5_daily(era5_source)
    elif isinstance(era5_source, pd.DataFrame):
        df = era5_source.copy()
        if "date" in df.columns and not np.issubdtype(df["date"].dtype, np.datetime64):
            df["date"] = pd.to_datetime(df["date"])
    else:
        raise TypeError(f"Unsupported era5_source type: {type(era5_source)}")

    target_dt = pd.to_datetime(peak_date)
    sub_df = df[df["date"] <= target_dt].sort_values("date").reset_index(drop=True)

    if len(sub_df) == 0:
        # Graceful zero fallback if date is outside range
        return {
            "peak_date": target_dt.strftime("%Y-%m-%d"),
            "precip_1d_mm": 0.0,
            "precip_3d_sum_mm": 0.0,
            "precip_7d_sum_mm": 0.0,
            "precip_14d_sum_mm": 0.0,
            "temp_mean_3d_c": 15.0,
            "temp_mean_7d_c": 15.0,
            "snowmelt_3d_sum_mm": 0.0,
            "snowmelt_7d_sum_mm": 0.0,
            "api_7d_mm": 0.0,
            "api_14d_mm": 0.0,
            "hydrological_forcing_7d_mm": 0.0,
            "weather_risk_level": "LOW",
            "weather_risk_score": 0.0,
        }

    # Extract windows
    w1 = sub_df.iloc[-1:]
    w3 = sub_df.iloc[-window_3d:] if len(sub_df) >= window_3d else sub_df
    w7 = sub_df.iloc[-window_7d:] if len(sub_df) >= window_7d else sub_df
    w14 = sub_df.iloc[-window_14d:] if len(sub_df) >= window_14d else sub_df

    p_1d = float(w1["precip_mm"].iloc[-1])
    p_3d = float(w3["precip_mm"].sum())
    p_7d = float(w7["precip_mm"].sum())
    p_14d = float(w14["precip_mm"].sum())

    t_3d = float(w3["temp_c"].mean())
    t_7d = float(w7["temp_c"].mean())

    sm_3d = float(w3["snowmelt_mm"].sum())
    sm_7d = float(w7["snowmelt_mm"].sum())

    api_7d = compute_window_api(w7["precip_mm"].values, decay_factor=decay_factor)
    api_14d = compute_window_api(w14["precip_mm"].values, decay_factor=decay_factor)

    risk = assess_weather_flood_risk(
        precip_7d_mm=p_7d,
        api_7d_mm=api_7d,
        snowmelt_7d_mm=sm_7d,
    )

    return {
        "peak_date": target_dt.strftime("%Y-%m-%d"),
        "precip_1d_mm": round(p_1d, 2),
        "precip_3d_sum_mm": round(p_3d, 2),
        "precip_7d_sum_mm": round(p_7d, 2),
        "precip_14d_sum_mm": round(p_14d, 2),
        "temp_mean_3d_c": round(t_3d, 2),
        "temp_mean_7d_c": round(t_7d, 2),
        "snowmelt_3d_sum_mm": round(sm_3d, 4),
        "snowmelt_7d_sum_mm": round(sm_7d, 4),
        "api_7d_mm": round(api_7d, 2),
        "api_14d_mm": round(api_14d, 2),
        "hydrological_forcing_7d_mm": round(p_7d + sm_7d, 2),
        "weather_risk_level": risk["level"],
        "weather_risk_level_ru": risk["level_ru"],
        "weather_risk_score": risk["score"],
    }


def batch_extract_pair_weather(
    rasters_dir: Union[str, Path],
    pairs_catalogue: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Extracts weather metrics for all pairs in the official catalogue.

    Args:
        rasters_dir: Path to directory containing raster subfolders.
        pairs_catalogue: Dictionary of pairs (like scripts.eda.PAIR_CATALOGUE).

    Returns:
        Dictionary mapping pair_id to extracted weather features.
    """
    base_dir = Path(rasters_dir)
    results = {}

    for pair_id, meta in pairs_catalogue.items():
        event_id = meta.get("event_id")
        aoi = meta.get("aoi")
        peak_date = meta.get("date_peak")

        # Find ERA5 file
        pair_dir = base_dir / event_id / aoi if event_id and aoi else None
        era5_file = None
        if pair_dir and pair_dir.exists():
            matches = list(pair_dir.glob("ERA5_daily_*.csv"))
            if matches:
                era5_file = matches[0]

        if era5_file and peak_date:
            features = extract_event_weather_features(era5_file, peak_date)
            results[pair_id] = features
        else:
            # Fallback
            results[pair_id] = extract_event_weather_features(
                pd.DataFrame({"date": [peak_date], "precip_mm": [0.0], "temp_c": [15.0], "snowmelt_mm": [0.0]}),
                peak_date or "2020-01-01",
            )

    return results
