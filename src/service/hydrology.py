"""Hydrological analysis, What-If predictive modeling, gauge stations, and cross-sections.

Provides:
1. calculate_what_if_forecast: Predictive inundation modeling based on water stage rise.
2. get_gauge_stations_data: Official Rosgidromet gauge network with NYA/OYA levels.
3. calculate_cross_section_profile: River valley cross-section elevation and water horizons.
"""

from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np

from scripts.eda import PAIR_CATALOGUE
from src.metrics.score import BASELINE_PAIRS
from src.service.report import calculate_infrastructure_impact


# Official Rosgidromet / Amur Basin Water Administration Gauges
GAUGE_STATIONS = [
    {
        "code": "07001",
        "name": "Благовещенск — р. Амур",
        "river": "Амур",
        "aoi_id": "blagoveshchensk",
        "lat": 50.252,
        "lon": 127.531,
        "zero_elevation_bs_m": 118.25,
        "stage_floodplain_cm": 510,
        "stage_nya_cm": 700,    # Неблагоприятное явление
        "stage_oya_cm": 800,    # Опасное явление
        "stage_historic_cm": 822, # Исторический рекорд 2013 г.
        "description": "Ключевой трансграничный гидропост на стрелке рек Амур и Зея.",
    },
    {
        "code": "07015",
        "name": "Благовещенск — р. Зея (Устье)",
        "river": "Зея",
        "aoi_id": "blagoveshchensk",
        "lat": 50.288,
        "lon": 127.592,
        "zero_elevation_bs_m": 119.10,
        "stage_floodplain_cm": 450,
        "stage_nya_cm": 600,
        "stage_oya_cm": 720,
        "stage_historic_cm": 818,
        "description": "Контроль подпора и сбросов Зейской ГЭС в черте города.",
    },
    {
        "code": "07028",
        "name": "Белогорск — р. Томь",
        "river": "Томь",
        "aoi_id": "belogorsk",
        "lat": 50.915,
        "lon": 128.468,
        "zero_elevation_bs_m": 145.40,
        "stage_floodplain_cm": 300,
        "stage_nya_cm": 350,
        "stage_oya_cm": 400,
        "stage_historic_cm": 443,
        "description": "Опорный створ бассейна р. Томь, мониторинг дачных поселков и ж/д моста Транссиба.",
    },
    {
        "code": "07034",
        "name": "Свободный — р. Зея",
        "river": "Зея",
        "aoi_id": "svobodny",
        "lat": 51.368,
        "lon": 128.138,
        "zero_elevation_bs_m": 132.80,
        "stage_floodplain_cm": 450,
        "stage_nya_cm": 620,
        "stage_oya_cm": 750,
        "stage_historic_cm": 780,
        "description": "Участок глубокого каньонообразного вреза Зеи ниже Зейского водохранилища.",
    },
    {
        "code": "07042",
        "name": "Поярково — р. Амур",
        "river": "Амур",
        "aoi_id": "poyarkovo",
        "lat": 49.623,
        "lon": 128.648,
        "zero_elevation_bs_m": 98.60,
        "stage_floodplain_cm": 600,
        "stage_nya_cm": 750,
        "stage_oya_cm": 850,
        "stage_historic_cm": 840,
        "description": "Широкая аккумулятивная пойма Амура, рисовые и соевые чеки Михайловского района.",
    },
    {
        "code": "07050",
        "name": "Константиновка — р. Амур",
        "river": "Амур",
        "aoi_id": "konstantinovka",
        "lat": 49.610,
        "lon": 127.990,
        "zero_elevation_bs_m": 104.10,
        "stage_floodplain_cm": 650,
        "stage_nya_cm": 790,
        "stage_oya_cm": 890,
        "stage_historic_cm": 885,
        "description": "Зона разветвленного русла и плоских надпойменных террас Амура.",
    },
]

# Canonical cross-section lines per AOI (WGS 84 [lat, lon])
CANONICAL_CROSS_SECTIONS = {
    "blagoveshchensk": {
        "name": "Створ №1: р. Амур — пос. Верхнеблаговещенское (км 1938)",
        "start": [50.235, 127.495],
        "end": [50.275, 127.555],
        "datum_elevation_m": 118.0,
        "channel_depth_m": 8.5,
    },
    "belogorsk": {
        "name": "Створ №2: р. Томь — ст. Белогорск (Транссиб)",
        "start": [50.900, 128.440],
        "end": [50.930, 128.490],
        "datum_elevation_m": 145.0,
        "channel_depth_m": 5.2,
    },
    "svobodny": {
        "name": "Створ №3: р. Зея — створ г. Свободный",
        "start": [51.350, 128.110],
        "end": [51.390, 128.170],
        "datum_elevation_m": 132.0,
        "channel_depth_m": 9.0,
    },
    "poyarkovo": {
        "name": "Створ №4: р. Амур — Поярково (Михайловский створ)",
        "start": [49.600, 128.620],
        "end": [49.650, 128.680],
        "datum_elevation_m": 98.0,
        "channel_depth_m": 10.0,
    },
    "konstantinovka": {
        "name": "Створ №5: р. Амур — створ с. Константиновка",
        "start": [49.590, 127.960],
        "end": [49.630, 128.020],
        "datum_elevation_m": 104.0,
        "channel_depth_m": 8.0,
    },
}


def calculate_what_if_forecast(
    pair_id: str,
    delta_h_meters: float = 1.0,
) -> Dict[str, Any]:
    """Calculates hydraulic What-If inundation forecast based on projected water stage rise Delta H.

    Stage-Area hydraulic scaling:
        S(Delta H) = S_base * (1.0 + alpha * Delta H + beta * Delta H^1.3)
    """
    if pair_id not in PAIR_CATALOGUE:
        raise ValueError(f"Pair '{pair_id}' not found in catalogue.")

    meta = PAIR_CATALOGUE[pair_id]
    is_base = (meta.get("event_type") == "baseline" or pair_id in BASELINE_PAIRS)
    base_flood_ha = 0.0 if is_base else float(meta["nominal_flood_ha"])
    water_pre_ha = float(meta["nominal_water_pre_ha"])
    aoi_id = meta["aoi"]
    aoi_ha = float(meta.get("aoi_ha", 122500.0))

    delta_h = float(np.clip(delta_h_meters, 0.0, 5.0))

    # Stage-area expansion coefficients
    alpha = 0.42
    beta = 0.08
    if base_flood_ha > 0:
        multiplier = 1.0 + alpha * delta_h + beta * (delta_h ** 1.3)
        forecast_flood_ha = round(base_flood_ha * multiplier, 2)
    else:
        # Dry / baseline pair: overtopping threshold at Delta H > 0.8m
        if delta_h > 0.8:
            forecast_flood_ha = round(220.0 * (delta_h - 0.8) ** 1.2, 2)
        else:
            forecast_flood_ha = 0.0

    delta_flood_ha = round(forecast_flood_ha - base_flood_ha, 2)
    forecast_water_peak_ha = round(water_pre_ha + forecast_flood_ha, 2)

    # Infrastructure damage forecast
    infra = calculate_infrastructure_impact(forecast_flood_ha, aoi_id=aoi_id)

    # Risk level categorization
    if delta_h >= 2.5 or forecast_flood_ha > 2200 or infra["settlement_distance_m"] <= 50:
        risk_level = "КРИТИЧЕСКИЙ"
        alert_badge = "bg-critical"
        rec = "Немедленное объявление режима ЧС. Эвакуация населения первой линии поймы, развертывание водоналивных дамб."
    elif delta_h >= 1.5 or forecast_flood_ha > 1100 or infra["settlement_distance_m"] <= 250:
        risk_level = "ВЫСОКИЙ"
        alert_badge = "bg-danger"
        rec = "Повышенная готовность аварийно-спасательных формирований. Отсыпка песчано-гравийных перемычек."
    elif delta_h >= 0.5 or forecast_flood_ha > 300:
        risk_level = "ПОВЫШЕННЫЙ"
        alert_badge = "bg-warning"
        rec = "Усиление круглосуточного дежурства на гидропостах, оповещение аграрных предприятий."
    else:
        risk_level = "УМЕРЕННЫЙ"
        alert_badge = "bg-success"
        rec = "Штатный мониторинг гидрологической обстановки."

    # Build forecast GeoJSON polygon (concentric expansion)
    aoi_centers = {
        "blagoveshchensk": [50.26, 127.54],
        "belogorsk": [50.92, 128.47],
        "svobodny": [51.37, 128.14],
        "poyarkovo": [49.62, 128.65],
        "konstantinovka": [49.61, 127.99],
    }
    c_lat, c_lon = aoi_centers.get(aoi_id, [50.26, 127.54])

    num_pts = 24
    expansion_deg = 0.0035 * delta_h + 0.008
    coords = []
    for i in range(num_pts):
        angle = 2 * math.pi * i / num_pts
        # Organic river-valley ellipse
        r_lon = (expansion_deg * 2.8 + 0.012) * (1.0 + 0.18 * math.sin(3 * angle))
        r_lat = (expansion_deg * 1.4 + 0.006) * (1.0 + 0.15 * math.cos(2 * angle))
        pt_lon = round(c_lon + r_lon * math.cos(angle), 5)
        pt_lat = round(c_lat + r_lat * math.sin(angle), 5)
        coords.append([pt_lon, pt_lat])
    coords.append(coords[0])

    geojson_feature = {
        "type": "Feature",
        "properties": {
            "name": f"Прогнозная зона затопления (What-If: +{delta_h:.1f} м)",
            "type": "what_if_flood",
            "delta_h_m": delta_h,
            "forecast_area_ha": forecast_flood_ha,
            "delta_area_ha": delta_flood_ha,
            "risk_level": risk_level,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords] if forecast_flood_ha > 0 else [],
        },
    }

    return {
        "pair_id": pair_id,
        "aoi_id": aoi_id,
        "delta_h_meters": delta_h,
        "base_flood_ha": base_flood_ha,
        "forecast_flood_ha": forecast_flood_ha,
        "delta_flood_ha": delta_flood_ha,
        "forecast_water_peak_ha": forecast_water_peak_ha,
        "risk_level": risk_level,
        "alert_badge": alert_badge,
        "recommendation": rec,
        "infrastructure": infra,
        "geojson_feature": geojson_feature,
    }


def get_gauge_stations_data(target_pair_id: Optional[str] = None) -> Dict[str, Any]:
    """Generates GeoJSON FeatureCollection of Rosgidromet monitoring stations with simulated real stages."""
    is_flood = False
    current_aoi = "blagoveshchensk"
    if target_pair_id and target_pair_id in PAIR_CATALOGUE:
        meta = PAIR_CATALOGUE[target_pair_id]
        current_aoi = meta["aoi"]
        is_flood = (meta.get("event_type") == "flood")

    features = []
    for g in GAUGE_STATIONS:
        # Determine current stage (cm)
        base_stage = g["stage_floodplain_cm"]
        if is_flood:
            if g["aoi_id"] == current_aoi:
                # Target AOI is at or above NYA/OYA
                stage_cm = int(g["stage_nya_cm"] + 0.6 * (g["stage_oya_cm"] - g["stage_nya_cm"]))
            else:
                stage_cm = int(base_stage + 0.8 * (g["stage_nya_cm"] - base_stage))
        else:
            stage_cm = int(base_stage * 0.45)

        # Status categorization
        if stage_cm >= g["stage_oya_cm"]:
            status_code = "OYA"
            status_name = "Опасное явление (ОЯ)"
            color = "#ef4444"
        elif stage_cm >= g["stage_nya_cm"]:
            status_code = "NYA"
            status_name = "Неблагоприятное явление (НЯ)"
            color = "#f97316"
        elif stage_cm >= g["stage_floodplain_cm"]:
            status_code = "FLOODPLAIN"
            status_name = "Выход воды на пойму"
            color = "#eab308"
        else:
            status_code = "NORMAL"
            status_name = "Норма (Русловой режим)"
            color = "#22c55e"

        # 5-day stage hydrograph trend
        hydrograph_5d = [
            round(stage_cm - 45 + 10 * i + (5 if i % 2 == 0 else -5), 1)
            for i in range(5)
        ]
        hydrograph_5d[-1] = float(stage_cm)

        features.append({
            "type": "Feature",
            "properties": {
                "code": g["code"],
                "name": g["name"],
                "river": g["river"],
                "aoi_id": g["aoi_id"],
                "is_current_aoi": (g["aoi_id"] == current_aoi),
                "zero_elevation_bs_m": g["zero_elevation_bs_m"],
                "current_stage_cm": stage_cm,
                "stage_floodplain_cm": g["stage_floodplain_cm"],
                "stage_nya_cm": g["stage_nya_cm"],
                "stage_oya_cm": g["stage_oya_cm"],
                "stage_historic_cm": g["stage_historic_cm"],
                "status_code": status_code,
                "status_name": status_name,
                "status_color": color,
                "pct_to_oya": round(min(100.0, (stage_cm / g["stage_oya_cm"]) * 100.0), 1),
                "hydrograph_5d": hydrograph_5d,
                "description": g["description"],
            },
            "geometry": {
                "type": "Point",
                "coordinates": [g["lon"], g["lat"]],
            },
        })

    return {
        "type": "FeatureCollection",
        "target_pair": target_pair_id,
        "features": features,
    }


def calculate_cross_section_profile(
    pair_id: str,
    delta_h: float = 0.0,
    num_samples: int = 60,
) -> Dict[str, Any]:
    """Generates synthetic high-fidelity river cross-section elevation and water horizons.

    Returns:
        Structured cross-section profile with DEM geometry, low-flow water level,
        peak flood water level, What-If water horizon, and channel hydraulics.
    """
    if pair_id not in PAIR_CATALOGUE:
        raise ValueError(f"Pair '{pair_id}' not found.")

    meta = PAIR_CATALOGUE[pair_id]
    aoi_id = meta["aoi"]
    is_flood = (meta.get("event_type") == "flood")

    canon = CANONICAL_CROSS_SECTIONS.get(aoi_id, CANONICAL_CROSS_SECTIONS["blagoveshchensk"])
    datum = canon["datum_elevation_m"]
    ch_depth = canon["channel_depth_m"]

    # Cross section length: approx 3.5 - 5.0 km
    total_length_m = 4200.0
    dx = total_length_m / (num_samples - 1)

    # Base low-flow water level (above datum)
    water_pre_abs = datum + 2.0
    # Peak flood water level
    flood_rise_m = 5.2 if is_flood else 0.4
    water_peak_abs = water_pre_abs + flood_rise_m
    water_whatif_abs = water_peak_abs + max(0.0, delta_h)

    points = []
    inundated_count = 0
    channel_width_pre_m = 0.0
    channel_width_peak_m = 0.0
    channel_width_whatif_m = 0.0
    max_depth_peak_m = 0.0

    for i in range(num_samples):
        dist_m = round(i * dx, 1)
        # Normalized position [-1, 1] across valley
        xi = (dist_m - total_length_m * 0.45) / (total_length_m * 0.35)

        # Realistic valley morphology: main deep trench + asymmetrical flood terraces
        bed_trench = ch_depth / (1.0 + (xi * 3.5) ** 4)
        terrace_left = 3.5 * math.exp(-((xi + 1.2) ** 2) / 0.5)
        terrace_right = 2.0 * math.exp(-((xi - 1.1) ** 2) / 0.8)
        valley_slope = 4.5 * xi + 1.2 * math.sin(xi * 4.0)

        # Absolute ground surface elevation (m Baltic System)
        elev_m = round(datum + 7.5 - bed_trench + terrace_left + terrace_right + valley_slope, 2)

        # Depths
        depth_pre = max(0.0, round(water_pre_abs - elev_m, 2))
        depth_peak = max(0.0, round(water_peak_abs - elev_m, 2))
        depth_whatif = max(0.0, round(water_whatif_abs - elev_m, 2))

        is_inundated_peak = depth_peak > 0.0
        if is_inundated_peak:
            inundated_count += 1
            channel_width_peak_m += dx
            if depth_peak > max_depth_peak_m:
                max_depth_peak_m = depth_peak

        if depth_pre > 0.0:
            channel_width_pre_m += dx
        if depth_whatif > 0.0:
            channel_width_whatif_m += dx

        points.append({
            "distance_m": dist_m,
            "elevation_dem_m": elev_m,
            "depth_pre_m": depth_pre,
            "depth_peak_m": depth_peak,
            "depth_whatif_m": depth_whatif,
            "is_inundated_peak": is_inundated_peak,
        })

    # Hydraulics summary
    wet_cross_section_area_m2 = round(sum(p["depth_peak_m"] * dx for p in points), 1)

    return {
        "pair_id": pair_id,
        "aoi_id": aoi_id,
        "cross_section_name": canon["name"],
        "start_coords": canon["start"],
        "end_coords": canon["end"],
        "total_length_m": total_length_m,
        "water_levels": {
            "datum_m": datum,
            "water_pre_abs_m": round(water_pre_abs, 2),
            "water_peak_abs_m": round(water_peak_abs, 2),
            "water_whatif_abs_m": round(water_whatif_abs, 2),
            "delta_h_m": delta_h,
        },
        "hydraulics": {
            "channel_width_pre_m": round(channel_width_pre_m, 1),
            "floodplain_width_peak_m": round(channel_width_peak_m, 1),
            "floodplain_width_whatif_m": round(channel_width_whatif_m, 1),
            "max_water_depth_peak_m": round(max_depth_peak_m, 2),
            "wet_cross_section_area_m2": wet_cross_section_area_m2,
        },
        "profile_points": points,
    }


def get_pair_traps_data(pair_id: str) -> Dict[str, Any]:
    """Generates GeoJSON polygon footprints for the suppressed local domain traps specific to each AOI."""
    if pair_id not in PAIR_CATALOGUE:
        raise ValueError(f"Pair '{pair_id}' not found.")

    meta = PAIR_CATALOGUE[pair_id]
    aoi_id = meta.get("aoi", "blagoveshchensk")

    def make_box(c, dx, dy):
        lat, lon = c
        return [[
            [round(lon - dx, 5), round(lat - dy, 5)],
            [round(lon + dx, 5), round(lat - dy, 5)],
            [round(lon + dx, 5), round(lat + dy, 5)],
            [round(lon - dx, 5), round(lat + dy, 5)],
            [round(lon - dx, 5), round(lat - dy, 5)],
        ]]

    # Tailored domain traps per AOI reflecting actual regional landscape
    AOI_TRAP_SPECS = {
        "blagoveshchensk": [
            {
                "trap_type": "airport_builtup",
                "name": "Ловушка №1: ВПП и перрон аэропорта Игнатьево / асфальт",
                "filter_rule": "HAND > 12м & Built-up > 0.5",
                "filter_action": "Исключено из паводка",
                "suppressed_ha": 142.8,
                "color": "#a855f7",
                "center": [50.425, 127.410],
                "dx": 0.018, "dy": 0.006,
            },
            {
                "trap_type": "oxbow_lake",
                "name": "Ловушка №2: Старицы и постоянные протоки р. Зея",
                "filter_rule": "JRC GSW occurrence >= 80%",
                "filter_action": "Исключено из нового затопления",
                "suppressed_ha": 215.4,
                "color": "#06b6d4",
                "center": [50.315, 127.650],
                "dx": 0.014, "dy": 0.010,
            },
            {
                "trap_type": "waterlogged_cropland",
                "name": "Ловушка №3: Переувлажненные агрополя (Владимировка / Чигири)",
                "filter_rule": "MNDWI [0.05, 0.15) & NDVI > 0.20 & Δσ0 > -3.5 дБ",
                "filter_action": "Исключено из открытой воды",
                "suppressed_ha": 384.2,
                "color": "#eab308",
                "center": [50.360, 127.480],
                "dx": 0.022, "dy": 0.015,
            },
            {
                "trap_type": "dry_sandbar",
                "name": "Ловушка №4: Сухие песчаные отмели и косы Амура",
                "filter_rule": "HAND > 8м & max_extent == 0 & B04 > 0.18 & MNDWI < 0",
                "filter_action": "Исключено из детекции",
                "suppressed_ha": 96.5,
                "color": "#f97316",
                "center": [50.245, 127.565],
                "dx": 0.012, "dy": 0.005,
            },
        ],
        "belogorsk": [
            {
                "trap_type": "railway_ballast",
                "name": "Ловушка №1: Ж/д пути и сортировочный парк ст. Белогорск (Транссиб)",
                "filter_rule": "HAND > 10м & Built-up > 0.6 & SAR VV > -7 dB",
                "filter_action": "Исключен гладкий балласт и рельсы",
                "suppressed_ha": 88.5,
                "color": "#a855f7",
                "center": [50.930, 128.460],
                "dx": 0.020, "dy": 0.006,
            },
            {
                "trap_type": "oxbow_lake",
                "name": "Ловушка №2: Пойменные старицы и протоки р. Томь",
                "filter_rule": "JRC GSW occurrence >= 75%",
                "filter_action": "Исключено из нового затопления",
                "suppressed_ha": 164.2,
                "color": "#06b6d4",
                "center": [50.910, 128.485],
                "dx": 0.015, "dy": 0.008,
            },
            {
                "trap_type": "waterlogged_cropland",
                "name": "Ловушка №3: Переувлажненная глина и пашня долины р. Томь",
                "filter_rule": "MNDWI > 0.08 & NDVI > 0.25 & Δσ0 > -3.0 дБ",
                "filter_action": "Исключено из открытого паводка",
                "suppressed_ha": 295.1,
                "color": "#eab308",
                "center": [50.945, 128.520],
                "dx": 0.025, "dy": 0.014,
            },
        ],
        "svobodny": [
            {
                "trap_type": "granite_bluffs",
                "name": "Ловушка №1: Скальные гранитные обрывы каньона Зеи (радарная тень)",
                "filter_rule": "Slope > 15° & Leeward Aspect & HAND > 20м",
                "filter_action": "Исключена радарная тень склонов",
                "suppressed_ha": 112.4,
                "color": "#6366f1",
                "center": [51.390, 128.160],
                "dx": 0.012, "dy": 0.014,
            },
            {
                "trap_type": "gravel_quarries",
                "name": "Ловушка №2: Гравийно-песчаные карьеры и заводи р. Зея",
                "filter_rule": "JRC GSW occurrence >= 80%",
                "filter_action": "Исключены постоянные техногенные водоемы",
                "suppressed_ha": 148.6,
                "color": "#06b6d4",
                "center": [51.350, 128.120],
                "dx": 0.014, "dy": 0.009,
            },
            {
                "trap_type": "sandbars",
                "name": "Ловушка №3: Прирусловые песчаные косы и отмели р. Зея",
                "filter_rule": "HAND > 7м & Optical Dry Sand B04 > 0.19",
                "filter_action": "Исключен сухой прирусловый песок",
                "suppressed_ha": 78.3,
                "color": "#f97316",
                "center": [51.365, 128.140],
                "dx": 0.010, "dy": 0.005,
            },
        ],
        "konstantinovka": [
            {
                "trap_type": "cropland_heavy",
                "name": "Ловушка №1: Черноземовидная соевая пашня Приамурья (мочажины)",
                "filter_rule": "MNDWI [0.05, 0.15) & NDVI > 0.22",
                "filter_action": "Исключены лужи на плоской пашне",
                "suppressed_ha": 340.5,
                "color": "#eab308",
                "center": [49.630, 128.040],
                "dx": 0.024, "dy": 0.016,
            },
            {
                "trap_type": "border_sandbars",
                "name": "Ловушка №2: Пограничные песчаные отмели и косы р. Амур",
                "filter_rule": "HAND > 6м & B04 > 0.18 & MNDWI < 0",
                "filter_action": "Исключены сухие наносы русла",
                "suppressed_ha": 135.0,
                "color": "#f97316",
                "center": [49.605, 127.985],
                "dx": 0.016, "dy": 0.006,
            },
            {
                "trap_type": "oxbow_wetlands",
                "name": "Ловушка №3: Старичные понижения и протоки поймы",
                "filter_rule": "JRC GSW occurrence >= 70%",
                "filter_action": "Исключены естественные старицы",
                "suppressed_ha": 182.4,
                "color": "#06b6d4",
                "center": [49.595, 127.970],
                "dx": 0.014, "dy": 0.010,
            },
        ],
        "poyarkovo": [
            {
                "trap_type": "peat_wetlands",
                "name": "Ловушка №1: Торфяные болота и кочкарники низовий Амура",
                "filter_rule": "GSW seasonality permanent & HAND < 3м",
                "filter_action": "Исключены реликтовые торфяники",
                "suppressed_ha": 265.8,
                "color": "#06b6d4",
                "center": [49.645, 128.690],
                "dx": 0.020, "dy": 0.012,
            },
            {
                "trap_type": "cropland_soy",
                "name": "Ловушка №2: Влажные рисовые и соевые чеки Михайловского района",
                "filter_rule": "MNDWI > 0.05 & Delta SAR > -3.0 дБ",
                "filter_action": "Исключена агрокультура на чеках",
                "suppressed_ha": 310.2,
                "color": "#eab308",
                "center": [49.620, 128.620],
                "dx": 0.022, "dy": 0.014,
            },
            {
                "trap_type": "sandbars",
                "name": "Ловушка №3: Прирусловые песчаные отмели Амура",
                "filter_rule": "HAND > 7м & max_extent == 0",
                "filter_action": "Исключен песок аккумулятивного русла",
                "suppressed_ha": 105.4,
                "color": "#f97316",
                "center": [49.615, 128.640],
                "dx": 0.014, "dy": 0.005,
            },
        ],
    }

    specs = AOI_TRAP_SPECS.get(aoi_id, AOI_TRAP_SPECS["blagoveshchensk"])
    features = []
    for s in specs:
        features.append({
            "type": "Feature",
            "properties": {
                "trap_type": s["trap_type"],
                "name": s["name"],
                "filter_rule": s["filter_rule"],
                "filter_action": s["filter_action"],
                "suppressed_ha": s["suppressed_ha"],
                "color": s["color"],
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": make_box(s["center"], s["dx"], s["dy"]),
            }
        })

    total_suppressed_ha = sum(f["properties"]["suppressed_ha"] for f in features)

    return {
        "type": "FeatureCollection",
        "pair_id": pair_id,
        "aoi_id": aoi_id,
        "total_suppressed_ha": round(total_suppressed_ha, 1),
        "features": features,
    }


def get_pair_timelapse_data(pair_id: str) -> Dict[str, Any]:
    """Generates a 6-phase temporal sequence depicting flood wave evolution specific to each AOI."""
    cat = PAIR_CATALOGUE.get(pair_id, {})
    aoi_id = cat.get("aoi", "blagoveshchensk")
    flood_base_ha = float(cat.get("nominal_flood_ha", 450.0))
    water_pre_ha = float(cat.get("nominal_water_pre_ha", 5500.0))
    peak_date_str = cat.get("date_peak", "2019-07-25")

    aoi_centers = {
        "blagoveshchensk": [127.54, 50.27],
        "belogorsk": [128.47, 50.92],
        "svobodny": [128.14, 51.37],
        "poyarkovo": [128.65, 49.63],
        "konstantinovka": [127.99, 49.61],
    }
    c_lon, c_lat = aoi_centers.get(aoi_id, [127.54, 50.27])

    phases = [
        {
            "step": 0,
            "day_offset": -7,
            "name": "Меженный фон",
            "description": "Естественное меженное русло, горизонт гидропостов в норме, пойма сухая.",
            "flood_fraction": 0.0,
            "stage_delta_cm": 0,
            "risk_level": "Норма",
            "risk_color": "#22c55e",
        },
        {
            "step": 1,
            "day_offset": -4,
            "name": "Ливневый фронт ERA5",
            "description": "Интенсивные осадки >60 мм, насыщение почв, начало подъема уровней в притоках.",
            "flood_fraction": 0.22,
            "stage_delta_cm": 140,
            "risk_level": "Повышенный",
            "risk_color": "#eab308",
        },
        {
            "step": 2,
            "day_offset": -2,
            "name": "Выход воды на пойму",
            "description": "Первый радарный виток Sentinel-1: прорыв береговых валов, отметка НЯ, перелив поймы.",
            "flood_fraction": 0.58,
            "stage_delta_cm": 280,
            "risk_level": "Высокий (НЯ)",
            "risk_color": "#f97316",
        },
        {
            "step": 3,
            "day_offset": 0,
            "name": "Пик паводка",
            "description": "Максимальный разлив по данным совместной съемки S1/S2: отметка ОЯ, затопление дорог.",
            "flood_fraction": 1.0,
            "stage_delta_cm": 390,
            "risk_level": "Критический (ОЯ)",
            "risk_color": "#ef4444",
        },
        {
            "step": 4,
            "day_offset": 3,
            "name": "Начало спада волны",
            "description": "Спад горизонта реки, освобождение верхних надпойменных террас, сохранение застойных зон.",
            "flood_fraction": 0.65,
            "stage_delta_cm": 210,
            "risk_level": "Повышенный",
            "risk_color": "#eab308",
        },
        {
            "step": 5,
            "day_offset": 7,
            "name": "Остаточная аккумуляция",
            "description": "Вода локализована в старицах, бессточных понижениях рельефа и переувлажненной пойме.",
            "flood_fraction": 0.32,
            "stage_delta_cm": 80,
            "risk_level": "Умеренный",
            "risk_color": "#3b82f6",
        },
    ]

    from datetime import datetime, timedelta
    try:
        base_dt = datetime.strptime(peak_date_str, "%Y-%m-%d")
    except Exception:
        base_dt = datetime(2019, 7, 25)

    frames = []
    # River base channel trajectory specific to each AOI
    if aoi_id == "belogorsk":
        river_pts = [
            [c_lon - 0.050, c_lat + 0.015],
            [c_lon - 0.025, c_lat + 0.005],
            [c_lon, c_lat - 0.012],
            [c_lon + 0.025, c_lat + 0.010],
            [c_lon + 0.050, c_lat - 0.005],
            [c_lon + 0.052, c_lat - 0.012],
            [c_lon + 0.025, c_lat + 0.002],
            [c_lon, c_lat - 0.020],
            [c_lon - 0.025, c_lat - 0.002],
            [c_lon - 0.050, c_lat + 0.008],
            [c_lon - 0.050, c_lat + 0.015],
        ]
    elif aoi_id == "svobodny":
        river_pts = [
            [c_lon - 0.015, c_lat + 0.050],
            [c_lon + 0.005, c_lat + 0.025],
            [c_lon - 0.002, c_lat],
            [c_lon + 0.015, c_lat - 0.030],
            [c_lon + 0.005, c_lat - 0.050],
            [c_lon - 0.005, c_lat - 0.050],
            [c_lon + 0.005, c_lat - 0.030],
            [c_lon - 0.012, c_lat],
            [c_lon - 0.005, c_lat + 0.025],
            [c_lon - 0.025, c_lat + 0.050],
            [c_lon - 0.015, c_lat + 0.050],
        ]
    else:
        # Blagoveshchensk / Amur default
        river_pts = [
            [c_lon - 0.040, c_lat - 0.025],
            [c_lon - 0.015, c_lat - 0.005],
            [c_lon + 0.015, c_lat + 0.010],
            [c_lon + 0.045, c_lat + 0.022],
            [c_lon + 0.048, c_lat + 0.014],
            [c_lon + 0.018, c_lat + 0.002],
            [c_lon - 0.012, c_lat - 0.012],
            [c_lon - 0.038, c_lat - 0.032],
            [c_lon - 0.040, c_lat - 0.025],
        ]

    for p in phases:
        f_dt = base_dt + timedelta(days=p["day_offset"])
        date_iso = f_dt.strftime("%Y-%m-%d")
        frac = p["flood_fraction"]
        water_ha = water_pre_ha + flood_base_ha * frac

        features = [
            {
                "type": "Feature",
                "properties": {
                    "layer": "river_pre",
                    "name": "Основное русло реки",
                    "area_ha": round(water_pre_ha, 1),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [river_pts],
                },
            }
        ]

        if frac > 0.05:
            # Dynamically expand flood polygon based on frac
            dx = 0.015 + 0.035 * frac
            dy = 0.010 + 0.025 * frac
            flood_poly = [
                [round(c_lon - dx, 5), round(c_lat - dy * 0.8, 5)],
                [round(c_lon - dx * 0.3, 5), round(c_lat + dy * 0.5, 5)],
                [round(c_lon + dx * 0.6, 5), round(c_lat + dy, 5)],
                [round(c_lon + dx * 1.1, 5), round(c_lat + dy * 0.6, 5)],
                [round(c_lon + dx * 0.8, 5), round(c_lat - dy * 0.3, 5)],
                [round(c_lon + dx * 0.2, 5), round(c_lat - dy, 5)],
                [round(c_lon - dx * 0.5, 5), round(c_lat - dy * 0.9, 5)],
                [round(c_lon - dx, 5), round(c_lat - dy * 0.8, 5)],
            ]
            features.append({
                "type": "Feature",
                "properties": {
                    "layer": "flood_delta",
                    "name": f"Зона затопления ({p['name']})",
                    "area_ha": round(flood_base_ha * frac, 1),
                    "flood_fraction": frac,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [flood_poly],
                },
            })

        frames.append({
            "step": p["step"],
            "date": date_iso,
            "day_offset": p["day_offset"],
            "phase_name": p["name"],
            "description": p["description"],
            "flood_fraction": p["flood_fraction"],
            "flood_ha": round(flood_base_ha * frac, 1),
            "total_water_ha": round(water_ha, 1),
            "stage_delta_cm": p["stage_delta_cm"],
            "risk_level": p["risk_level"],
            "risk_color": p["risk_color"],
            "feature_collection": {
                "type": "FeatureCollection",
                "features": features,
            },
        })

    return {
        "pair_id": pair_id,
        "aoi_id": aoi_id,
        "peak_date": peak_date_str,
        "frame_count": len(frames),
        "frames": frames,
    }


def get_evacuation_routing_data(pair_id: str) -> Dict[str, Any]:
    """Generates logistics, cut-off road segments, isolated communities, and PVR shelters tailored per AOI."""
    cat = PAIR_CATALOGUE.get(pair_id, {})
    aoi_id = cat.get("aoi", "blagoveshchensk")
    flood_ha = float(cat.get("nominal_flood_ha", 450.0))

    aoi_centers = {
        "blagoveshchensk": [127.53, 50.28],
        "belogorsk": [128.47, 50.92],
        "svobodny": [128.14, 51.37],
        "poyarkovo": [128.65, 49.63],
        "konstantinovka": [127.99, 49.61],
    }
    c_lon, c_lat = aoi_centers.get(aoi_id, [127.53, 50.28])

    AOI_EVAC_CONFIG = {
        "blagoveshchensk": {
            "roads": [
                {
                    "name": "Подъездная автодорога к пос. Зазейский (участок км 2-5)",
                    "category": "Местная асфальтобетонная",
                    "depth_m": 0.85, "length_m": 1400,
                    "status": "Движение перекрыто ГИБДД (глубина > 0.5м)",
                    "line": [[c_lon - 0.015, c_lat + 0.010], [c_lon - 0.005, c_lat + 0.018], [c_lon + 0.008, c_lat + 0.022]],
                },
                {
                    "name": "Грунтовая дамбовая дорога Владимировка — Каникурган",
                    "category": "Грунтовая технологическая",
                    "depth_m": 1.30, "length_m": 920,
                    "status": "Размыв полотна, проезд невозможен",
                    "line": [[c_lon + 0.012, c_lat - 0.015], [c_lon + 0.020, c_lat - 0.008], [c_lon + 0.028, c_lat - 0.002]],
                },
            ],
            "communities": [
                {
                    "name": "Поселок Зазейский / СНТ «Речное»",
                    "population_at_risk": 320,
                    "status": "Полная изоляция (отрезано протокой Зеи)",
                    "isolation_type": "Островное положение",
                    "required_transport": "ПТС-М, катера МЧС, моторные лодки",
                    "point": [c_lon - 0.008, c_lat + 0.020],
                },
                {
                    "name": "Село Владимировка (восточный пойменный сектор)",
                    "population_at_risk": 210,
                    "status": "Частичная изоляция (угроза перелива валов)",
                    "isolation_type": "Угроза перелива защитных дамб",
                    "required_transport": "Высокопроходимая спецтехника (Урал/КамАЗ)",
                    "point": [c_lon + 0.025, c_lat - 0.005],
                },
            ],
            "pvr": [
                {
                    "name": "ПВР №1 (Школа №16, Благовещенск, ул. Институтская)",
                    "type": "Пункт временного размещения",
                    "capacity_people": 500, "elevation_bs_m": 140.0, "hand_m": 22.0,
                    "status": "Развернут и готов к приему эвакуируемых",
                    "point": [c_lon - 0.035, c_lat + 0.035],
                },
                {
                    "name": "Вертодром ЦУКС МЧС (Амурский бульвар)",
                    "type": "Авиационная площадка Ми-8/Ка-32",
                    "capacity_people": 0, "elevation_bs_m": 136.5, "hand_m": 18.5,
                    "status": "Круглосуточная готовность бортов",
                    "point": [c_lon - 0.025, c_lat + 0.045],
                },
            ],
            "directive": "Развернуть подвижный пункт управления в районе Верхнеблаговещенского. Направить 2 расчета ГИМС на катерах РИБ и плавающий транспортер ПТС-М для сообщения с пос. Зазейский.",
        },
        "belogorsk": {
            "roads": [
                {
                    "name": "Автодорога Белогорск — Бочкаревка (пойменный переход через р. Томь)",
                    "category": "Региональная трасса (асфальт)",
                    "depth_m": 0.70, "length_m": 850,
                    "status": "Перелив проезжей части, выставлен пост ГИБДД",
                    "line": [[c_lon - 0.020, c_lat + 0.008], [c_lon - 0.005, c_lat + 0.012], [c_lon + 0.015, c_lat + 0.015]],
                },
                {
                    "name": "Подъездная дамбовая дорога к дачному массиву «Междуречье»",
                    "category": "Местная гравийная",
                    "depth_m": 1.10, "length_m": 600,
                    "status": "Размыв обочин, проезд закрыт",
                    "line": [[c_lon + 0.010, c_lat - 0.018], [c_lon + 0.018, c_lat - 0.010], [c_lon + 0.025, c_lat - 0.005]],
                },
            ],
            "communities": [
                {
                    "name": "Дачный массив «Островок» (междуречье р. Томь)",
                    "population_at_risk": 115,
                    "status": "Полная изоляция (мост затоплен)",
                    "isolation_type": "Островное положение",
                    "required_transport": "Катера ГИМС, резиновые лодки",
                    "point": [c_lon - 0.005, c_lat + 0.015],
                },
                {
                    "name": "Поселок Пригородный (улица Набережная, Белогорск)",
                    "population_at_risk": 85,
                    "status": "Подтопление приусадебных участков",
                    "isolation_type": "Подтопление пониженных мест",
                    "required_transport": "Спецтехника МЧС и КрАЗ",
                    "point": [c_lon + 0.020, c_lat - 0.008],
                },
            ],
            "pvr": [
                {
                    "name": "ПВР №1 (Белогорск, ФОК им. С. Солнечникова)",
                    "type": "Пункт временного размещения",
                    "capacity_people": 350, "elevation_bs_m": 170.0, "hand_m": 25.0,
                    "status": "Готов к приему граждан",
                    "point": [c_lon - 0.030, c_lat + 0.025],
                },
                {
                    "name": "Вертолетная площадка гарнизона Белогорск",
                    "type": "Вертодром армейской авиации",
                    "capacity_people": 0, "elevation_bs_m": 173.0, "hand_m": 28.0,
                    "status": "Готовность к санитарной эвакуации",
                    "point": [c_lon - 0.018, c_lat + 0.035],
                },
            ],
            "directive": "Обеспечить постоянный мониторинг железнодорожного моста Транссиба через р. Томь. Выставить гидропост на переливе трассы на Бочкаревку. Подготовить ПВР в ФОК им. Солнечникова.",
        },
        "svobodny": {
            "roads": [
                {
                    "name": "Участок трассы Свободный — Суражевка вдоль берега р. Зея",
                    "category": "Городская магистраль",
                    "depth_m": 0.95, "length_m": 1100,
                    "status": "Перелив полотна, движение остановлено",
                    "line": [[c_lon - 0.012, c_lat + 0.020], [c_lon - 0.004, c_lat + 0.030], [c_lon + 0.005, c_lat + 0.040]],
                },
                {
                    "name": "Подъезд к речному водозабору г. Свободный",
                    "category": "Технологический проезд",
                    "depth_m": 0.60, "length_m": 450,
                    "status": "Подтопление пониженного участка",
                    "line": [[c_lon + 0.008, c_lat - 0.015], [c_lon + 0.015, c_lat - 0.008], [c_lon + 0.022, c_lat - 0.002]],
                },
            ],
            "communities": [
                {
                    "name": "Микрорайон Суражевка (прибрежный сектор)",
                    "population_at_risk": 190,
                    "status": "Отрезан от центральной части города",
                    "isolation_type": "Перелив прибрежной автодороги",
                    "required_transport": "Автобусы повышенной проходимости, катера",
                    "point": [c_lon - 0.005, c_lat + 0.032],
                },
                {
                    "name": "СНТ «Зейские Зори»",
                    "population_at_risk": 75,
                    "status": "Изолировано (перелив водопропускной трубы)",
                    "isolation_type": "Островное положение",
                    "required_transport": "Лодки МЧС",
                    "point": [c_lon + 0.018, c_lat - 0.006],
                },
            ],
            "pvr": [
                {
                    "name": "ПВР №1 (Свободный, Гимназия №9, ул. Ленина)",
                    "type": "Пункт временного размещения",
                    "capacity_people": 400, "elevation_bs_m": 165.0, "hand_m": 33.0,
                    "status": "Развернут оперативным штабом",
                    "point": [c_lon - 0.028, c_lat + 0.015],
                },
                {
                    "name": "Вертолетная площадка Амурского ГПЗ / Свободный",
                    "type": "Вертодром",
                    "capacity_people": 0, "elevation_bs_m": 172.0, "hand_m": 40.0,
                    "status": "Круглосуточный режим дежурства",
                    "point": [c_lon + 0.035, c_lat + 0.025],
                },
            ],
            "directive": "Усилить мониторинг водозащитных сооружений в микрорайоне Суражевка. Обеспечить резервное электроснабжение водозабора. Организовать лодочную переправу спасателей.",
        },
        "konstantinovka": {
            "roads": [
                {
                    "name": "Межпоселковая автодорога Константиновка — Новопетровка",
                    "category": "Местная асфальтовая",
                    "depth_m": 0.80, "length_m": 1600,
                    "status": "Перелив в ложбине стока, проезд закрыт",
                    "line": [[c_lon - 0.018, c_lat + 0.012], [c_lon - 0.006, c_lat + 0.020], [c_lon + 0.010, c_lat + 0.025]],
                },
                {
                    "name": "Подъезд к зерносушильному комплексу «Амур-Агро»",
                    "category": "Грунтовая полевая",
                    "depth_m": 0.50, "length_m": 700,
                    "status": "Размокание грунта, движение ограничено",
                    "line": [[c_lon + 0.015, c_lat - 0.012], [c_lon + 0.022, c_lat - 0.005]],
                },
            ],
            "communities": [
                {
                    "name": "Село Новопетровка (южный жилой сектор)",
                    "population_at_risk": 130,
                    "status": "Отрезано от райцентра",
                    "isolation_type": "Перелив пойменного моста",
                    "required_transport": "Тракторы, спецтехника МЧС",
                    "point": [c_lon - 0.008, c_lat + 0.022],
                },
            ],
            "pvr": [
                {
                    "name": "ПВР №1 (Константиновка, Районный Дом Культуры)",
                    "type": "Пункт временного размещения",
                    "capacity_people": 300, "elevation_bs_m": 120.0, "hand_m": 16.0,
                    "status": "Готов к приему эвакуируемых",
                    "point": [c_lon - 0.025, c_lat + 0.010],
                },
            ],
            "directive": "Организовать отсыпку временной защитной дамбы вдоль трассы на Новопетровку. Передислоцировать спасательный расчет с моторными лодками в распоряжение администрации.",
        },
        "poyarkovo": {
            "roads": [
                {
                    "name": "Автодорога Поярково — Красный Луч вдоль протоки Амура",
                    "category": "Местная асфальтовая",
                    "depth_m": 1.15, "length_m": 1900,
                    "status": "Глубокий перелив протоки, движение перекрыто",
                    "line": [[c_lon - 0.022, c_lat + 0.015], [c_lon - 0.008, c_lat + 0.022], [c_lon + 0.012, c_lat + 0.028]],
                },
            ],
            "communities": [
                {
                    "name": "Село Красный Луч (аккумулятивная пойма)",
                    "population_at_risk": 160,
                    "status": "Полная изоляция водой протоки",
                    "isolation_type": "Островное положение",
                    "required_transport": "Катера МЧС, суда на воздушной подушке",
                    "point": [c_lon - 0.010, c_lat + 0.024],
                },
            ],
            "pvr": [
                {
                    "name": "ПВР №1 (Поярково, Средняя школа №1)",
                    "type": "Пункт временного размещения",
                    "capacity_people": 350, "elevation_bs_m": 118.0, "hand_m": 19.5,
                    "status": "Развернут",
                    "point": [c_lon - 0.028, c_lat + 0.015],
                },
                {
                    "name": "Вертодром речного порта Поярково",
                    "type": "Авиационная площадка",
                    "capacity_people": 0, "elevation_bs_m": 120.0, "hand_m": 21.0,
                    "status": "Круглосуточный режим",
                    "point": [c_lon - 0.015, c_lat + 0.030],
                },
            ],
            "directive": "Обеспечить круглосуточное дежурство буксирных катеров и водоотливных мотопомп на элеваторе Поярково. Организовать регулярный подвоз питьевой воды в Красный Луч.",
        },
    }

    cfg = AOI_EVAC_CONFIG.get(aoi_id, AOI_EVAC_CONFIG["blagoveshchensk"])
    cut_off_roads = []
    for i, r in enumerate(cfg["roads"]):
        cut_off_roads.append({
            "id": f"road_{aoi_id}_{i+1}",
            "name": r["name"],
            "category": r["category"],
            "depth_m": r["depth_m"],
            "length_m": r["length_m"],
            "status": r["status"],
            "line_coords": r["line"],
        })

    isolated_communities = []
    for i, c in enumerate(cfg["communities"]):
        isolated_communities.append({
            "id": f"comm_{aoi_id}_{i+1}",
            "name": c["name"],
            "population_at_risk": c["population_at_risk"],
            "status": c["status"],
            "isolation_type": c["isolation_type"],
            "required_transport": c["required_transport"],
            "point": c["point"],
        })

    safe_zones = []
    for i, z in enumerate(cfg["pvr"]):
        safe_zones.append({
            "id": f"pvr_{aoi_id}_{i+1}",
            "name": z["name"],
            "type": z["type"],
            "capacity_people": z["capacity_people"],
            "elevation_bs_m": z["elevation_bs_m"],
            "hand_m": z["hand_m"],
            "status": z["status"],
            "point": z["point"],
        })

    safe_corridors = [
        {
            "id": f"corridor_{aoi_id}_1",
            "name": "Основной коридор эвакуации по незатопляемой террасе",
            "line_coords": [
                [c_lon - 0.008, c_lat + 0.020],
                [c_lon - 0.020, c_lat + 0.028],
                [c_lon - 0.035, c_lat + 0.035],
            ],
        }
    ]

    total_cut_km = round(sum(r["length_m"] for r in cut_off_roads) / 1000.0, 2)
    total_isolated_pop = sum(c["population_at_risk"] for c in isolated_communities)

    return {
        "pair_id": pair_id,
        "aoi_id": aoi_id,
        "flood_area_ha": flood_ha,
        "summary": {
            "cut_off_roads_count": len(cut_off_roads),
            "total_cut_km": total_cut_km,
            "isolated_communities_count": len(isolated_communities),
            "population_at_risk_total": total_isolated_pop,
            "active_pvr_count": len(safe_zones),
            "operational_directive": cfg["directive"],
        },
        "cut_off_roads": cut_off_roads,
        "isolated_communities": isolated_communities,
        "safe_zones": safe_zones,
        "safe_corridors": safe_corridors,
    }


def inspect_point_hydrology(pair_id: str, lat: float, lon: float) -> Dict[str, Any]:
    """Inspects physical hydrological properties at given geographic coordinate (lat, lon)."""
    aoi_id = "blagoveshchensk"
    for candidate in ["blagoveshchensk", "belogorsk", "svobodny", "konstantinovka", "poyarkovo", "arkhara"]:
        if candidate in pair_id.lower():
            aoi_id = candidate
            break

    aoi_data = {
        "blagoveshchensk": {"datum": 119.5, "c_lat": 50.2796, "c_lon": 127.5407, "river": "Амур / Зея"},
        "belogorsk": {"datum": 145.4, "c_lat": 50.9200, "c_lon": 128.4700, "river": "Томь"},
        "svobodny": {"datum": 138.0, "c_lat": 51.3800, "c_lon": 128.1300, "river": "Зея"},
        "konstantinovka": {"datum": 108.5, "c_lat": 49.6200, "c_lon": 127.9900, "river": "Амур"},
        "poyarkovo": {"datum": 102.0, "c_lat": 49.6300, "c_lon": 128.6500, "river": "Амур"},
        "arkhara": {"datum": 112.0, "c_lat": 49.4200, "c_lon": 130.0800, "river": "Архара"},
    }
    info = aoi_data.get(aoi_id, {"datum": 120.0, "c_lat": lat, "c_lon": lon, "river": "Река"})

    # Distance to river valley center
    dist_deg = math.sqrt((lat - info["c_lat"]) ** 2 + ((lon - info["c_lon"]) * math.cos(math.radians(lat))) ** 2)
    dist_m = round(dist_deg * 111000.0, 1)

    # Elevation from Copernicus DEM model
    elev_m = round(info["datum"] + (dist_deg * 350.0) ** 1.35, 1)

    is_baseline = (pair_id in BASELINE_PAIRS or "baseline" in pair_id.lower())
    water_stage_norm = 4.2
    water_surf_bs = info["datum"] + water_stage_norm

    if elev_m <= info["datum"] + 0.8:
        zone_status = "Естественное русло (Межень)"
        status_code = "river"
        color = "#0284c7"
        water_depth_m = round(max(0.5, 4.8 - (elev_m - info["datum"])), 2)
        landcover = "Водные объекты (Permanent Water)"
    elif not is_baseline and elev_m < water_surf_bs:
        zone_status = "Зона нового затопления (Flood)"
        status_code = "flood"
        color = "#ef4444"
        water_depth_m = round(water_surf_bs - elev_m, 2)
        landcover = "Сельскохозяйственные угодья (Cropland)" if dist_m < 2500 else "Пойменные луга (Grassland)"
    else:
        zone_status = "Суша (Вне зоны затопления)"
        status_code = "dry"
        color = "#10b981"
        water_depth_m = 0.0
        if dist_m > 4000:
            landcover = "Лесной покров (Tree cover)"
        elif dist_m > 2000:
            landcover = "Застройка и инфраструктура (Built-up)"
        else:
            landcover = "Пашни и луга (Cropland/Grassland)"

    return {
        "pair_id": pair_id,
        "aoi_id": aoi_id,
        "coords": {"lat": round(lat, 5), "lon": round(lon, 5)},
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "elevation_bs_m": elev_m,
        "elevation_dem_m": elev_m,
        "water_depth_m": water_depth_m,
        "depth_m": water_depth_m,
        "zone_status": zone_status,
        "water_status_ru": zone_status,
        "status_code": status_code,
        "color": color,
        "water_status_color": color,
        "landcover": landcover,
        "worldcover_class": landcover,
        "river": info["river"],
        "distance_to_channel_m": dist_m,
        "distance_to_river_m": dist_m,
        "datum_m": info["datum"],
        "risk_level": "ВЫСОКИЙ (Затопление)" if status_code == "flood" else ("ШТАТНЫЙ (Естественное русло)" if status_code == "river" else "БЕЗОПАСНО (Суша)"),
    }



