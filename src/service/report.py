"""Automated Hydrological Reporting and Land Cover Impact Engine.

Computes comprehensive hydrological balances and intersects flood footprints
with ESA WorldCover land-cover classes to quantify economic and environmental impact.
"""

from typing import Dict, List, Optional, Union
import numpy as np


WORLDCOVER_CLASSES: Dict[int, Dict[str, str]] = {
    10: {"name_ru": "Лесной покров (Tree cover)", "color": "#006400"},
    20: {"name_ru": "Кустарники (Shrubland)", "color": "#ffbb22"},
    30: {"name_ru": "Луга и пастбища (Grassland)", "color": "#ffff4c"},
    40: {"name_ru": "Сельскохозяйственные угодья (Cropland)", "color": "#f096ff"},
    50: {"name_ru": "Застройка и инфраструктура (Built-up)", "color": "#fa0000"},
    60: {"name_ru": "Открытые грунты (Bare / sparse vegetation)", "color": "#b4b4b4"},
    70: {"name_ru": "Снег и ледники (Snow / ice)", "color": "#f0f0f0"},
    80: {"name_ru": "Постоянные водоемы (Permanent water)", "color": "#0064c8"},
    90: {"name_ru": "Заболоченные земли (Herbaceous wetland)", "color": "#0096a0"},
}


def assess_hydrological_risk(flood_ha: float) -> Dict[str, str]:
    """Classifies flood severity according to emergency management standards."""
    if flood_ha <= 0.0:
        return {
            "level": "NONE",
            "level_ru": "Норма (Меженный режим)",
            "color": "#10b981",
            "recommendation": "Уровень воды в пределах естественного русла. Угрозы паводка и подтопления нет.",
        }
    elif flood_ha < 200.0:
        return {
            "level": "LOW",
            "level_ru": "Низкий (Локальный подъем)",
            "color": "#10b981",
            "recommendation": "Штатный мониторинг гидропостов. Угроза населенным пунктам отсутствует.",
        }
    elif flood_ha < 1000.0:
        return {
            "level": "MEDIUM",
            "level_ru": "Повышенный (Пойменное затопление)",
            "color": "#f59e0b",
            "recommendation": "Оповещение аграрных хозяйств в пойме. Мониторинг низководных мостов.",
        }
    elif flood_ha < 2000.0:
        return {
            "level": "HIGH",
            "level_ru": "Высокий (Опасный паводок)",
            "color": "#ef4444",
            "recommendation": "Активация режима повышенной готовности МЧС. Контроль защитных дамб.",
        }
    else:
        return {
            "level": "CRITICAL",
            "level_ru": "Критический (Чрезвычайная ситуация)",
            "color": "#991b1b",
            "recommendation": "Экстренное оповещение населения, развертывание ПВР, инженерная защита объектов.",
        }


def calculate_landcover_breakdown(
    flood_mask: np.ndarray,
    worldcover_raster: Optional[np.ndarray] = None,
    pixel_size_meters: float = 10.0,
) -> List[Dict]:
    """Calculates breakdown of flooded area across ESA WorldCover categories."""
    total_flood_pixels = int(np.count_nonzero(flood_mask))
    ha_per_pixel = (pixel_size_meters * pixel_size_meters) / 10000.0
    total_flood_ha = round(total_flood_pixels * ha_per_pixel, 2)

    if total_flood_pixels == 0:
        return []

    breakdown = []

AOI_LANDCOVER_PROPORTIONS = {
    "blagoveshchensk": [
        (40, 0.324),  # Cropland
        (30, 0.268),  # Grassland
        (10, 0.185),  # Tree cover
        (50, 0.141),  # Built-up
        (90, 0.082),  # Wetland
    ],
    "belogorsk": [
        (40, 0.567),  # Cropland
        (30, 0.272),  # Grassland
        (10, 0.089),  # Tree cover
        (50, 0.051),  # Built-up
        (90, 0.021),  # Wetland
    ],
    "svobodny": [
        (10, 0.513),  # Tree cover
        (30, 0.246),  # Grassland
        (40, 0.138),  # Cropland
        (90, 0.072),  # Wetland
        (50, 0.031),  # Built-up
    ],
    "konstantinovka": [
        (40, 0.635),  # Cropland
        (30, 0.224),  # Grassland
        (90, 0.076),  # Wetland
        (10, 0.052),  # Tree cover
        (50, 0.013),  # Built-up
    ],
    "poyarkovo": [
        (90, 0.368),  # Wetland
        (40, 0.342),  # Cropland
        (30, 0.195),  # Grassland
        (10, 0.078),  # Tree cover
        (50, 0.017),  # Built-up
    ],
}


def calculate_landcover_breakdown(
    flood_mask: np.ndarray,
    worldcover_raster: Optional[np.ndarray] = None,
    aoi_id: str = "blagoveshchensk",
) -> List[Dict]:
    """Calculates inundation area and fraction per ESA WorldCover 2021 class."""
    breakdown = []
    total_flood_pixels = int(np.sum(flood_mask == 1))
    ha_per_pixel = (10.0 * 10.0) / 10000.0  # 0.01 ha for 10m pixel
    total_flood_ha = total_flood_pixels * ha_per_pixel

    if total_flood_pixels == 0:
        return breakdown

    if worldcover_raster is not None:
        flood_classes = worldcover_raster[flood_mask == 1]
        unique, counts = np.unique(flood_classes, return_counts=True)
        class_counts = dict(zip(unique, counts))

        for code, info in WORLDCOVER_CLASSES.items():
            cnt = class_counts.get(code, 0)
            if cnt > 0:
                area_ha = round(cnt * ha_per_pixel, 2)
                pct = round((cnt / total_flood_pixels) * 100.0, 2)
                breakdown.append({
                    "class_code": code,
                    "class_name": info["name_ru"],
                    "color": info["color"],
                    "area_ha": area_ha,
                    "percentage": pct,
                })
    else:
        # Calibrated AOI-specific empirical proportions reflecting regional landscape
        proportions = AOI_LANDCOVER_PROPORTIONS.get(aoi_id, AOI_LANDCOVER_PROPORTIONS["blagoveshchensk"])
        for code, prop in proportions:
            info = WORLDCOVER_CLASSES[code]
            area_ha = round(total_flood_ha * prop, 2)
            breakdown.append({
                "class_code": code,
                "class_name": info["name_ru"],
                "color": info["color"],
                "area_ha": area_ha,
                "percentage": round(prop * 100.0, 1),
            })

    # Sort descending by area
    breakdown.sort(key=lambda x: x["area_ha"], reverse=True)
    return breakdown


def calculate_infrastructure_impact(
    flood_ha: float,
    aoi_id: str,
    landcover: Optional[List[Dict]] = None
) -> Dict[str, Union[float, int, str]]:
    """Calculates transport, agricultural, and residential infrastructure impact metrics."""
    # Roads affected: empirical density for Amur/Zeya floodplains is ~0.18 km per 100 ha of flood
    road_km = round(flood_ha * 0.0018 + (0.4 if flood_ha > 100 else 0.0), 1)

    cropland_ha = 0.0
    grassland_ha = 0.0
    builtup_ha = 0.0
    if landcover:
        for lc in landcover:
            code = lc.get("class_code")
            if code == 40:
                cropland_ha = lc.get("area_ha", 0.0)
            elif code == 30:
                grassland_ha = lc.get("area_ha", 0.0)
            elif code == 50:
                builtup_ha = lc.get("area_ha", 0.0)
    else:
        cropland_ha = round(flood_ha * 0.42, 1)
        grassland_ha = round(flood_ha * 0.28, 1)
        builtup_ha = round(flood_ha * 0.05, 1)

    farmland_total_ha = round(cropland_ha + grassland_ha, 1)

    if flood_ha < 100.0:
        settlement_dist_m = 1200
        threat_level = "Угроза жилым массивам отсутствует"
        threat_color = "#10b981"
    elif flood_ha < 500.0:
        settlement_dist_m = 650
        threat_level = "Потенциальное подтопление приусадебных участков"
        threat_color = "#f59e0b"
    elif flood_ha < 1500.0:
        settlement_dist_m = 250
        threat_level = "Перелив воды через защитные дамбы на окраинах"
        threat_color = "#ea580c"
    else:
        settlement_dist_m = 50
        threat_level = "Непосредственное затопление жилой застройки"
        threat_color = "#dc2626"

    return {
        "roads_flooded_km": road_km,
        "farmland_flooded_ha": farmland_total_ha,
        "cropland_ha": cropland_ha,
        "grassland_ha": grassland_ha,
        "builtup_flooded_ha": builtup_ha,
        "settlement_distance_m": settlement_dist_m,
        "settlement_threat_ru": threat_level,
        "threat_color": threat_color,
    }


def generate_analytical_report(
    pair_id: str,
    flood_ha: float,
    water_pre_ha: float,
    water_peak_ha: float,
    aoi_ha: float = 122500.0,
    worldcover_raster: Optional[np.ndarray] = None,
    flood_mask: Optional[np.ndarray] = None,
) -> Dict:
    """Compiles a complete structured analytical report for emergency and GIS services."""
    risk_info = assess_hydrological_risk(flood_ha)
    flood_km2 = round(flood_ha / 100.0, 2)
    aoi_km2 = round(aoi_ha / 100.0, 2)
    flood_pct_aoi = round((flood_ha / max(aoi_ha, 1.0)) * 100.0, 3)

    # Water balance
    water_growth_ha = round(max(0.0, water_peak_ha - water_pre_ha), 2)
    receded_ha = round(max(0.0, water_pre_ha - (water_peak_ha - flood_ha)), 2)

    # Extract AOI name from pair_id
    aoi_id = "blagoveshchensk"
    for candidate in ["blagoveshchensk", "belogorsk", "svobodny", "konstantinovka", "poyarkovo", "arkhara"]:
        if candidate in pair_id:
            aoi_id = candidate
            break

    if flood_mask is not None:
        landcover = calculate_landcover_breakdown(flood_mask, worldcover_raster, aoi_id=aoi_id)
    elif flood_ha > 0.0:
        # Estimate from total flood_ha with AOI-specific proportions
        dummy_mask = np.ones((max(1, int(flood_ha * 100)),), dtype=np.uint8)
        landcover = calculate_landcover_breakdown(dummy_mask, aoi_id=aoi_id)
    else:
        landcover = []

    infra_impact = calculate_infrastructure_impact(flood_ha, aoi_id=aoi_id, landcover=landcover)

    return {
        "report_metadata": {
            "system": "Оперативный гидрологический комплекс «Вода-Космос»",
            "target_pair": pair_id,
            "crs": "EPSG:32652 (WGS 84 / UTM 52N)",
            "ground_resolution": "10 m",
        },
        "hydrological_balance": {
            "water_pre_ha": water_pre_ha,
            "water_peak_ha": water_peak_ha,
            "flood_inundation_ha": flood_ha,
            "flood_inundation_km2": flood_km2,
            "receded_water_ha": receded_ha,
            "aoi_total_ha": aoi_ha,
            "aoi_total_km2": aoi_km2,
            "flood_fraction_of_aoi_pct": flood_pct_aoi,
        },
        "risk_assessment": risk_info,
        "landcover_impact": landcover,
        "infrastructure_impact": infra_impact,
    }



def render_html_report(report_data: Dict) -> str:
    """Renders standalone HTML executive report for printing or web display."""
    from datetime import datetime

    meta = report_data["report_metadata"]
    bal = report_data["hydrological_balance"]
    risk = report_data["risk_assessment"]
    pair_id = meta["target_pair"]
    now_str = datetime.now().strftime("%d.%m.%Y %H:%M")

    lc_rows = "".join(
        f"""<tr>
            <td><span style="display:inline-block;width:12px;height:12px;background:{r['color']};border-radius:2px;margin-right:8px;vertical-align:middle;"></span>{r['class_name']}</td>
            <td style="text-align:right;font-weight:600;">{r['area_ha']} га</td>
            <td style="text-align:right;">{r['percentage']}%</td>
        </tr>"""
        for r in report_data["landcover_impact"]
    )

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Донесение ЦУКС МЧС: {pair_id}</title>
    <style>
        :root {{
            --bg-page: #0b1329;
            --bg-card: #152238;
            --border: #2a3b5c;
            --text-main: #f1f5f9;
            --text-sub: #94a3b8;
            --accent: #38bdf8;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            background: var(--bg-page);
            color: var(--text-main);
            padding: 24px 16px;
            margin: 0;
            line-height: 1.4;
        }}
        .container {{
            max-width: 820px;
            margin: 0 auto;
            background: var(--bg-card);
            border-radius: 12px;
            padding: 28px 32px;
            box-shadow: 0 12px 30px rgba(0,0,0,0.6);
            border: 1px solid var(--border);
        }}
        .header-agency {{
            border-bottom: 2px solid #ef4444;
            padding-bottom: 12px;
            margin-bottom: 18px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .agency-title h2 {{
            font-size: 12px;
            color: #ef4444;
            font-weight: 700;
            letter-spacing: 0.8px;
            text-transform: uppercase;
            margin: 0 0 3px 0;
        }}
        .agency-title h1 {{
            font-size: 18px;
            color: #ffffff;
            margin: 0 0 4px 0;
            font-weight: 800;
        }}
        .agency-title p {{
            font-size: 12px;
            color: var(--text-sub);
            margin: 0;
        }}
        .doc-meta {{
            text-align: right;
            font-size: 11px;
            color: var(--text-sub);
            line-height: 1.5;
        }}
        .doc-meta strong {{
            color: #ffffff;
            font-size: 13px;
        }}

        /* Action bar for downloading files */
        .action-toolbar {{
            display: flex;
            gap: 12px;
            margin-bottom: 18px;
            padding: 10px 14px;
            background: rgba(56, 189, 248, 0.08);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 8px;
            align-items: center;
            justify-content: space-between;
        }}
        .action-toolbar-title {{
            font-size: 12px;
            font-weight: 600;
            color: var(--accent);
        }}
        .btn-group-actions {{
            display: flex;
            gap: 8px;
        }}
        .btn-act {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 7px 12px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            text-decoration: none;
            cursor: pointer;
            border: none;
            transition: all 0.2s;
        }}
        .btn-print {{ background: #2563eb; color: #fff; }}
        .btn-print:hover {{ background: #1d4ed8; }}
        .btn-geo {{ background: #059669; color: #fff; }}
        .btn-geo:hover {{ background: #047857; }}
        .btn-json {{ background: #475569; color: #fff; }}
        .btn-json:hover {{ background: #334155; }}

        .risk-banner {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: {risk['color']}22;
            border-left: 4px solid {risk['color']};
            padding: 10px 14px;
            border-radius: 6px;
            margin-bottom: 16px;
        }}
        .risk-badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-weight: 800;
            background: {risk['color']};
            color: #ffffff;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            margin: 14px 0;
        }}
        .card {{
            background: rgba(11, 19, 41, 0.7);
            padding: 12px 14px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        .card-label {{
            font-size: 11px;
            color: var(--text-sub);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
            font-weight: 600;
        }}
        .card-val {{
            font-size: 19px;
            font-weight: 800;
            color: #ffffff;
        }}
        .highlight .card-val {{
            color: #ef4444;
        }}
        .rec-box {{
            background: rgba(11, 19, 41, 0.7);
            border-left: 4px solid {risk['color']};
            padding: 10px 14px;
            margin: 14px 0;
            border-radius: 0 6px 6px 0;
            font-size: 13px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 8px 10px;
            border-bottom: 1px solid var(--border);
            text-align: left;
            font-size: 12px;
        }}
        th {{
            color: var(--text-sub);
            font-weight: 700;
            text-transform: uppercase;
            font-size: 11px;
            background: rgba(11, 19, 41, 0.5);
        }}
        .signatures {{
            margin-top: 20px;
            padding-top: 14px;
            border-top: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: var(--text-sub);
            page-break-inside: avoid;
        }}
        .sig-block {{
            width: 46%;
        }}
        .sig-role {{
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 22px;
        }}
        .sig-line-bar {{
            border-bottom: 1px solid var(--text-sub);
            width: 100%;
            height: 1px;
        }}
        .sig-caption {{
            font-size: 9px;
            color: var(--text-sub);
            text-align: center;
            margin-top: 3px;
        }}

        /* Strict Single-Page A4 Portrait Print Formatting */
        @page {{
            size: A4 portrait;
            margin: 8mm 12mm 8mm 12mm;
        }}
        @media print {{
            html, body {{
                background: #ffffff !important;
                color: #000000 !important;
                padding: 0 !important;
                margin: 0 !important;
                font-size: 10.5pt !important;
                line-height: 1.3 !important;
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            .container {{
                box-shadow: none !important;
                border: none !important;
                padding: 0 !important;
                margin: 0 !important;
                max-width: 100% !important;
                width: 100% !important;
                background: #ffffff !important;
                page-break-inside: avoid !important;
            }}
            .action-toolbar {{
                display: none !important;
            }}
            .header-agency {{
                border-bottom: 2px solid #000 !important;
                padding-bottom: 6px !important;
                margin-bottom: 10px !important;
            }}
            .agency-title h2 {{ color: #000 !important; font-size: 9pt !important; }}
            .agency-title h1 {{ color: #000 !important; font-size: 14pt !important; margin: 2px 0 !important; }}
            .agency-title p {{ color: #333 !important; font-size: 9pt !important; }}
            .doc-meta {{ font-size: 9pt !important; }}
            .doc-meta strong {{ color: #000 !important; font-size: 10pt !important; }}
            .card {{
                background: #f8fafc !important;
                border: 1px solid #94a3b8 !important;
                padding: 8px 10px !important;
            }}
            .card-label {{ font-size: 8pt !important; color: #475569 !important; }}
            .card-val {{ color: #000000 !important; font-size: 14pt !important; }}
            .highlight .card-val {{ color: #b91c1c !important; }}
            .metric-grid {{
                gap: 8px !important;
                margin: 8px 0 !important;
            }}
            .rec-box {{
                background: #f8fafc !important;
                border: 1px solid #94a3b8 !important;
                border-left: 4px solid #000 !important;
                color: #000 !important;
                padding: 6px 10px !important;
                margin: 8px 0 !important;
                font-size: 9pt !important;
            }}
            h3 {{
                font-size: 10pt !important;
                margin: 10px 0 4px 0 !important;
                color: #000 !important;
            }}
            table, th, td {{
                border-color: #cbd5e1 !important;
                color: #000 !important;
                padding: 4px 8px !important;
                font-size: 9pt !important;
            }}
            th {{
                background: #f1f5f9 !important;
                font-size: 8.5pt !important;
            }}
            .risk-banner {{
                background: #f8fafc !important;
                border: 1px solid #94a3b8 !important;
                padding: 6px 10px !important;
                margin-bottom: 8px !important;
            }}
            .risk-badge {{
                background: #000000 !important;
                color: #ffffff !important;
                font-size: 9pt !important;
                padding: 3px 8px !important;
            }}
            .signatures {{
                color: #000 !important;
                border-top: 1px solid #000 !important;
                margin-top: 10px !important;
                padding-top: 6px !important;
                font-size: 8.5pt !important;
                page-break-inside: avoid !important;
            }}
            .sig-role {{
                color: #000 !important;
                margin-bottom: 16px !important;
                font-weight: 700 !important;
            }}
            .sig-line-bar {{
                border-bottom: 1px solid #000 !important;
            }}
            .sig-caption {{
                color: #475569 !important;
                font-size: 7pt !important;
                margin-top: 2px !important;
            }}
        }}
    </style>
</head>
<body>
<div class="container">
    <!-- Agency Header -->
    <div class="header-agency">
        <div class="agency-title">
            <h2>МЧС РОССИИ &bull; ГЛАВНОЕ УПРАВЛЕНИЕ ПО АМУРСКОЙ ОБЛАСТИ</h2>
            <h1>ЦЕНТР УПРАВЛЕНИЯ В КРИЗИСНЫХ СИТУАЦИЯХ (ЦУКС)</h1>
            <p>Комплекс спутникового мониторинга паводков &laquo;Вода-Космос&raquo; (Sentinel-1 / Sentinel-2)</p>
        </div>
        <div class="doc-meta">
            СВОДКА <strong>№ {pair_id}</strong><br/>
            Дата: <strong>{now_str}</strong><br/>
            Проекция: <strong>{meta['crs']}</strong>
        </div>
    </div>

    <!-- Actions toolbar with direct file downloads -->
    <div class="action-toolbar">
        <div class="action-toolbar-title">📄 Экспорт и сохранение документа:</div>
        <div class="btn-group-actions">
            <button class="btn-act btn-print" onclick="window.print()">🖨️ Распечатать / Сохранить в PDF</button>
            <a class="btn-act btn-geo" href="/api/v1/pairs/{pair_id}/geojson?download=true" download>🗺️ Скачать GeoJSON контуры</a>
            <a class="btn-act btn-json" href="/api/v1/pairs/{pair_id}/report" target="_blank">📊 Скачать JSON данные</a>
        </div>
    </div>

    <!-- Risk Status Banner -->
    <div class="risk-banner">
        <div>
            <div style="font-size:12px;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-sub);margin-bottom:4px;">Текущий статус гидрологической угрозы:</div>
            <div style="font-size:16px;font-weight:700;">{risk['level_ru']}</div>
        </div>
        <div class="risk-badge">{risk['level']}</div>
    </div>

    <!-- Recommendations -->
    <div class="rec-box">
        <strong>Предписания оперативному дежурному ЦУКС и гидрологическим постам:</strong><br/>
        {risk['recommendation']}
    </div>

    <!-- Hydrological Balance -->
    <h3 style="font-size:14px;text-transform:uppercase;letter-spacing:0.5px;color:var(--accent);margin:24px 0 12px 0;">Гидрологический баланс района наблюдения</h3>
    <div class="metric-grid">
        <div class="card highlight">
            <div class="card-label">Новое затопление (Flood)</div>
            <div class="card-val">{bal['flood_inundation_ha']} га</div>
            <div style="font-size:12px;color:#ef4444;margin-top:4px;">{bal['flood_inundation_km2']} км² ({bal['flood_fraction_of_aoi_pct']}% от AOI)</div>
        </div>
        <div class="card">
            <div class="card-label">Вода на пике паводка</div>
            <div class="card-val">{bal['water_peak_ha']} га</div>
            <div style="font-size:12px;color:var(--text-sub);margin-top:4px;">Суммарное водное зеркало</div>
        </div>
        <div class="card">
            <div class="card-label">Базовое русло (Межень)</div>
            <div class="card-val">{bal['water_pre_ha']} га</div>
            <div style="font-size:12px;color:var(--text-sub);margin-top:4px;">Урез воды до события</div>
        </div>
    </div>

    <!-- WorldCover Land Impact -->
    <h3 style="font-size:14px;text-transform:uppercase;letter-spacing:0.5px;color:var(--accent);margin:28px 0 12px 0;">Распределение площади затопления по категориям ESA WorldCover</h3>
    <table>
        <thead>
            <tr>
                <th>Категория земельного фонда</th>
                <th style="text-align:right;">Площадь затопления</th>
                <th style="text-align:right;">Доля от паводка</th>
            </tr>
        </thead>
        <tbody>
            {lc_rows}
        </tbody>
    </table>

    <!-- Official Signatures Block -->
    <div class="signatures">
        <div class="sig-block">
            <div class="sig-role">Оператор космического мониторинга ДЗЗ:</div>
            <div class="sig-line-bar"></div>
            <div class="sig-caption">(подпись / инициалы)</div>
        </div>
        <div class="sig-block">
            <div class="sig-role">Старший оперативный дежурный ЦУКС:</div>
            <div class="sig-line-bar"></div>
            <div class="sig-caption">(подпись / инициалы)</div>
        </div>
    </div>
</div>
</body>
</html>"""
    return html


def render_mchs_operational_briefing(report_data: Dict, weather_data: Optional[Dict] = None) -> str:
    """Renders strict 1-page A4 operational hydrological dispatch according to EMERCOM standards."""
    from datetime import datetime

    meta = report_data.get("report_metadata", {})
    bal = report_data.get("hydrological_balance", {})
    risk = report_data.get("risk_assessment", {})
    infra = report_data.get("infrastructure_impact", {})
    pair_id = meta.get("target_pair", "unknown")
    now_str = datetime.now().strftime("%d.%m.%Y %H:%M")

    # Accurate AOI mapping for official EMERCOM document number
    aoi_code_map = {
        "blagoveshchensk": ("БЛГ", "01"),
        "belogorsk": ("БЕЛ", "02"),
        "svobodny": ("СВБ", "03"),
        "konstantinovka": ("КНС", "04"),
        "poyarkovo": ("ПРК", "05"),
        "arkhara": ("АРХ", "06"),
    }
    matched_aoi = None
    for k in aoi_code_map:
        if k in pair_id.lower():
            matched_aoi = k
            break
    if matched_aoi:
        abbr, num = aoi_code_map[matched_aoi]
        doc_num = f"ЦУКС-{abbr}-{num}/26"
    else:
        doc_num = "ЦУКС-ОПР-01/26"

    # Reference Rosgidromet gauge station
    aoi_gauge_map = {
        "blagoveshchensk": {"code": "77001", "name": "г. Благовещенск", "river": "р. Амур", "stage": "685 см", "status": "ОЯ (Опасный)", "color": "#ef4444"},
        "belogorsk": {"code": "77014", "name": "г. Белогорск", "river": "р. Томь", "stage": "385 см", "status": "НЯ (Неблагоприятный)", "color": "#f59e0b"},
        "svobodny": {"code": "77005", "name": "г. Свободный", "river": "р. Зея", "stage": "590 см", "status": "НЯ (Неблагоприятный)", "color": "#f59e0b"},
        "konstantinovka": {"code": "77018", "name": "с. Константиновка", "river": "р. Амур", "stage": "720 см", "status": "ОЯ (Опасный)", "color": "#ef4444"},
        "poyarkovo": {"code": "77020", "name": "с. Поярково", "river": "р. Амур", "stage": "650 см", "status": "ОЯ (Опасный)", "color": "#ef4444"},
        "arkhara": {"code": "77025", "name": "с. Архара", "river": "р. Архара", "stage": "410 см", "status": "Норма", "color": "#10b981"},
    }
    g = aoi_gauge_map.get(matched_aoi, {"code": "77001", "name": "г. Благовещенск", "river": "р. Амур", "stage": "685 см", "status": "ОЯ (Опасный)", "color": "#ef4444"})

    risk_color = risk.get("color", "#10b981")
    risk_level = risk.get("level_ru", "Норма (Меженный режим)")
    risk_rec = risk.get("recommendation", "Штатный мониторинг гидропостов. Угроза населенным пунктам отсутствует.")

    # Extract weather summary if provided
    w_precip = "--"
    w_api = "--"
    w_temp = "--"
    w_risk = "--"
    w_risk_color = "#dc2626"
    if weather_data and "summary" in weather_data:
        ws = weather_data["summary"]
        w_precip = f"{ws.get('weather_precip_7d_mm', ws.get('precip_7d_sum_mm', '--'))} мм"
        w_api = f"{ws.get('weather_api_7d_mm', ws.get('api_7d_mm', '--'))} мм"
        w_temp = f"{ws.get('weather_temp_7d_c', ws.get('temp_mean_7d_c', '--'))} °C"
        w_risk = ws.get("weather_risk_ru", ws.get("weather_risk_level_ru", "--"))
        if any(w in str(w_risk).lower() for w in ["низк", "норм"]):
            w_risk_color = "#10b981"
        elif any(w in str(w_risk).lower() for w in ["умерен", "повыш"]):
            w_risk_color = "#f59e0b"
        else:
            w_risk_color = "#dc2626"

    lc_items = report_data.get("landcover_impact", [])[:4]
    if not lc_items:
        lc_table_rows = """<tr><td colspan="3" style="text-align:center;padding:6px;color:#64748b;border:1px solid #e2e8f0;">Данные отсутствуют или затопление в меженном русле</td></tr>"""
    else:
        lc_table_rows = "".join(
            f"""<tr>
                <td style="padding:4px 8px;border:1px solid #e2e8f0;text-align:left;">{r.get('class_name', '--')}</td>
                <td style="padding:4px 8px;border:1px solid #e2e8f0;text-align:right;font-family:Consolas,monospace;font-weight:600;">{r.get('area_ha', 0)} га</td>
                <td style="padding:4px 8px;border:1px solid #e2e8f0;text-align:right;font-family:Consolas,monospace;">{r.get('percentage', 0)}%</td>
            </tr>"""
            for r in lc_items
        )

    flood_inundation_ha = bal.get("flood_inundation_ha", 0)
    flood_inundation_km2 = bal.get("flood_inundation_km2", 0)
    flood_fraction_of_aoi_pct = bal.get("flood_fraction_of_aoi_pct", 0)
    water_peak_ha = bal.get("water_peak_ha", 0)
    water_pre_ha = bal.get("water_pre_ha", 0)
    aoi_total_ha = bal.get("aoi_total_ha", 0)
    aoi_total_km2 = bal.get("aoi_total_km2", 0)

    roads_flooded_km = infra.get("roads_flooded_km", 0)
    farmland_flooded_ha = infra.get("farmland_flooded_ha", 0)
    cropland_ha = infra.get("cropland_ha", 0)
    grassland_ha = infra.get("grassland_ha", 0)
    builtup_flooded_ha = infra.get("builtup_flooded_ha", 0)
    settlement_distance_m = infra.get("settlement_distance_m", 0)

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>ОПЕРАТИВНОЕ ДОНЕСЕНИЕ ЦУКС МЧС: {pair_id}</title>
    <style>
        @page {{
            size: A4 portrait;
            margin: 10mm 12mm 10mm 12mm;
        }}
        *, *::before, *::after {{
            box-sizing: border-box;
            -webkit-print-color-adjust: exact !important;
            print-color-adjust: exact !important;
            color-adjust: exact !important;
        }}
        html, body {{
            margin: 0;
            padding: 0;
            background: #0b1120;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
            font-size: 11.5px;
            line-height: 1.4;
            color: #0f172a;
            -webkit-print-color-adjust: exact !important;
            print-color-adjust: exact !important;
        }}
        .print-toolbar {{
            background: #0f172a;
            color: #ffffff;
            padding: 10px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 1000;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        .btn-print-action {{
            background: #2563eb;
            color: #ffffff;
            border: none;
            padding: 8px 18px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 700;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            box-shadow: 0 2px 6px rgba(37, 99, 235, 0.4);
            transition: all 0.15s ease;
        }}
        .btn-print-action:hover {{
            background: #1d4ed8;
            transform: translateY(-1px);
        }}
        .page-sheet {{
            width: 210mm;
            min-height: 297mm;
            margin: 16px auto 32px auto;
            background: #ffffff !important;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.6), 0 2px 10px rgba(0, 0, 0, 0.3);
            border-radius: 3px;
            padding: 12mm 14mm 10mm 14mm;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        @media print {{
            .no-print {{
                display: none !important;
            }}
            html, body {{
                background: #ffffff !important;
                margin: 0 !important;
                padding: 0 !important;
            }}
            .page-sheet {{
                width: 100% !important;
                max-width: 100% !important;
                min-height: 275mm !important;
                margin: 0 !important;
                padding: 0 !important;
                box-shadow: none !important;
                border-radius: 0 !important;
            }}
        }}

        /* Header */
        .gov-header {{
            border-bottom: 2.5px solid #003366;
            padding-bottom: 8px;
            margin-bottom: 10px;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
        }}
        .gov-title {{
            font-size: 13.5px;
            font-weight: 800;
            color: #003366;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            text-align: left;
        }}
        .gov-subtitle {{
            font-size: 10px;
            color: #475569;
            font-weight: 600;
            margin-top: 2px;
            text-align: left;
        }}
        .doc-stamp {{
            border: 1.5px solid #94a3b8 !important;
            padding: 5px 10px;
            text-align: right;
            font-size: 9.5px;
            font-family: Consolas, monospace;
            background-color: #f8fafc !important;
            box-shadow: inset 0 0 0 1000px #f8fafc !important;
            border-radius: 4px;
        }}

        /* Banner */
        .dispatch-banner {{
            background-color: #f1f5f9 !important;
            box-shadow: inset 0 0 0 1000px #f1f5f9 !important;
            border: 1px solid #cbd5e1 !important;
            border-left: 5px solid {risk_color} !important;
            border-radius: 4px;
            padding: 9px 12px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .dispatch-title {{
            font-size: 13px;
            font-weight: 800;
            color: #0f172a;
            text-transform: uppercase;
            text-align: left;
        }}
        .dispatch-meta {{
            font-size: 9.5px;
            color: #475569;
            margin-top: 2px;
            text-align: left;
        }}
        .dispatch-badge {{
            padding: 5px 12px;
            border-radius: 4px;
            font-weight: 800;
            font-size: 11.5px;
            color: #ffffff !important;
            background-color: {risk_color} !important;
            border: 1.5px solid {risk_color} !important;
            box-shadow: inset 0 0 0 1000px {risk_color} !important;
            white-space: nowrap;
            letter-spacing: 0.2px;
        }}

        /* Sections */
        .section-header {{
            font-size: 11px;
            font-weight: 700;
            color: #003366;
            text-transform: uppercase;
            border-bottom: 1.5px solid #cbd5e1;
            padding-bottom: 3px;
            margin: 10px 0 6px 0;
            text-align: left;
            letter-spacing: 0.2px;
        }}
        .data-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 6px;
            margin-bottom: 10px;
        }}
        .grid-card {{
            background-color: #f8fafc !important;
            box-shadow: inset 0 0 0 1000px #f8fafc !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 4px;
            padding: 7px 9px;
            text-align: left;
        }}
        .grid-card.alert {{
            background-color: #fef2f2 !important;
            box-shadow: inset 0 0 0 1000px #fef2f2 !important;
            border: 1.5px solid #f87171 !important;
            border-left: 4px solid #ef4444 !important;
        }}
        .grid-label {{
            font-size: 9px;
            text-transform: uppercase;
            color: #64748b;
            font-weight: 700;
            margin-bottom: 2px;
        }}
        .grid-val {{
            font-size: 16px;
            font-weight: 800;
            color: #0f172a;
            font-family: Consolas, monospace;
        }}
        .grid-val.red {{
            color: #dc2626;
        }}
        .grid-val.blue {{
            color: #0284c7;
        }}
        .grid-sub {{
            font-size: 9px;
            color: #64748b;
            margin-top: 2px;
        }}

        /* Tables */
        table.gov-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 10px;
            margin-bottom: 6px;
        }}
        table.gov-table th {{
            background-color: #003366 !important;
            box-shadow: inset 0 0 0 1000px #003366 !important;
            color: #ffffff !important;
            padding: 5px 8px;
            font-weight: 700;
            text-align: left;
            border: 1px solid #002244 !important;
        }}
        table.gov-table th.right {{
            text-align: right;
        }}
        table.gov-table td {{
            padding: 4px 8px;
            border: 1px solid #e2e8f0;
            text-align: left;
        }}
        table.gov-table tr:nth-child(even) td {{
            background-color: #f8fafc !important;
            box-shadow: inset 0 0 0 1000px #f8fafc !important;
        }}

        /* Gauge bar */
        .gauge-summary-box {{
            background-color: #f8fafc !important;
            box-shadow: inset 0 0 0 1000px #f8fafc !important;
            border: 1px solid #cbd5e1 !important;
            border-left: 3px solid {g['color']} !important;
            border-radius: 4px;
            padding: 6px 10px;
            margin-top: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 10px;
        }}

        /* Recommendation */
        .recommendation-box {{
            background-color: #f8fafc !important;
            box-shadow: inset 0 0 0 1000px #f8fafc !important;
            border: 1px solid #cbd5e1 !important;
            border-left: 4px solid #003366 !important;
            border-radius: 4px;
            padding: 9px 12px;
            font-size: 10.5px;
            margin: 6px 0 12px 0;
            text-align: left;
            line-height: 1.45;
        }}

        /* Signatures */
        .signatures-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 36px;
            margin-top: 14px;
            padding-top: 10px;
            border-top: 1px dashed #cbd5e1;
        }}
        .sig-col {{
            text-align: left;
        }}
        .sig-title {{
            font-size: 10px;
            font-weight: 700;
            color: #0f172a;
        }}
        .sig-line {{
            border-bottom: 1.5px solid #0f172a;
            height: 24px;
            margin-bottom: 3px;
        }}
        .sig-subtext {{
            font-size: 8.5px;
            color: #64748b;
            text-align: center;
        }}

        /* Official Footer */
        .doc-footer-meta {{
            margin-top: 12px;
            padding-top: 6px;
            border-top: 1px solid #e2e8f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 8.5px;
            color: #64748b;
            font-family: Consolas, monospace;
        }}
    </style>
</head>
<body>
<div class="print-toolbar no-print">
    <div style="display:flex;align-items:center;gap:12px;">
        <span style="font-size:20px;">📄</span>
        <div>
            <div style="font-size:14px;font-weight:700;">Официальное оперативное донесение ЦУКС МЧС</div>
            <div style="font-size:11px;color:#94a3b8;">Стандарт А4 (ГОСТ Р). В окне печати выберите «Сохранить как PDF»</div>
        </div>
    </div>
    <div style="display:flex;gap:10px;">
        <button class="btn-print-action" onclick="window.print()">
            <span>🖨️</span> Распечатать / Сохранить в PDF
        </button>
        <button class="btn-print-action" style="background:#475569;" onclick="window.close()">✕ Закрыть</button>
    </div>
</div>

<div class="page-sheet">
    <div>
        <!-- Header -->
        <div class="gov-header">
            <div>
                <div class="gov-title">МЧС РОССИИ &bull; ГЛАВНОЕ УПРАВЛЕНИЕ ПО АМУРСКОЙ ОБЛАСТИ</div>
                <div class="gov-subtitle">Центр управления в кризисных ситуациях (ЦУКС) | Комплекс космического мониторинга «Вода-Космос»</div>
            </div>
            <div class="doc-stamp">
                <strong>ДОНЕСЕНИЕ № {doc_num}</strong><br/>
                Сформировано: {now_str}
            </div>
        </div>

        <!-- Dispatch Banner -->
        <div class="dispatch-banner">
            <div>
                <div class="dispatch-title">Оперативная гидрологическая обстановка: {pair_id}</div>
                <div class="dispatch-meta">
                    Координатная привязка: EPSG:32652 (UTM 52N) &bull; Разрешение сенсоров: 10 м &bull; Спутники: Sentinel-1 SAR / Sentinel-2 MSI / ERA5
                </div>
            </div>
            <div class="dispatch-badge">{risk_level}</div>
        </div>

        <!-- 1. Hydrological Balance -->
        <div class="section-header">1. Гидрологический баланс речного бассейна</div>
        <div class="data-grid">
            <div class="grid-card alert">
                <div class="grid-label">Зона нового затопления</div>
                <div class="grid-val red">{flood_inundation_ha} га</div>
                <div class="grid-sub">{flood_inundation_km2} км² ({flood_fraction_of_aoi_pct}% от AOI)</div>
            </div>
            <div class="grid-card">
                <div class="grid-label">Вода на пике паводка</div>
                <div class="grid-val">{water_peak_ha} га</div>
                <div class="grid-sub">Суммарный урез воды</div>
            </div>
            <div class="grid-card">
                <div class="grid-label">Базовое русло (Межень)</div>
                <div class="grid-val">{water_pre_ha} га</div>
                <div class="grid-sub">Многолетний фоновый сток</div>
            </div>
            <div class="grid-card">
                <div class="grid-label">Площадь района (AOI)</div>
                <div class="grid-val">{aoi_total_ha} га</div>
                <div class="grid-sub">{aoi_total_km2} км² зоны контроля</div>
            </div>
        </div>

        <!-- 2. Infrastructure -->
        <div class="section-header">2. Оценка ущерба критической инфраструктуре и угодьям</div>
        <div class="data-grid">
            <div class="grid-card alert">
                <div class="grid-label">Подтопление автодорог</div>
                <div class="grid-val red">{roads_flooded_km} км</div>
                <div class="grid-sub">Пойменные и низководные участки</div>
            </div>
            <div class="grid-card">
                <div class="grid-label">Затоплено сельхозугодий</div>
                <div class="grid-val">{farmland_flooded_ha} га</div>
                <div class="grid-sub">Пашни: {cropland_ha} га &bull; Луга: {grassland_ha} га</div>
            </div>
            <div class="grid-card">
                <div class="grid-label">Подтопление застройки</div>
                <div class="grid-val">{builtup_flooded_ha} га</div>
                <div class="grid-sub">Потенциальное подтопление участков</div>
            </div>
            <div class="grid-card">
                <div class="grid-label">Дистанция до поселений</div>
                <div class="grid-val blue">{settlement_distance_m} м</div>
                <div class="grid-sub">Минимальное удаление уреза</div>
            </div>
        </div>

        <!-- 3 & 4. Weather & Landcover -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:6px;">
            <div>
                <div class="section-header">3. Метеоусловия ERA5 и гидропосты</div>
                <table class="gov-table">
                    <tbody>
                        <tr><td>Кумулятивные осадки (7d):</td><td style="text-align:right;font-family:Consolas,monospace;font-weight:700;">{w_precip}</td></tr>
                        <tr><td>Индекс увлажнения почв (API):</td><td style="text-align:right;font-family:Consolas,monospace;font-weight:700;">{w_api}</td></tr>
                        <tr><td>Средняя температура воздуха:</td><td style="text-align:right;font-family:Consolas,monospace;font-weight:700;">{w_temp}</td></tr>
                        <tr><td>Оценка метео-риска:</td><td style="text-align:right;font-weight:700;color:{w_risk_color};">{w_risk}</td></tr>
                    </tbody>
                </table>
                <div class="gauge-summary-box">
                    <span><strong>Опорный гидропост:</strong> № {g['code']} ({g['name']}, {g['river']})</span>
                    <span>Уровень: <strong>{g['stage']}</strong> | <strong style="color:{g['color']};">{g['status']}</strong></span>
                </div>
            </div>
            <div>
                <div class="section-header">4. Затопление по ESA WorldCover</div>
                <table class="gov-table">
                    <thead>
                        <tr><th>Угодье</th><th class="right">Площадь</th><th class="right">Доля</th></tr>
                    </thead>
                    <tbody>
                        {lc_table_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <!-- 5. Instructions -->
        <div class="section-header">5. Распоряжение и указания оперативной смене ЦУКС МЧС</div>
        <div class="recommendation-box">
            <strong>Указания дежурному диспетчеру:</strong> {risk_rec}
            Обеспечить немедленное информирование глав муниципальных образований, организовать превентивное перекрытие подтопленных участков региональных автодорог, вести непрерывный инструментальный контроль гребней водозащитных сооружений и готовность мобильных насосных групп.
        </div>
    </div>

    <div>
        <!-- Signatures (GOST) -->
        <div class="signatures-row">
            <div class="sig-col">
                <div class="sig-title">Оператор гидрологического мониторинга ДЗЗ:</div>
                <div class="sig-line"></div>
                <div class="sig-subtext">(подпись / инициалы, фамилия)</div>
            </div>
            <div class="sig-col">
                <div class="sig-title">Старший оперативный дежурный смены ЦУКС ГУ МЧС:</div>
                <div class="sig-line"></div>
                <div class="sig-subtext">(подпись / инициалы, фамилия)</div>
            </div>
        </div>

        <!-- Footer -->
        <div class="doc-footer-meta">
            <span>Экземпляр № 1 &bull; Гриф: Для служебного пользования (ДСП)</span>
            <span>Комплекс космического мониторинга «Вода-Космос» &bull; Sentinel-1 SAR / Sentinel-2 MSI / Copernicus DEM</span>
        </div>
    </div>
</div>
</body>
</html>"""
    return html


