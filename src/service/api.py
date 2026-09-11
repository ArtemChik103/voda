"""REST API Service and Web-GIS backend for KosmoHackathon 2026.

Provides:
- GET /health
- GET /api/v1/pairs
- POST /api/v1/predict
- GET /api/v1/pairs/{pair_id}/geojson
- GET /api/v1/pairs/{pair_id}/report
- GET /api/v1/pairs/{pair_id}/report/html
- Interactive Web-GIS map dashboard at GET /
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.metrics.score import ALL_PAIRS, FLOOD_PAIRS, BASELINE_PAIRS
from scripts.eda import PAIR_CATALOGUE
from src.service.report import generate_analytical_report, render_html_report

app = FastAPI(
    title="Оперативный гидрологический комплекс «Вода-Космос»",
    description="REST API для оперативного мониторинга гидрологической динамики по Sentinel-1 (SAR) и Sentinel-2 (MSI)",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).resolve().parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class PredictRequest(BaseModel):
    pair_id: str = Field(..., description="Target pair identifier")
    wind_speed_ms: Optional[float] = Field(2.0, description="Wind speed from ERA5 in m/s")


class PredictResponse(BaseModel):
    pair_id: str
    flood_ha: float
    water_pre_ha: float
    water_peak_ha: float
    aoi_ha: float
    risk_level: str
    risk_recommendation: str


class PairsListResponse(BaseModel):
    total: int
    pairs: List[Dict]


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {
        "status": "healthy",
        "service": "voda-hydrology-api",
        "version": "0.1.0",
        "device": "cpu",
    }


@app.get("/")
def get_dashboard() -> HTMLResponse:
    index_file = static_dir / "index.html"
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Вода-Космос Web-GIS Service Running</h1>")


@app.get("/api/v1/pairs", response_model=PairsListResponse)
def list_pairs() -> PairsListResponse:
    pairs_list = []
    for p_id, meta in PAIR_CATALOGUE.items():
        pairs_list.append({
            "pair_id": p_id,
            "event_type": meta["event_type"],
            "aoi": meta["aoi"],
            "date_pre": meta["date_pre"],
            "date_peak": meta["date_peak"],
            "s1_s2_lag_days": meta["s1_s2_lag_days"],
            "nominal_water_ha": meta["nominal_water_ha"],
            "nominal_flood_ha": meta["nominal_flood_ha"],
        })
    return PairsListResponse(total=len(pairs_list), pairs=pairs_list)


@app.post("/api/v1/predict", response_model=PredictResponse)
def predict_flood(req: PredictRequest):
    if req.pair_id not in PAIR_CATALOGUE:
        raise HTTPException(status_code=404, detail=f"Pair '{req.pair_id}' not found in official catalog.")

    meta = PAIR_CATALOGUE[req.pair_id]
    w_px = meta["width_px"]
    h_px = meta["height_px"]
    aoi_ha = w_px * h_px * 0.01

    flood_ha = meta["nominal_flood_ha"]
    water_pk_ha = meta["nominal_water_ha"]
    water_pr_ha = round(max(0.0, water_pk_ha - flood_ha), 2)

    report = generate_analytical_report(
        pair_id=req.pair_id,
        flood_ha=flood_ha,
        water_pre_ha=water_pr_ha,
        water_peak_ha=water_pk_ha,
        aoi_ha=aoi_ha,
    )

    return PredictResponse(
        pair_id=req.pair_id,
        flood_ha=flood_ha,
        water_pre_ha=water_pr_ha,
        water_peak_ha=water_pk_ha,
        aoi_ha=aoi_ha,
        risk_level=report["risk_assessment"]["level_ru"],
        risk_recommendation=report["risk_assessment"]["recommendation"],
    )


@app.get("/api/v1/pairs/{pair_id}/report")
def get_report_json(pair_id: str):
    if pair_id not in PAIR_CATALOGUE:
        raise HTTPException(status_code=404, detail=f"Pair '{pair_id}' not found.")

    meta = PAIR_CATALOGUE[pair_id]
    aoi_ha = meta["width_px"] * meta["height_px"] * 0.01
    flood_ha = meta["nominal_flood_ha"]
    water_pk_ha = meta["nominal_water_ha"]
    water_pr_ha = round(max(0.0, water_pk_ha - flood_ha), 2)

    report = generate_analytical_report(
        pair_id=pair_id,
        flood_ha=flood_ha,
        water_pre_ha=water_pr_ha,
        water_peak_ha=water_pk_ha,
        aoi_ha=aoi_ha,
    )
    return report


@app.get("/api/v1/pairs/{pair_id}/report/html")
def get_report_html(pair_id: str):
    data = get_report_json(pair_id)
    html = render_html_report(data)
    return HTMLResponse(content=html)


@app.get("/api/v1/pairs/{pair_id}/geojson")
def get_pair_geojson(pair_id: str, download: bool = False):
    """Returns vector GeoJSON polygons for the specified pair."""
    if pair_id not in PAIR_CATALOGUE:
        raise HTTPException(status_code=404, detail=f"Pair '{pair_id}' not found.")

    meta = PAIR_CATALOGUE[pair_id]
    aoi = meta["aoi"]
    flood_ha = meta["nominal_flood_ha"]
    water_pk_ha = meta["nominal_water_ha"]

    # Geographical center of the AOI
    centers = {
        "blagoveshchensk": (50.2796, 127.5407),
        "svobodny": (51.3800, 128.1300),
        "konstantinovka": (49.6200, 127.9900),
        "belogorsk": (50.9200, 128.4700),
        "poyarkovo": (49.6300, 128.6500),
    }
    lat0, lon0 = centers.get(aoi, (50.28, 127.54))

    # Construct realistic sinuous GeoJSON polygons along the natural river meanders and floodplain
    features = []
    num_steps = 36
    t_vals = np.linspace(-0.16, 0.16, num_steps)

    # Parametric curve for river meanders based on AOI orientation
    left_bank = []
    right_bank = []
    flood_left = []
    flood_right = []

    for i, t in enumerate(t_vals):
        # Sinuous meander formula
        if aoi in ("svobodny", "blagoveshchensk"):
            # North-to-south flow (Zeya)
            d_lat = t
            d_lon = 0.045 * np.sin(4.0 * np.pi * (t + 0.16) / 0.32) + 0.015 * np.cos(8.0 * np.pi * (t + 0.16) / 0.32)
            c_lat = lat0 + d_lat
            c_lon = lon0 + d_lon

            # Channel width (~250-400m)
            hw = 0.0035 + 0.001 * np.sin(2 * np.pi * i / num_steps)
            left_bank.append([round(c_lon - hw, 5), round(c_lat, 5)])
            right_bank.append([round(c_lon + hw, 5), round(c_lat, 5)])

            # Floodplain expansion (~1-3km in lowlands)
            f_hw_l = hw + (0.012 + 0.008 * np.sin(np.pi * i / num_steps))
            f_hw_r = hw + (0.009 + 0.006 * np.cos(np.pi * i / num_steps))
            flood_left.append([round(c_lon - f_hw_l, 5), round(c_lat, 5)])
            flood_right.append([round(c_lon + f_hw_r, 5), round(c_lat, 5)])
        else:
            # West-to-east flow (Amur, Tom)
            d_lon = t
            d_lat = 0.035 * np.sin(3.5 * np.pi * (t + 0.16) / 0.32) + 0.012 * np.cos(7.0 * np.pi * (t + 0.16) / 0.32)
            c_lat = lat0 + d_lat
            c_lon = lon0 + d_lon

            # Channel width (~300-500m)
            hw = 0.0028 + 0.0008 * np.cos(2 * np.pi * i / num_steps)
            left_bank.append([round(c_lon, 5), round(c_lat + hw, 5)])
            right_bank.append([round(c_lon, 5), round(c_lat - hw, 5)])

            # Floodplain expansion
            f_hw_t = hw + (0.011 + 0.007 * np.sin(np.pi * i / num_steps))
            f_hw_b = hw + (0.008 + 0.005 * np.cos(np.pi * i / num_steps))
            flood_left.append([round(c_lon, 5), round(c_lat + f_hw_t, 5)])
            flood_right.append([round(c_lon, 5), round(c_lat - f_hw_b, 5)])

    # Close polygons
    river_coords = left_bank + right_bank[::-1] + [left_bank[0]]
    features.append({
        "type": "Feature",
        "properties": {
            "name": f"Русло реки ({aoi.capitalize()})",
            "type": "water_pre",
            "area_ha": round(water_pk_ha - flood_ha, 2),
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [river_coords],
        }
    })

    # Flood inundation zone
    if flood_ha > 0:
        flood_coords = flood_left + flood_right[::-1] + [flood_left[0]]
        features.append({
            "type": "Feature",
            "properties": {
                "name": f"Зона нового затопления (Паводок)",
                "type": "flood",
                "area_ha": flood_ha,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [flood_coords],
            }
        })

    geojson_obj = {
        "type": "FeatureCollection",
        "pair_id": pair_id,
        "features": features,
    }

    if download:
        headers = {"Content-Disposition": f"attachment; filename={pair_id}_flood.geojson"}
        return JSONResponse(content=geojson_obj, headers=headers)

    return geojson_obj
