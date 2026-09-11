"""Unit tests for FastAPI service and Web-GIS endpoints (src/service/)."""

import pytest
from fastapi.testclient import TestClient

from src.service.api import app
from src.service.report import assess_hydrological_risk, calculate_landcover_breakdown
import numpy as np

client = TestClient(app)


def test_health_check():
    """Health check endpoint must return healthy status."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "voda-hydrology-api"


def test_dashboard_html():
    """Root endpoint must serve interactive dashboard HTML."""
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Вода-Космос" in resp.text
    assert "leaflet" in resp.text.lower()


def test_list_pairs():
    """Pairs catalog must return all 11 competition pairs."""
    resp = client.get("/api/v1/pairs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 11
    assert len(data["pairs"]) == 11


def test_predict_endpoint():
    """Predict endpoint returns correct flood statistics and risk assessment."""
    payload = {"pair_id": "flood_2019_07_amur__blagoveshchensk"}
    resp = client.post("/api/v1/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["pair_id"] == "flood_2019_07_amur__blagoveshchensk"
    assert data["flood_ha"] > 1000.0
    assert data["risk_level"] in ["Высокий (Опасный паводок)", "Критический (Чрезвычайная ситуация)"]


def test_geojson_endpoint():
    """GeoJSON endpoint returns valid GeoJSON polygons for web rendering."""
    resp = client.get("/api/v1/pairs/flood_2019_07_amur__blagoveshchensk/geojson")
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) >= 2  # river + flood
    assert data["features"][0]["geometry"]["type"] == "Polygon"


def test_report_endpoints():
    """JSON and HTML report endpoints return valid executive summaries."""
    # JSON report
    resp_json = client.get("/api/v1/pairs/flood_2019_07_amur__belogorsk/report")
    assert resp_json.status_code == 200
    rep = resp_json.json()
    assert "hydrological_balance" in rep
    assert "landcover_impact" in rep
    assert len(rep["landcover_impact"]) > 0

    # HTML report
    resp_html = client.get("/api/v1/pairs/flood_2019_07_amur__belogorsk/report/html")
    assert resp_html.status_code == 200
    assert "text/html" in resp_html.headers["content-type"]
    assert "ЦЕНТР УПРАВЛЕНИЯ В КРИЗИСНЫХ СИТУАЦИЯХ" in resp_html.text


def test_landcover_breakdown():
    """Land cover breakdown must partition flood pixels properly."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:20, :20] = 1  # 400 pixels = 4 ha
    breakdown = calculate_landcover_breakdown(mask)
    assert len(breakdown) > 0
    total_pct = sum(b["percentage"] for b in breakdown)
    assert pytest.approx(total_pct, abs=1.0) == 100.0
