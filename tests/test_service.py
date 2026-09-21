"""Unit tests for FastAPI service and Web-GIS endpoints (src/service/)."""

import pytest
from pathlib import Path
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
    assert data["flood_ha"] == pytest.approx(386.3, 0.5)
    assert data["risk_level"] in ["Повышенный (Пойменное затопление)", "Высокий (Опасный паводок)", "Критический (Чрезвычайная ситуация)"]


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


def test_aoi_vectors_endpoint():
    """Endpoint /api/v1/vectors/aoi must return official AOI GeoJSON polygons."""
    resp = client.get("/api/v1/vectors/aoi")
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 5
    aoi_ids = {f["properties"]["aoi_id"] for f in data["features"]}
    assert "blagoveshchensk" in aoi_ids
    assert "svobodny" in aoi_ids


def test_weather_endpoint():
    """Endpoint /api/v1/weather/{pair_id} must return ERA5 time series and risk summary."""
    resp = client.get("/api/v1/weather/flood_2019_07_amur__blagoveshchensk")
    assert resp.status_code == 200
    data = resp.json()
    assert data["pair_id"] == "flood_2019_07_amur__blagoveshchensk"
    assert "summary" in data
    assert "daily_timeline" in data
    assert "weather_precip_7d_mm" in data["summary"]
    assert "weather_risk_level" in data["summary"]
    assert len(data["daily_timeline"]) > 0
    assert any(d["is_peak"] for d in data["daily_timeline"])


def test_geojson_aoi_boundary():
    """Pair GeoJSON endpoint must include the official AOI boundary feature."""
    resp = client.get("/api/v1/pairs/flood_2019_07_amur__blagoveshchensk/geojson")
    assert resp.status_code == 200
    data = resp.json()
    types = [f["properties"].get("type") for f in data["features"]]
    assert "aoi_boundary" in types
    assert "water_pre" in types
    assert "flood" in types


def test_shapefile_endpoint():
    """Endpoint /api/v1/pairs/{pair_id}/shapefile must return valid ESRI Shapefile zip."""
    import zipfile
    import io

    resp = client.get("/api/v1/pairs/flood_2019_07_amur__blagoveshchensk/shapefile")
    assert resp.status_code == 200
    assert "application/zip" in resp.headers["content-type"]
    assert "flood_shapefile.zip" in resp.headers["content-disposition"]

    zf = zipfile.ZipFile(io.BytesIO(resp.content))
    filenames = zf.namelist()
    extensions = {Path(fn).suffix.lower() for fn in filenames}
    assert ".shp" in extensions
    assert ".shx" in extensions
    assert ".dbf" in extensions
    assert ".prj" in extensions



def test_mchs_report_endpoint():
    """Endpoint /api/v1/pairs/{pair_id}/report/mchs must serve 1-page A4 EMERCOM dispatch."""
    resp = client.get("/api/v1/pairs/flood_2019_07_amur__blagoveshchensk/report/mchs")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "МЧС РОССИИ" in resp.text
    assert "ДОНЕСЕНИЕ" in resp.text
    assert "Оценка ущерба критической инфраструктуре" in resp.text
    assert "@page" in resp.text


def test_infrastructure_impact_calculation():
    """Function calculate_infrastructure_impact must quantify road and farmland impact."""
    from src.service.report import calculate_infrastructure_impact

    impact = calculate_infrastructure_impact(flood_ha=500.0, aoi_id="blagoveshchensk")
    assert impact["roads_flooded_km"] > 0
    assert impact["farmland_flooded_ha"] > 0
    assert impact["settlement_distance_m"] > 0
    assert "threat_color" in impact


def test_what_if_endpoint():
    """Endpoint /api/v1/pairs/{pair_id}/what_if must return valid hydraulic forecast."""
    resp = client.get("/api/v1/pairs/flood_2019_07_amur__blagoveshchensk/what_if?delta_h=1.5")
    assert resp.status_code == 200
    data = resp.json()
    assert data["delta_h_meters"] == 1.5
    assert data["forecast_flood_ha"] > data["base_flood_ha"]
    assert data["delta_flood_ha"] > 0
    assert "geojson_feature" in data
    assert data["geojson_feature"]["geometry"]["type"] == "Polygon"
    assert "infrastructure" in data


def test_gauge_stations_endpoint():
    """Endpoint /api/v1/gauge_stations must return Rosgidromet monitoring network."""
    resp = client.get("/api/v1/gauge_stations?pair_id=flood_2019_07_amur__blagoveshchensk")
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 6
    codes = {f["properties"]["code"] for f in data["features"]}
    assert "07001" in codes  # Blagoveshchensk Amur
    assert "07034" in codes  # Svobodny Zeya
    for f in data["features"]:
        props = f["properties"]
        assert props["current_stage_cm"] > 0
        assert props["stage_oya_cm"] > props["stage_nya_cm"]
        assert "hydrograph_5d" in props


def test_cross_section_profile_endpoint():
    """Endpoint /api/v1/pairs/{pair_id}/profile must return river cross-section geometry."""
    resp = client.get("/api/v1/pairs/flood_2019_07_amur__blagoveshchensk/profile?delta_h=1.0")
    assert resp.status_code == 200
    data = resp.json()
    assert "cross_section_name" in data
    assert "profile_points" in data
    assert len(data["profile_points"]) == 60
    assert data["hydraulics"]["channel_width_pre_m"] > 0
    assert data["hydraulics"]["floodplain_width_peak_m"] > 0
    assert data["water_levels"]["delta_h_m"] == 1.0


def test_quick_audit_execution():
    """scripts/quick_audit.py main function must execute and verify solution."""
    from scripts.quick_audit import main
    # Should complete without throwing exceptions
    main()


def test_traps_endpoint():
    """Endpoint /api/v1/pairs/{pair_id}/traps must return 4 domain traps with geometries."""
    resp = client.get("/api/v1/pairs/flood_2019_07_amur__blagoveshchensk/traps")
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 4
    trap_types = {f["properties"]["trap_type"] for f in data["features"]}
    assert "airport_builtup" in trap_types
    assert "oxbow_lake" in trap_types
    assert "waterlogged_cropland" in trap_types
    assert "dry_sandbar" in trap_types
    assert data["total_suppressed_ha"] > 500.0



def test_benchmark_latency_execution():
    """scripts/benchmark_latency.py must run and generate structured benchmark report."""
    from scripts.benchmark_latency import run_benchmark
    res = run_benchmark(grid_size=512)
    assert "stages" in res
    assert len(res["stages"]) == 6
    assert res["summary"]["total_latency_ms"] > 0.0
    assert res["summary"]["peak_ram_mb"] < 16384.0


def test_timelapse_endpoint():
    """Endpoint /api/v1/pairs/{pair_id}/timelapse must return 6-phase temporal sequence."""
    resp = client.get("/api/v1/pairs/flood_2019_07_amur__blagoveshchensk/timelapse")
    assert resp.status_code == 200
    data = resp.json()
    assert "frames" in data
    assert len(data["frames"]) == 6
    assert data["frames"][0]["step"] == 0
    assert data["frames"][3]["flood_fraction"] == 1.0  # Peak
    for f in data["frames"]:
        assert "date" in f
        assert "phase_name" in f
        assert "feature_collection" in f
        assert f["total_water_ha"] > 0


def test_evacuation_endpoint():
    """Endpoint /api/v1/pairs/{pair_id}/evacuation must return cut-off roads and safe shelters."""
    resp = client.get("/api/v1/pairs/flood_2019_07_amur__blagoveshchensk/evacuation")
    assert resp.status_code == 200
    data = resp.json()
    assert "summary" in data
    assert "cut_off_roads" in data
    assert "isolated_communities" in data
    assert "safe_zones" in data
    assert len(data["cut_off_roads"]) > 0
    assert len(data["isolated_communities"]) > 0
    assert len(data["safe_zones"]) > 0
    assert data["summary"]["total_cut_km"] > 0


def test_geopackage_endpoint():
    """Endpoint /api/v1/pairs/{pair_id}/geopackage must stream valid OGC GeoPackage bytes."""
    resp = client.get("/api/v1/pairs/flood_2019_07_amur__blagoveshchensk/geopackage")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/geopackage+sqlite3"
    assert len(resp.content) > 1000  # SQLite database header + tables


def test_kmz_endpoint():
    """Endpoint /api/v1/pairs/{pair_id}/kmz must stream valid Google Earth 3D KMZ bytes."""
    import zipfile
    import io
    resp = client.get("/api/v1/pairs/flood_2019_07_amur__blagoveshchensk/kmz")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/vnd.google-earth.kmz"
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        assert "doc.kml" in zf.namelist()
        kml_content = zf.read("doc.kml").decode("utf-8")
        assert "<kml" in kml_content
        assert "<Polygon>" in kml_content


def test_benchmark_highload_execution():
    """scripts/benchmark_highload.py must execute async load test and return metrics."""
    import asyncio
    from scripts.benchmark_highload import run_load_test
    res = asyncio.run(run_load_test(total_requests=16, concurrency=4, pair_id="flood_2019_07_amur__blagoveshchensk"))
    assert res["throughput"]["successful_requests"] == 16
    assert res["throughput"]["failed_requests"] == 0
    assert res["throughput"]["requests_per_second"] > 0.0
    assert res["latency_ms"]["avg_ms"] > 0.0


def test_ablation_endpoint():
    """Endpoint /api/v1/ablation must return official incremental study steps."""
    resp = client.get("/api/v1/ablation")
    assert resp.status_code == 200
    data = resp.json()
    assert data["final_score"] == 0.99211
    assert len(data["experiments"]) == 6
    assert data["experiments"][0]["id"] == "baseline_sar"
    assert data["experiments"][-1]["score"] == 0.99211
    for exp in data["experiments"]:
        assert "name" in exp
        assert "score" in exp
        assert "delta_score" in exp
        assert "contribution" in exp






