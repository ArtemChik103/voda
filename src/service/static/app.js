// Web-GIS Dashboard Application Logic

let map;
let preLayer = null;
let floodLayer = null;
let aoiLayer = null;
let whatIfLayer = null;
let gaugeStationsLayer = null;
let profilePolyline = null;
let trapsLayer = null;
let currentPairId = null;
let currentLayerMode = 'all';
let currentSwipePct = 50;
let isDraggingSwipe = false;
let currentWhatIfDelta = 0.0;
let showGauges = true;
let isProfileOpen = false;
let showTraps = false;
let lastNominalInfra = null;
let currentBaseTileLayer = null;
let isOfflineBasemap = false;
let currentAoiBounds = null;

// AOI Coordinates in WGS84
const AOI_CENTERS = {
    "blagoveshchensk": [50.2796, 127.5407],
    "svobodny": [51.3800, 128.1300],
    "konstantinovka": [49.6200, 127.9900],
    "belogorsk": [50.9200, 128.4700],
    "poyarkovo": [49.6300, 128.6500],
};

document.addEventListener("DOMContentLoaded", () => {
    initMap();
    initSwipeEvents();
    initPointInspector();
    initKeyboardShortcuts();
    loadPairsList();

    document.getElementById("eventSelect").addEventListener("change", (e) => {
        onSelectPair(e.target.value);
    });
});

function switchSidebarTab(tabName) {
    const isOverview = tabName === 'overview';
    const overviewContent = document.getElementById("tabContentOverview");
    const forecastContent = document.getElementById("tabContentForecast");
    const btnOverview = document.getElementById("tabBtnOverview");
    const btnForecast = document.getElementById("tabBtnForecast");

    if (overviewContent) {
        overviewContent.style.display = isOverview ? "flex" : "none";
    }
    if (forecastContent) {
        forecastContent.style.display = isOverview ? "none" : "flex";
    }
    if (btnOverview) {
        btnOverview.classList.toggle("active", isOverview);
    }
    if (btnForecast) {
        btnForecast.classList.toggle("active", !isOverview);
    }
}

function initMap() {
    // Initial view on Amur Region, Blagoveshchensk
    map = L.map('map', {
        center: [50.28, 127.54],
        zoom: 10,
        zoomControl: true,
    });

    // Remove any third-party political flag icons or prefixes from map attribution
    if (map.attributionControl) {
        map.attributionControl.setPrefix(false);
    }

    // Dedicated panes for Swipe split-screen comparison
    map.createPane('prePane');
    map.getPane('prePane').style.zIndex = 410;
    map.createPane('floodPane');
    map.getPane('floodPane').style.zIndex = 420;

    // 1. High-Resolution Satellite imagery (Esri World Imagery)
    const esriSatellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics',
        maxZoom: 18
    });

    // 2. Pure Dark Hydrological Map (OSM with dark filter)
    const osmDark = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        className: 'osm-dark-tiles',
        maxZoom: 19
    });

    // 3. OpenStreetMap Standard
    const osmLight = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 19
    });

    // Default to satellite basemap for space monitoring context
    currentBaseTileLayer = esriSatellite;
    esriSatellite.addTo(map);

    map.on('baselayerchange', (e) => {
        currentBaseTileLayer = e.layer;
    });

    const baseLayers = {
        "🛰️ Спутник (Esri Satellite)": esriSatellite,
        "🗺️ Тёмная карта (OSM Dark)": osmDark,
        "🗺️ Топографическая (OSM)": osmLight
    };

    L.control.layers(baseLayers, null, { position: 'topright' }).addTo(map);
    loadGaugeStations();

    // Recalculate swipe clipping on map move and zoom
    map.on('move', () => {
        if (currentLayerMode === 'swipe') {
            updateSwipeClip(currentSwipePct);
        }
    });
    map.on('zoom', () => {
        if (currentLayerMode === 'swipe') {
            updateSwipeClip(currentSwipePct);
        }
    });
}

function initSwipeEvents() {
    const divider = document.getElementById('swipeDivider');
    const container = document.getElementById('mapContainer');
    if (!divider || !container) return;

    divider.addEventListener('mousedown', (e) => {
        isDraggingSwipe = true;
        e.preventDefault();
    });

    window.addEventListener('mouseup', () => {
        isDraggingSwipe = false;
    });

    window.addEventListener('mousemove', (e) => {
        if (!isDraggingSwipe || currentLayerMode !== 'swipe') return;
        const rect = container.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const pct = Math.max(5, Math.min(95, (x / rect.width) * 100));
        updateSwipeClip(pct);
    });

    // Mobile / Touch Dragging
    divider.addEventListener('touchstart', () => {
        isDraggingSwipe = true;
    }, { passive: true });

    window.addEventListener('touchend', () => {
        isDraggingSwipe = false;
    });

    window.addEventListener('touchmove', (e) => {
        if (!isDraggingSwipe || currentLayerMode !== 'swipe' || !e.touches[0]) return;
        const rect = container.getBoundingClientRect();
        const x = e.touches[0].clientX - rect.left;
        const pct = Math.max(5, Math.min(95, (x / rect.width) * 100));
        updateSwipeClip(pct);
    }, { passive: true });
}

function updateSwipeClip(pct) {
    currentSwipePct = pct;
    const divider = document.getElementById('swipeDivider');
    if (divider) {
        divider.style.left = `${pct}%`;
    }

    if (currentLayerMode !== 'swipe' || !map) return;

    const fp = document.querySelector('.leaflet-flood-pane');
    const pp = document.querySelector('.leaflet-pre-pane');
    if (!fp || !pp) return;

    const mapSize = map.getSize();
    const x = mapSize.x * (pct / 100);

    const nw = map.containerPointToLayerPoint([0, 0]);
    const se = map.containerPointToLayerPoint(mapSize);
    const clipX = map.containerPointToLayerPoint([x, 0]).x;

    // Clip pre-event water (river) to the LEFT of the divider
    pp.style.clip = 'rect(' + nw.y + 'px, ' + clipX + 'px, ' + se.y + 'px, ' + nw.x + 'px)';
    pp.style.clipPath = 'none';

    // Clip flood inundation to the RIGHT of the divider
    fp.style.clip = 'rect(' + nw.y + 'px, ' + se.x + 'px, ' + se.y + 'px, ' + clipX + 'px)';
    fp.style.clipPath = 'none';
}

const AOI_NAMES_RU = {
    "blagoveshchensk": "📍 г. Благовещенск (р. Амур / р. Зея)",
    "belogorsk": "📍 г. Белогорск (р. Томь)",
    "svobodny": "📍 г. Свободный (р. Зея)",
    "konstantinovka": "📍 с. Константиновка (р. Амур)",
    "poyarkovo": "📍 пгт Поярково (р. Амур / р. Завитая)",
};

function formatPairLabel(pairId, eventType) {
    if (pairId.includes("2019_07")) {
        return "🌊 Паводок 2019 г. (Пик разлива)";
    } else if (pairId.includes("2021_08")) {
        return "🌊 Паводок 2021 г. (Экстремальный паводок)";
    } else if (pairId.includes("baseline_2018_09")) {
        return "🌾 Меженный режим 2018 г. (Естественное русло)";
    } else {
        const badge = eventType === "flood" ? "🌊 Паводок" : "🌾 Межень";
        return `[${badge}] ${pairId}`;
    }
}

async function loadPairsList() {
    try {
        const response = await fetch("/api/v1/pairs");
        const data = await response.json();
        const select = document.getElementById("eventSelect");
        select.innerHTML = "";

        // Group pairs by AOI into optgroups
        const groups = {};
        data.pairs.forEach((p) => {
            let aoiKey = "other";
            for (const k of Object.keys(AOI_NAMES_RU)) {
                if (p.pair_id.includes(k)) {
                    aoiKey = k;
                    break;
                }
            }
            if (!groups[aoiKey]) groups[aoiKey] = [];
            groups[aoiKey].push(p);
        });

        for (const [aoiKey, pairs] of Object.entries(groups)) {
            const optgroup = document.createElement("optgroup");
            optgroup.label = AOI_NAMES_RU[aoiKey] || aoiKey;
            pairs.forEach((p) => {
                const opt = document.createElement("option");
                opt.value = p.pair_id;
                opt.textContent = formatPairLabel(p.pair_id, p.event_type);
                optgroup.appendChild(opt);
            });
            select.appendChild(optgroup);
        }

        // Select first flood event by default
        const defaultPair = data.pairs.find(p => p.event_type === "flood") || data.pairs[0];
        if (defaultPair) {
            select.value = defaultPair.pair_id;
            onSelectPair(defaultPair.pair_id);
        }
    } catch (err) {
        console.error("Failed to load pairs list:", err);
    }
}

async function onSelectPair(pairId) {
    currentPairId = pairId;
    updateSidebarLoading();

    try {
        // 1. Fetch prediction & report data
        const reportResp = await fetch(`/api/v1/pairs/${pairId}/report`);
        const report = await reportResp.json();
        renderMetrics(report);

        // 2. Fetch GeoJSON geometries
        const geoResp = await fetch(`/api/v1/pairs/${pairId}/geojson`);
        const geoData = await geoResp.json();
        renderGeoJSON(geoData, pairId);

        // 3. Fetch ERA5 meteorological data
        try {
            const weatherResp = await fetch(`/api/v1/weather/${pairId}`);
            if (weatherResp.ok) {
                const weatherData = await weatherResp.json();
                renderWeather(weatherData);
            }
        } catch (wErr) {
            console.warn("Weather data unavailable for", pairId, wErr);
        }

        // 4. Update Gauges, What-If & Profile for newly active pair
        applyWhatIf(0.0);
        loadGaugeStations();
        if (isProfileOpen) {
            loadCrossSectionProfile();
        }
        if (showTraps) {
            loadTrapsLayer();
        }
        if (timelapseActive) {
            loadTimelapseData();
        }
        if (evacuationActive) {
            loadEvacuationData();
        }

    } catch (err) {
        console.error("Error loading pair data:", err);
    }
}

function renderWeather(data) {
    if (!data || !data.summary) return;
    const s = data.summary;

    const p7d = s.precip_7d_sum_mm ?? s.weather_precip_7d_mm ?? 0;
    const api7d = s.api_7d_mm ?? s.weather_api_7d_mm ?? 0;
    const temp7d = s.temp_mean_7d_c ?? s.weather_temp_7d_c ?? 0;
    const riskLevel = s.weather_risk_level || "LOW";
    const riskRu = s.weather_risk_level_ru || s.weather_risk_ru || riskLevel;

    document.getElementById("valPrecip7d").textContent = `${p7d} мм`;
    document.getElementById("valApi7d").textContent = `${api7d} мм`;
    const tempSign = temp7d > 0 ? "+" : "";
    document.getElementById("valWeatherTemp").textContent = `${tempSign}${temp7d} °C`;

    const badge = document.getElementById("weatherRiskBadge");
    badge.textContent = riskRu;
    if (riskLevel === "CRITICAL" || riskLevel === "EXTREME") {
        badge.style.background = "#dc2626";
    } else if (riskLevel === "HIGH") {
        badge.style.background = "#ea580c";
    } else if (riskLevel === "MODERATE") {
        badge.style.background = "#d97706";
    } else {
        badge.style.background = "#16a34a";
    }

    // Render daily precipitation bars
    const barsContainer = document.getElementById("weatherBars");
    barsContainer.innerHTML = "";

    const timeline = data.daily_timeline || [];
    if (timeline.length === 0) {
        barsContainer.innerHTML = '<div style="color: #64748b; font-size: 12px;">Нет суточных данных</div>';
        return;
    }

    const maxPrecip = Math.max(...timeline.map(d => d.precip_mm), 10.0);

    timeline.forEach(d => {
        const barCol = document.createElement("div");
        barCol.className = "weather-bar-col";
        const heightPct = Math.max(4, Math.round((d.precip_mm / maxPrecip) * 100));
        const isPeak = d.is_peak;

        barCol.title = `${d.date}: ${d.precip_mm} мм (t: ${d.temp_c}°C, API: ${d.api_mm} мм)${isPeak ? " — ДАТА ПИКА" : ""}`;

        barCol.innerHTML = `
            <div class="weather-bar-fill ${isPeak ? 'peak' : ''}" style="height: ${heightPct}%;"></div>
        `;
        barsContainer.appendChild(barCol);
    });
}

function renderMetrics(report) {
    const bal = report.hydrological_balance;
    const risk = report.risk_assessment;

    document.getElementById("valFloodHa").textContent = `${bal.flood_inundation_ha.toLocaleString('ru-RU')} га`;
    document.getElementById("valFloodKm2").textContent = `${bal.flood_inundation_km2} км² (${bal.flood_fraction_of_aoi_pct}% от AOI)`;
    document.getElementById("valPeakHa").textContent = `${bal.water_peak_ha.toLocaleString('ru-RU')} га`;
    document.getElementById("valPreHa").textContent = `${bal.water_pre_ha.toLocaleString('ru-RU')} га`;

    const badge = document.getElementById("riskBadge");
    badge.textContent = risk.level_ru;
    badge.style.background = risk.color;
    document.getElementById("riskText").textContent = risk.recommendation;

    // Infrastructure impact metrics
    if (report.infrastructure_impact) {
        lastNominalInfra = report.infrastructure_impact;
        const infra = report.infrastructure_impact;
        document.getElementById("valInfraRoads").textContent = `${infra.roads_flooded_km} км`;
        document.getElementById("valInfraFarmland").textContent = `${infra.farmland_flooded_ha} га`;
        const threatEl = document.getElementById("valInfraThreat");
        threatEl.textContent = infra.settlement_threat_ru;
        document.getElementById("valInfraDist").textContent = `Дистанция до застройки: ${infra.settlement_distance_m} м`;
        const box = document.getElementById("infraThreatBox");
        if (box && infra.threat_color) {
            box.style.borderLeftColor = infra.threat_color;
        }
    }

    // Land cover bars
    const barsContainer = document.getElementById("landcoverBars");
    barsContainer.innerHTML = "";

    if (!report.landcover_impact || report.landcover_impact.length === 0) {
        barsContainer.innerHTML = `
            <div style="background: rgba(30, 41, 59, 0.5); border: 1px dashed #475569; border-radius: 6px; padding: 10px; color: #94a3b8; font-size: 12px; text-align: center; margin-top: 4px;">
                🌾 Паводковое затопление отсутствует<br/><span style="font-size: 11px; color: #64748b;">(Меженный естественный режим русла)</span>
            </div>
        `;
    } else {
        report.landcover_impact.forEach(lc => {
            const item = document.createElement("div");
            item.className = "lc-bar-item";
            item.innerHTML = `
                <div class="lc-bar-header">
                    <span>${lc.class_name}</span>
                    <strong>${lc.area_ha} га (${lc.percentage}%)</strong>
                </div>
                <div class="lc-bar-track">
                    <div class="lc-bar-fill" style="width: ${lc.percentage}%; background: ${lc.color};"></div>
                </div>
            `;
            barsContainer.appendChild(item);
        });
    }
}

function renderGeoJSON(geoData, pairId) {
    if (preLayer) map.removeLayer(preLayer);
    if (floodLayer) map.removeLayer(floodLayer);
    if (aoiLayer) map.removeLayer(aoiLayer);

    const aoiFeats = [];
    const preFeats = [];
    const floodFeats = [];

    (geoData.features || []).forEach(f => {
        const t = f.properties && f.properties.type;
        if (t === 'aoi_boundary') {
            aoiFeats.push(f);
        } else if (t === 'flood') {
            floodFeats.push(f);
        } else {
            preFeats.push(f);
        }
    });

    // 1. AOI Boundary (transparent perimeter, clicks pass through)
    if (aoiFeats.length > 0) {
        aoiLayer = L.geoJSON({ type: "FeatureCollection", features: aoiFeats }, {
            interactive: false,
            style: {
                color: '#f59e0b',
                weight: 2.5,
                dashArray: '6, 6',
                fillOpacity: 0.0,
            }
        }).addTo(map);
    }

    // 2. Pre-event water ("До") in prePane
    if (preFeats.length > 0) {
        preLayer = L.geoJSON({ type: "FeatureCollection", features: preFeats }, {
            pane: 'prePane',
            style: { color: '#0284c7', weight: 1.2, fillOpacity: 0.55, fillColor: '#0284c7' },
            onEachFeature: (feature, layer) => {
                layer.on('click', (e) => {
                    handlePointInspect(e.latlng, e.originalEvent);
                });
            }
        }).addTo(map);
    }

    // 3. Flood inundation zone ("Пик") in floodPane
    if (floodFeats.length > 0) {
        floodLayer = L.geoJSON({ type: "FeatureCollection", features: floodFeats }, {
            pane: 'floodPane',
            style: { color: '#ef4444', weight: 1.5, fillOpacity: 0.70, fillColor: '#ef4444' },
            onEachFeature: (feature, layer) => {
                layer.on('click', (e) => {
                    handlePointInspect(e.latlng, e.originalEvent);
                });
            }
        }).addTo(map);
    }

    // Zoom to features bounds or AOI center
    const allLayersGroup = L.featureGroup([aoiLayer, preLayer, floodLayer].filter(Boolean));
    const bounds = allLayersGroup.getBounds();
    if (bounds.isValid()) {
        currentAoiBounds = bounds;
        map.fitBounds(bounds, { padding: [30, 30] });
    } else {
        for (const [aoiKey, coords] of Object.entries(AOI_CENTERS)) {
            if (pairId.includes(aoiKey)) {
                map.setView(coords, 10);
                break;
            }
        }
    }

    // Apply current layer mode
    setLayerMode(currentLayerMode);
}

function setLayerMode(mode) {
    currentLayerMode = mode;
    ['btnLayerAll', 'btnLayerSwipe'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.classList.remove('active');
    });

    const activeBtn = (mode === 'swipe') ? 'btnLayerSwipe' : 'btnLayerAll';
    const btnEl = document.getElementById(activeBtn);
    if (btnEl) btnEl.classList.add('active');

    const swipeDivider = document.getElementById('swipeDivider');
    const floodPane = map.getPane('floodPane');

    if (mode === 'swipe') {
        if (swipeDivider) swipeDivider.style.display = 'block';
        updateSwipeClip(currentSwipePct);
        if (preLayer) preLayer.setStyle({ opacity: 1.0, fillOpacity: 0.60 });
        if (floodLayer) floodLayer.setStyle({ opacity: 1.0, fillOpacity: 0.70 });
        return;
    }

    // Default 'all' combined mode
    if (swipeDivider) swipeDivider.style.display = 'none';
    const pp = map.getPane('prePane');
    if (floodPane) {
        floodPane.style.clip = 'auto';
        floodPane.style.clipPath = 'none';
        floodPane.style.webkitClipPath = 'none';
    }
    if (pp) {
        pp.style.clip = 'auto';
        pp.style.clipPath = 'none';
        pp.style.webkitClipPath = 'none';
    }

    if (preLayer) preLayer.setStyle({ opacity: 1.0, fillOpacity: 0.55 });
    if (floodLayer) floodLayer.setStyle({ opacity: 1.0, fillOpacity: 0.70 });
}

function toggleSwipeMode() {
    setLayerMode(currentLayerMode === 'swipe' ? 'all' : 'swipe');
}

function updateSidebarLoading() {
    document.getElementById("valFloodHa").textContent = "Расчет...";
    document.getElementById("valFloodKm2").textContent = "Загрузка...";
    document.getElementById("valPeakHa").textContent = "Расчет...";
    document.getElementById("valPreHa").textContent = "Расчет...";
    document.getElementById("valPrecip7d").textContent = "...";
    document.getElementById("valApi7d").textContent = "...";
    document.getElementById("valWeatherTemp").textContent = "...";
    document.getElementById("weatherRiskBadge").textContent = "...";
    document.getElementById("weatherRiskBadge").style.background = "#475569";
    document.getElementById("weatherBars").innerHTML = '<div style="color: #64748b; font-size: 12px;">Загрузка метеоданных...</div>';
    document.getElementById("valInfraRoads").textContent = "...";
    document.getElementById("valInfraFarmland").textContent = "...";
    document.getElementById("valInfraThreat").textContent = "Расчет...";
}

function toggleExportMenu() {
    const menu = document.getElementById("exportMenu");
    if (!menu) return;
    const isShowing = menu.style.display === "block";
    menu.style.display = isShowing ? "none" : "block";
}

function closeExportMenu() {
    const menu = document.getElementById("exportMenu");
    if (menu) menu.style.display = "none";
}

function exportGeoJSON() {
    closeExportMenu();
    if (!currentPairId) return;
    window.open(`/api/v1/pairs/${currentPairId}/geojson?download=true`, '_blank');
}

function exportShapefile() {
    closeExportMenu();
    if (!currentPairId) return;
    window.open(`/api/v1/pairs/${currentPairId}/shapefile`, '_blank');
}

function exportGeoPackage() {
    closeExportMenu();
    if (!currentPairId) return;
    window.open(`/api/v1/pairs/${currentPairId}/geopackage`, '_blank');
}

function exportKMZ() {
    closeExportMenu();
    if (!currentPairId) return;
    window.open(`/api/v1/pairs/${currentPairId}/kmz`, '_blank');
}

function openMchsReport() {
    closeExportMenu();
    if (!currentPairId) return;
    window.open(`/api/v1/pairs/${currentPairId}/report/mchs`, '_blank');
}

function exportMapSnapshot() {
    closeExportMenu();
    if (!currentPairId) {
        alert("Пожалуйста, выберите пару снимков для выгрузки картографического снимка.");
        return;
    }
    const a = document.createElement("a");
    a.href = `/api/v1/pairs/${currentPairId}/snapshot.png`;
    a.download = `amur_flood_map_${currentPairId}_300dpi.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// Close export dropdown on outside click
document.addEventListener("click", function(e) {
    const container = document.querySelector(".export-dropdown-container");
    const menu = document.getElementById("exportMenu");
    if (container && menu && !container.contains(e.target)) {
        menu.style.display = "none";
    }
});

// ==========================================
// 1. ROSGIDROMET GAUGE STATIONS
// ==========================================
async function loadGaugeStations() {
    try {
        const url = currentPairId ? `/api/v1/gauge_stations?pair_id=${currentPairId}` : '/api/v1/gauge_stations';
        const resp = await fetch(url);
        if (!resp.ok) return;
        const data = await resp.json();

        if (gaugeStationsLayer) {
            map.removeLayer(gaugeStationsLayer);
        }

        const markers = [];
        data.features.forEach(f => {
            const p = f.properties;
            const [lon, lat] = f.geometry.coordinates;

            const isOya = p.status_code === 'OYA';
            const pulseClass = isOya ? 'gauge-oya-pulse' : '';

            const iconHtml = `<div class="gauge-marker-icon ${pulseClass}" style="background:${p.status_color};">📍</div>`;
            const customIcon = L.divIcon({
                className: 'gauge-marker-wrap',
                html: iconHtml,
                iconSize: [26, 26],
                iconAnchor: [13, 13],
                popupAnchor: [0, -14],
            });

            const marker = L.marker([lat, lon], { icon: customIcon });

            const popupContent = `
                <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;min-width:240px;color:#0f172a;">
                    <div style="font-size:11px;font-weight:700;text-transform:uppercase;color:#64748b;margin-bottom:2px;">Пост №${p.code} • р. ${p.river}</div>
                    <div style="font-size:14px;font-weight:700;color:#0f172a;margin-bottom:6px;">${p.name}</div>
                    <div style="background:#f1f5f9;border-radius:6px;padding:8px;margin-bottom:8px;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
                            <span style="font-size:12px;color:#475569;">Текущий горизонт:</span>
                            <strong style="font-family:Consolas,monospace;font-size:14px;color:${p.status_color};">${p.current_stage_cm} см</strong>
                        </div>
                        <div style="display:flex;justify-content:space-between;font-size:11px;color:#64748b;margin-bottom:2px;">
                            <span>Выход на пойму:</span>
                            <span style="font-family:Consolas,monospace;">${p.stage_floodplain_cm} см</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;font-size:11px;color:#ea580c;margin-bottom:2px;">
                            <span>Неблагоприятное (НЯ):</span>
                            <span style="font-family:Consolas,monospace;font-weight:600;">${p.stage_nya_cm} см</span>
                        </div>
                        <div style="display:flex;justify-content:space-between;font-size:11px;color:#dc2626;">
                            <span>Опасное (ОЯ):</span>
                            <span style="font-family:Consolas,monospace;font-weight:700;">${p.stage_oya_cm} см</span>
                        </div>
                    </div>
                    <div style="font-size:11px;font-weight:600;color:${p.status_color};margin-bottom:4px;">
                        Статус: ${p.status_name} (${p.pct_to_oya}% до ОЯ)
                    </div>
                    <div style="font-size:10px;color:#64748b;line-height:1.3;">${p.description}</div>
                </div>
            `;
            marker.bindPopup(popupContent);
            markers.push(marker);
        });

        gaugeStationsLayer = L.featureGroup(markers);
        if (showGauges) {
            gaugeStationsLayer.addTo(map);
        }
    } catch (err) {
        console.error("Failed to load gauge stations:", err);
    }
}

function toggleGaugeStations() {
    showGauges = !showGauges;
    const btn = document.getElementById("btnToggleGauges");
    if (btn) btn.classList.toggle("active", showGauges);

    if (gaugeStationsLayer) {
        if (showGauges) {
            gaugeStationsLayer.addTo(map);
        } else {
            map.removeLayer(gaugeStationsLayer);
        }
    }
}

// ==========================================
// 2. WHAT-IF PREDICTIVE HYDRAULIC MODELING
// ==========================================
function onWhatIfSliderInput(val) {
    const deltaH = parseFloat(val);
    const label = document.getElementById("whatIfSliderLabel");
    if (label) {
        label.textContent = `+${deltaH.toFixed(1)} м`;
    }
}

async function applyWhatIf(deltaH) {
    currentWhatIfDelta = deltaH;

    // Sync slider and label
    const slider = document.getElementById("whatIfSlider");
    if (slider) slider.value = deltaH.toFixed(1);
    const label = document.getElementById("whatIfSliderLabel");
    if (label) label.textContent = `+${deltaH.toFixed(1)} м`;

    // Update button active state
    document.querySelectorAll(".btn-whatif").forEach(b => {
        const hVal = parseFloat(b.getAttribute("data-h"));
        b.classList.toggle("active", Math.abs(hVal - deltaH) < 0.05);
    });

    const box = document.getElementById("whatIfResultBox");

    if (deltaH === 0.0) {
        if (whatIfLayer) {
            map.removeLayer(whatIfLayer);
            whatIfLayer = null;
        }
        if (box) box.style.display = "none";

        // Restore nominal infrastructure metrics
        if (lastNominalInfra) {
            document.getElementById("valInfraRoads").textContent = `${lastNominalInfra.roads_flooded_km} км`;
            document.getElementById("valInfraFarmland").textContent = `${lastNominalInfra.farmland_flooded_ha} га`;
            document.getElementById("valInfraThreat").textContent = lastNominalInfra.settlement_threat_ru;
            document.getElementById("valInfraDist").textContent = `Дистанция до застройки: ${lastNominalInfra.settlement_distance_m} м`;
            const tBox = document.getElementById("infraThreatBox");
            if (tBox && lastNominalInfra.threat_color) {
                tBox.style.borderLeftColor = lastNominalInfra.threat_color;
            }
        }
        if (isProfileOpen) loadCrossSectionProfile();
        return;
    }

    if (!currentPairId) return;

    try {
        const resp = await fetch(`/api/v1/pairs/${currentPairId}/what_if?delta_h=${deltaH}`);
        if (!resp.ok) return;
        const res = await resp.json();

        if (box) {
            box.style.display = "block";
            document.getElementById("valWhatIfArea").textContent = `${res.forecast_flood_ha.toLocaleString('ru-RU')} га`;
            document.getElementById("valWhatIfDelta").textContent = `+${res.delta_flood_ha.toLocaleString('ru-RU')} га`;
            document.getElementById("valWhatIfRec").textContent = res.recommendation;
        }

        // Update infrastructure panel with forecast impact
        if (res.infrastructure) {
            const infra = res.infrastructure;
            document.getElementById("valInfraRoads").textContent = `${infra.roads_flooded_km} км`;
            document.getElementById("valInfraFarmland").textContent = `${infra.farmland_flooded_ha} га`;
            document.getElementById("valInfraThreat").textContent = infra.settlement_threat_ru;
            document.getElementById("valInfraDist").textContent = `Дистанция до застройки: ${infra.settlement_distance_m} м`;
            const tBox = document.getElementById("infraThreatBox");
            if (tBox && infra.threat_color) {
                tBox.style.borderLeftColor = infra.threat_color;
            }
        }

        // Render forecast polygon on map
        if (whatIfLayer) {
            map.removeLayer(whatIfLayer);
        }

        whatIfLayer = L.geoJSON(res.geojson_feature, {
            style: {
                color: "#c084fc",
                weight: 2,
                dashArray: "6, 4",
                fillColor: "#a855f7",
                fillOpacity: 0.35,
            }
        }).bindPopup(`
            <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#0f172a;">
                <strong style="color:#7c3aed;">Прогнозная зона затопления (What-If)</strong><br>
                <span>Подъем горизонта воды: <strong>+${deltaH.toFixed(1)} м</strong></span><br>
                <span>Прогнозная площадь: <strong>${res.forecast_flood_ha} га</strong> (+${res.delta_flood_ha} га)</span><br>
                <span style="color:#dc2626;font-weight:600;">Уровень тревоги: ${res.risk_level}</span>
            </div>
        `).addTo(map);

        if (isProfileOpen) loadCrossSectionProfile();

    } catch (err) {
        console.error("Failed to calculate What-If forecast:", err);
    }
}

// ==========================================
// 3. CROSS-SECTION RIVER PROFILE DRAWER
// ==========================================
function toggleProfileDrawer() {
    isProfileOpen = !isProfileOpen;
    const drawer = document.getElementById("profileDrawer");
    const btn = document.getElementById("btnToggleProfile");
    if (drawer) {
        drawer.style.display = isProfileOpen ? "flex" : "none";
    }
    if (btn) {
        btn.classList.toggle("active", isProfileOpen);
    }

    if (isProfileOpen) {
        // Mutual exclusion: Close other map tools
        if (typeof timelapseActive !== "undefined" && timelapseActive) {
            toggleTimelapse();
        }
        if (typeof evacuationActive !== "undefined" && evacuationActive) {
            toggleEvacuationLayer();
        }
        autoCollapseLegendForDrawer(true);
        loadCrossSectionProfile();
    } else {
        if (profilePolyline) {
            map.removeLayer(profilePolyline);
            profilePolyline = null;
        }
        autoCollapseLegendForDrawer(false);
    }
}

async function loadCrossSectionProfile() {
    if (!currentPairId) return;

    try {
        const resp = await fetch(`/api/v1/pairs/${currentPairId}/profile?delta_h=${currentWhatIfDelta}`);
        if (!resp.ok) return;
        const data = await resp.json();

        // Update polyline on map
        if (profilePolyline) {
            map.removeLayer(profilePolyline);
        }
        profilePolyline = L.polyline([data.start_coords, data.end_coords], {
            color: "#38bdf8",
            weight: 3,
            dashArray: "6, 6",
        }).bindTooltip(`Створ долины: ${data.cross_section_name}`, { sticky: true }).addTo(map);

        // Update header and metrics
        document.getElementById("profileTitle").textContent = `📊 ${data.cross_section_name} (${data.total_length_m} м)`;
        document.getElementById("valPWidthPre").textContent = `${data.hydraulics.channel_width_pre_m} м`;
        document.getElementById("valPWidthPeak").textContent = `${data.hydraulics.floodplain_width_peak_m} м`;
        document.getElementById("valPDepthMax").textContent = `${data.hydraulics.max_water_depth_peak_m} м`;
        document.getElementById("valPArea").textContent = `${data.hydraulics.wet_cross_section_area_m2.toLocaleString('ru-RU')} м²`;

        // Render SVG Chart
        renderProfileSvg(data);
    } catch (err) {
        console.error("Failed to load cross-section profile:", err);
    }
}

function renderProfileSvg(data) {
    const svg = document.getElementById("profileSvg");
    if (!svg) return;
    svg.innerHTML = "";

    const pts = data.profile_points;
    if (!pts || pts.length === 0) return;

    const container = document.getElementById("profileChartContainer");
    const W = Math.max(750, container ? container.clientWidth : 850);
    const H = Math.max(150, container ? container.clientHeight : 170);

    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

    const padL = 65;
    const padR = 35;
    const padT = 24;
    const padB = 26;

    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    const totalLen = data.total_length_m;
    const wl = data.water_levels;

    const elevs = pts.map(p => p.elevation_dem_m);
    const minZ = Math.min(...elevs, wl.datum_m) - 1.5;
    const maxZ = Math.max(...elevs, wl.water_whatif_abs_m || wl.water_peak_abs_m) + 2.0;

    const mapX = (dist) => padL + (dist / totalLen) * plotW;
    const mapY = (elev) => padT + plotH - ((elev - minZ) / (maxZ - minZ)) * plotH;

    let svgHtml = `
        <defs>
            <linearGradient id="pDemGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="#334155" stop-opacity="0.95"/>
                <stop offset="100%" stop-color="#0f172a" stop-opacity="0.95"/>
            </linearGradient>
            <linearGradient id="pPeakGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.75"/>
                <stop offset="100%" stop-color="#0284c7" stop-opacity="0.40"/>
            </linearGradient>
            <linearGradient id="pPreGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="#0284c7" stop-opacity="0.95"/>
                <stop offset="100%" stop-color="#0369a1" stop-opacity="0.75"/>
            </linearGradient>
            <linearGradient id="pWhatIfGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stop-color="#c084fc" stop-opacity="0.60"/>
                <stop offset="100%" stop-color="#9333ea" stop-opacity="0.25"/>
            </linearGradient>
        </defs>
    `;

    // 1. Elevation Grid Lines (horizontal)
    const zStep = (maxZ - minZ) > 15 ? 5 : 2;
    const startZ = Math.ceil(minZ / zStep) * zStep;
    for (let z = startZ; z <= maxZ; z += zStep) {
        const y = mapY(z);
        svgHtml += `<line x1="${padL}" y1="${y}" x2="${W - padR}" y2="${y}" stroke="#1e293b" stroke-width="1" stroke-dasharray="3,3"/>`;
        svgHtml += `<text x="${padL - 8}" y="${y + 4}" fill="#64748b" font-size="10" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" text-anchor="end">${z} м</text>`;
    }

    // Distance ticks (vertical)
    const dStep = totalLen > 8000 ? 2000 : 1000;
    for (let d = 0; d <= totalLen; d += dStep) {
        const x = mapX(d);
        svgHtml += `<line x1="${x}" y1="${padT}" x2="${x}" y2="${H - padB}" stroke="#1e293b" stroke-width="1"/>`;
        svgHtml += `<text x="${x}" y="${H - padB + 16}" fill="#64748b" font-size="10" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" text-anchor="middle">${(d / 1000).toFixed(1)} км</text>`;
    }

    // Bank labels
    svgHtml += `<text x="${padL + 6}" y="${padT + 10}" fill="#94a3b8" font-size="10" font-weight="600" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif">◀ Левый берег</text>`;
    svgHtml += `<text x="${W - padR - 6}" y="${padT + 10}" fill="#94a3b8" font-size="10" font-weight="600" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" text-anchor="end">Правый берег ▶</text>`;

    // 2. Terrain Ground Polygon
    let terrainPath = `M ${mapX(pts[0].distance_m)} ${mapY(pts[0].elevation_dem_m)}`;
    pts.forEach(p => {
        terrainPath += ` L ${mapX(p.distance_m)} ${mapY(p.elevation_dem_m)}`;
    });
    const groundBottomY = H - padB;
    const terrainClosed = `${terrainPath} L ${mapX(pts[pts.length - 1].distance_m)} ${groundBottomY} L ${mapX(pts[0].distance_m)} ${groundBottomY} Z`;
    svgHtml += `<path d="${terrainClosed}" fill="url(#pDemGrad)" stroke="#475569" stroke-width="1.5"/>`;

    // 3. Peak Flood Water Body
    const yPeak = mapY(wl.water_peak_abs_m);
    let peakWaterPoints = pts.filter(p => p.depth_peak_m > 0);

    if (peakWaterPoints.length > 1) {
        const xStart = mapX(peakWaterPoints[0].distance_m);
        const xEnd = mapX(peakWaterPoints[peakWaterPoints.length - 1].distance_m);
        let waterPoly = `M ${xStart} ${yPeak}`;
        peakWaterPoints.forEach(p => {
            waterPoly += ` L ${mapX(p.distance_m)} ${mapY(p.elevation_dem_m)}`;
        });
        waterPoly += ` L ${xEnd} ${yPeak} Z`;
        svgHtml += `<path d="${waterPoly}" fill="url(#pPeakGrad)" stroke="#38bdf8" stroke-width="1.8"/>`;
        svgHtml += `<line x1="${xStart}" y1="${yPeak}" x2="${xEnd}" y2="${yPeak}" stroke="#38bdf8" stroke-width="2"/>`;
    }

    // 4. Pre-flood low water channel
    const yPre = mapY(wl.water_pre_abs_m);
    let preWaterPoints = pts.filter(p => p.depth_pre_m > 0);
    if (preWaterPoints.length > 1) {
        const xStart = mapX(preWaterPoints[0].distance_m);
        const xEnd = mapX(preWaterPoints[preWaterPoints.length - 1].distance_m);
        let prePoly = `M ${xStart} ${yPre}`;
        preWaterPoints.forEach(p => {
            prePoly += ` L ${mapX(p.distance_m)} ${mapY(p.elevation_dem_m)}`;
        });
        prePoly += ` L ${xEnd} ${yPre} Z`;
        svgHtml += `<path d="${prePoly}" fill="url(#pPreGrad)" stroke="#0284c7" stroke-width="2"/>`;
        svgHtml += `<line x1="${xStart}" y1="${yPre}" x2="${xEnd}" y2="${yPre}" stroke="#0284c7" stroke-width="2"/>`;
    }

    // 5. What-If Water Horizon (if delta_h > 0)
    if (wl.delta_h_m > 0 && wl.water_whatif_abs_m) {
        const yWhatIf = mapY(wl.water_whatif_abs_m);
        let whatIfPoints = pts.filter(p => p.depth_whatif_m > 0);
        if (whatIfPoints.length > 1) {
            const xStart = mapX(whatIfPoints[0].distance_m);
            const xEnd = mapX(whatIfPoints[whatIfPoints.length - 1].distance_m);
            let whatIfPoly = `M ${xStart} ${yWhatIf}`;
            whatIfPoints.forEach(p => {
                whatIfPoly += ` L ${mapX(p.distance_m)} ${mapY(p.elevation_dem_m)}`;
            });
            whatIfPoly += ` L ${xEnd} ${yWhatIf} Z`;
            svgHtml += `<path d="${whatIfPoly}" fill="url(#pWhatIfGrad)" stroke="#c084fc" stroke-width="2" stroke-dasharray="4,3"/>`;
        }
        svgHtml += `
            <rect x="${W - padR - 190}" y="${yWhatIf - 16}" width="185" height="18" rx="4" fill="rgba(15,23,42,0.85)" stroke="#c084fc" stroke-width="1"/>
            <text x="${W - padR - 8}" y="${yWhatIf - 3}" fill="#c084fc" font-size="10" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="700" text-anchor="end">What-If: +${wl.delta_h_m.toFixed(1)}м (${wl.water_whatif_abs_m}м БС)</text>
        `;
    }

    // High contrast water level badges
    svgHtml += `
        <rect x="${padL + 8}" y="${yPeak - 18}" width="124" height="18" rx="4" fill="rgba(15,23,42,0.85)" stroke="#38bdf8" stroke-width="1"/>
        <text x="${padL + 14}" y="${yPeak - 5}" fill="#38bdf8" font-size="10" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="700">Пик: ${wl.water_peak_abs_m} м БС</text>
    `;

    svgHtml += `
        <rect x="${padL + 8}" y="${yPre + 4}" width="136" height="18" rx="4" fill="rgba(15,23,42,0.85)" stroke="#0284c7" stroke-width="1"/>
        <text x="${padL + 14}" y="${yPre + 17}" fill="#38bdf8" font-size="10" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif" font-weight="600">Межень: ${wl.water_pre_abs_m} м БС</text>
    `;

    svg.innerHTML = svgHtml;
}

// ==========================================
// 4. TRAP INSPECTOR & DIAGNOSTIC LAYER
// ==========================================
function toggleTrapsLayer() {
    showTraps = !showTraps;
    const btn = document.getElementById("btnToggleTraps");
    if (btn) btn.classList.toggle("active", showTraps);

    const legContainer = document.getElementById("trapsLegendDynamic");
    if (legContainer) {
        legContainer.style.display = showTraps ? "block" : "none";
    }

    if (showTraps) {
        loadTrapsLayer();
    } else {
        if (trapsLayer) {
            map.removeLayer(trapsLayer);
            trapsLayer = null;
        }
    }
}

async function loadTrapsLayer() {
    if (!currentPairId) return;

    try {
        const resp = await fetch(`/api/v1/pairs/${currentPairId}/traps`);
        if (!resp.ok) return;
        const data = await resp.json();

        if (trapsLayer) {
            map.removeLayer(trapsLayer);
        }

        // Render dynamic legend items matching exactly the current AOI traps
        renderTrapsLegend(data.features || []);

        trapsLayer = L.geoJSON(data, {
            style: (feature) => {
                const color = feature.properties.color || "#a855f7";
                return {
                    color: color,
                    weight: 2,
                    fillColor: color,
                    fillOpacity: 0.45,
                };
            },
            onEachFeature: (feature, layer) => {
                const p = feature.properties;
                layer.bindPopup(`
                    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#0f172a;min-width:220px;">
                        <strong style="color:${p.color};">${p.name}</strong><br>
                        <div style="margin:4px 0;font-size:11px;color:#475569;">
                            Правило: <code style="background:#f1f5f9;padding:1px 4px;border-radius:3px;">${p.filter_rule}</code>
                        </div>
                        <div style="font-size:12px;font-weight:600;color:#0f172a;">
                            Статус: ${p.filter_action}
                        </div>
                        <div style="font-size:12px;font-weight:700;color:#16a34a;margin-top:2px;">
                            Предотвращено ложных тревог: ${p.suppressed_ha} га
                        </div>
                    </div>
                `);
            }
        }).addTo(map);

    } catch (err) {
        console.error("Failed to load traps layer:", err);
    }
}

function renderTrapsLegend(features) {
    let container = document.getElementById("trapsLegendDynamic");
    if (!container) {
        container = document.createElement("div");
        container.id = "trapsLegendDynamic";
        const mapLegend = document.querySelector(".map-legend");
        if (mapLegend) mapLegend.appendChild(container);
    }
    if (!showTraps || features.length === 0) {
        container.innerHTML = "";
        return;
    }
    let html = "";
    features.forEach(f => {
        const p = f.properties;
        const shortName = (p.name.includes(':') ? p.name.split(':')[1] : p.name).trim();
        html += `
            <div class="legend-item">
                <span class="color-box" style="background:${p.color};"></span>
                <span>${shortName} (${p.suppressed_ha} га)</span>
            </div>
        `;
    });
    container.innerHTML = html;
}

// ==========================================
// 6. TIMELAPSE PLAYER (TEMPORAL FLOOD WAVE)
// ==========================================
let timelapseActive = false;
let timelapseData = null;
let timelapseStep = 0;
let timelapseTimer = null;
let timelapseSpeed = 1;
let timelapseLayer = null;

async function toggleTimelapse() {
    timelapseActive = !timelapseActive;
    const dock = document.getElementById("timelapseDock");
    const btn = document.getElementById("btnToggleTimelapse");
    if (btn) btn.classList.toggle("active", timelapseActive);

    if (!timelapseActive) {
        if (dock) dock.style.display = "none";
        pauseTimelapse();
        if (timelapseLayer) {
            map.removeLayer(timelapseLayer);
            timelapseLayer = null;
        }
        // Restore standard static pre and flood layers
        if (preLayer && !map.hasLayer(preLayer)) map.addLayer(preLayer);
        if (floodLayer && !map.hasLayer(floodLayer)) map.addLayer(floodLayer);
        setLayerMode(currentLayerMode);
        autoCollapseLegendForDrawer(false);
        return;
    }

    // Mutual exclusion: Close other map tools
    if (typeof isProfileOpen !== "undefined" && isProfileOpen) {
        toggleProfileDrawer();
    }
    if (typeof evacuationActive !== "undefined" && evacuationActive) {
        toggleEvacuationLayer();
    }

    // Hide static pre and flood layers so dynamic timelapse animation is crystal clear!
    if (preLayer && map.hasLayer(preLayer)) map.removeLayer(preLayer);
    if (floodLayer && map.hasLayer(floodLayer)) map.removeLayer(floodLayer);

    if (dock) dock.style.display = "flex";
    autoCollapseLegendForDrawer(true);
    await loadTimelapseData();
}

async function loadTimelapseData() {
    if (!currentPairId) return;
    try {
        const resp = await fetch(`/api/v1/pairs/${currentPairId}/timelapse`);
        if (!resp.ok) return;
        timelapseData = await resp.json();
        timelapseStep = 0;
        renderTimelapseStep(0);
        if (timelapseLayer) {
            const b = timelapseLayer.getBounds();
            if (b.isValid()) {
                map.fitBounds(b, { padding: [50, 50], maxZoom: 12 });
            }
        }
    } catch (err) {
        console.error("Failed to load timelapse:", err);
    }
}

function renderTimelapseStep(step) {
    if (!timelapseData || !timelapseData.frames || !timelapseData.frames[step]) return;
    const f = timelapseData.frames[step];
    timelapseStep = step;

    document.getElementById("tlPhaseName").textContent = f.phase_name;
    document.getElementById("tlPhaseDesc").textContent = f.description;
    document.getElementById("tlDate").textContent = f.date;
    document.getElementById("tlWaterHa").textContent = `Вода: ${f.total_water_ha.toLocaleString('ru-RU')} га`;
    document.getElementById("tlStageCm").textContent = `ΔH: +${f.stage_delta_cm} см`;
    
    const rBadge = document.getElementById("tlRisk");
    rBadge.textContent = f.risk_level;
    rBadge.style.color = f.risk_color;

    const slider = document.getElementById("tlSlider");
    if (slider) slider.value = step;

    if (timelapseLayer) {
        map.removeLayer(timelapseLayer);
    }

    timelapseLayer = L.geoJSON(f.feature_collection, {
        style: (feature) => {
            const isRiver = feature.properties.layer === "river_pre";
            return {
                color: isRiver ? "#0284c7" : (f.risk_color || "#ef4444"),
                weight: isRiver ? 2 : 2.5,
                fillColor: isRiver ? "#0284c7" : (f.risk_color || "#ef4444"),
                fillOpacity: isRiver ? 0.75 : 0.65,
            };
        },
        onEachFeature: (feature, layer) => {
            const p = feature.properties;
            layer.bindPopup(`
                <div style="font-family: sans-serif; font-size: 13px;">
                    <strong>${p.name || 'Гидрологический контур'}</strong><br/>
                    Фаза: <strong>${f.phase_name} (${f.date})</strong><br/>
                    Площадь: <strong>${p.area_ha} га</strong><br/>
                    Подъем уровня: <strong>+${f.stage_delta_cm} см</strong>
                </div>
            `);
        }
    }).addTo(map);
}

function togglePlayTimelapse() {
    if (timelapseTimer) {
        pauseTimelapse();
    } else {
        playTimelapse();
    }
}

function playTimelapse() {
    if (!timelapseData || !timelapseData.frames) return;

    // If already at or beyond the last frame, restart smoothly from the beginning
    if (timelapseStep >= timelapseData.frame_count - 1) {
        renderTimelapseStep(0);
    }

    const btn = document.getElementById("btnTlPlay");
    if (btn) btn.textContent = "⏸ Пауза";

    const interval = Math.max(300, Math.round(1500 / timelapseSpeed));
    timelapseTimer = setInterval(() => {
        if (!timelapseData) return;
        const nextStep = timelapseStep + 1;
        if (nextStep >= timelapseData.frame_count) {
            // Reached final phase: stop single-pass playback
            pauseTimelapse();
            return;
        }
        renderTimelapseStep(nextStep);
    }, interval);
}

function pauseTimelapse() {
    if (timelapseTimer) {
        clearInterval(timelapseTimer);
        timelapseTimer = null;
    }
    const btn = document.getElementById("btnTlPlay");
    if (btn) btn.textContent = "▶ Старт";
}

function stepTimelapse(delta) {
    pauseTimelapse();
    if (!timelapseData) return;
    let nextStep = (timelapseStep + delta + timelapseData.frame_count) % timelapseData.frame_count;
    renderTimelapseStep(nextStep);
}

function onTimelapseSliderChange(val) {
    pauseTimelapse();
    renderTimelapseStep(parseInt(val, 10));
}

function setTimelapseSpeed(speed, btnElem) {
    timelapseSpeed = speed;
    document.querySelectorAll(".timelapse-speed .btn-speed").forEach(b => b.classList.remove("active"));
    if (btnElem) btnElem.classList.add("active");
    if (timelapseTimer) {
        pauseTimelapse();
        playTimelapse();
    }
}

// ==========================================
// 7. EVACUATION & LOGISTICS ALERT LAYER
// ==========================================
let evacuationActive = false;
let evacuationLayerGroup = null;

async function toggleEvacuationLayer() {
    evacuationActive = !evacuationActive;
    const btn = document.getElementById("btnToggleEvac");
    if (btn) btn.classList.toggle("active", evacuationActive);

    const drawer = document.getElementById("evacuationDrawer");
    const legRoad = document.getElementById("legEvacRoad");
    const legIsolate = document.getElementById("legEvacIsolate");
    const legPvr = document.getElementById("legEvacPvr");

    if (!evacuationActive) {
        if (drawer) drawer.style.display = "none";
        if (legRoad) legRoad.style.display = "none";
        if (legIsolate) legIsolate.style.display = "none";
        if (legPvr) legPvr.style.display = "none";
        if (evacuationLayerGroup) {
            map.removeLayer(evacuationLayerGroup);
            evacuationLayerGroup = null;
        }
        autoCollapseLegendForDrawer(false);
        return;
    }

    // Mutual exclusion: Close other map tools
    if (typeof isProfileOpen !== "undefined" && isProfileOpen) {
        toggleProfileDrawer();
    }
    if (typeof timelapseActive !== "undefined" && timelapseActive) {
        toggleTimelapse();
    }

    if (drawer) drawer.style.display = "flex";
    if (legRoad) legRoad.style.display = "flex";
    if (legIsolate) legIsolate.style.display = "flex";
    if (legPvr) legPvr.style.display = "flex";

    autoCollapseLegendForDrawer(true);
    await loadEvacuationData();
}

async function loadEvacuationData() {
    if (!currentPairId) return;
    try {
        const resp = await fetch(`/api/v1/pairs/${currentPairId}/evacuation`);
        if (!resp.ok) return;
        const data = await resp.json();

        if (evacuationLayerGroup) {
            map.removeLayer(evacuationLayerGroup);
        }

        const layers = [];

        // 1. Cut-off roads
        data.cut_off_roads.forEach(r => {
            const polyline = L.polyline(r.line_coords.map(c => [c[1], c[0]]), {
                color: "#ef4444",
                weight: 4,
                dashArray: "6, 6",
            }).bindPopup(`
                <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#0f172a;min-width:220px;">
                    <strong style="color:#ef4444;">⛔ Перелив автодороги</strong><br>
                    <div style="font-weight:600;margin:3px 0;">${r.name}</div>
                    <div style="font-size:11px;color:#475569;">Категория: ${r.category}</div>
                    <div style="font-size:12px;color:#dc2626;font-weight:700;">Глубина перелива: ${r.depth_m} м (длина: ${r.length_m} м)</div>
                    <div style="font-size:11px;color:#b91c1c;font-weight:600;margin-top:2px;">${r.status}</div>
                </div>
            `);
            layers.push(polyline);
        });

        // 2. Isolated communities
        data.isolated_communities.forEach(c => {
            const iconHtml = `<div style="background:#f97316;width:24px;height:24px;border-radius:50%;border:2px solid #fff;display:flex;align-items:center;justify-content:center;font-size:12px;box-shadow:0 0 8px rgba(0,0,0,0.8);">⚠️</div>`;
            const icon = L.divIcon({ html: iconHtml, className: 'evac-pin', iconSize: [24, 24], iconAnchor: [12, 12] });
            const marker = L.marker([c.point[1], c.point[0]], { icon: icon }).bindPopup(`
                <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#0f172a;min-width:220px;">
                    <strong style="color:#ea580c;">Отрезанный поселок / СНТ</strong><br>
                    <div style="font-weight:700;margin:3px 0;">${c.name}</div>
                    <div style="font-size:12px;color:#dc2626;font-weight:600;">Население под угрозой: ${c.population_at_risk} чел.</div>
                    <div style="font-size:11px;color:#475569;">Статус: ${c.status}</div>
                    <div style="font-size:11px;color:#0284c7;font-weight:600;margin-top:3px;">Требуемый транспорт: ${c.required_transport}</div>
                </div>
            `);
            layers.push(marker);
        });

        // 3. Safe zones (PVR & helipads)
        data.safe_zones.forEach(z => {
            const iconHtml = `<div style="background:#22c55e;width:24px;height:24px;border-radius:50%;border:2px solid #fff;display:flex;align-items:center;justify-content:center;font-size:12px;box-shadow:0 0 8px rgba(0,0,0,0.8);">🛡️</div>`;
            const icon = L.divIcon({ html: iconHtml, className: 'pvr-pin', iconSize: [24, 24], iconAnchor: [12, 12] });
            const marker = L.marker([z.point[1], z.point[0]], { icon: icon }).bindPopup(`
                <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#0f172a;min-width:220px;">
                    <strong style="color:#16a34a;">${z.type}</strong><br>
                    <div style="font-weight:700;margin:3px 0;">${z.name}</div>
                    ${z.capacity_people > 0 ? `<div style="font-size:12px;color:#16a34a;font-weight:600;">Вместимость: ${z.capacity_people} мест</div>` : ''}
                    <div style="font-size:11px;color:#475569;">Высота над водой (HAND): ${z.hand_m} м (отметка ${z.elevation_bs_m} м БС)</div>
                    <div style="font-size:11px;color:#16a34a;font-weight:600;margin-top:2px;">Статус: ${z.status}</div>
                </div>
            `);
            layers.push(marker);
        });

        // 4. Safe corridors
        data.safe_corridors.forEach(cor => {
            const line = L.polyline(cor.line_coords.map(c => [c[1], c[0]]), {
                color: "#22c55e",
                weight: 3,
                dashArray: "4, 4",
            }).bindPopup(`<strong>Маршрут эвакуации:</strong> ${cor.name}`);
            layers.push(line);
        });

        evacuationLayerGroup = L.featureGroup(layers).addTo(map);

        // Populate drawer content
        const s = data.summary;
        const drawerBody = document.getElementById("evacContent");
        if (drawerBody) {
            drawerBody.innerHTML = `
                <div style="display:flex;gap:6px;margin-bottom:8px;">
                    <div class="metric-card" style="flex:1;">
                        <div class="metric-title">Перерезано дорог</div>
                        <div class="metric-value text-red">${s.total_cut_km} км</div>
                    </div>
                    <div class="metric-card" style="flex:1;">
                        <div class="metric-title">Людей в изоляции</div>
                        <div class="metric-value text-orange">${s.population_at_risk_total} чел.</div>
                    </div>
                </div>

                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #334155; border-radius: 6px; padding: 7px 10px; margin-bottom: 8px; font-size: 11px; display: flex; flex-direction: column; gap: 4px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="display: inline-block; width: 22px; height: 0; border-top: 3px dashed #ef4444;"></span>
                        <span style="color: #fca5a5;">Красный пунктир: участок перелива автодороги (затоплен)</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="display: inline-block; width: 22px; height: 0; border-top: 3px dashed #22c55e;"></span>
                        <span style="color: #86efac;">Зеленый пунктир: коридор эвакуации (путь по террасе к ПВР)</span>
                    </div>
                </div>

                <div class="evac-section-title">Перерезанные участки автодорог (${data.cut_off_roads.length})</div>
                ${data.cut_off_roads.map(r => `
                    <div class="evac-item-card road">
                        <div class="evac-item-title">${r.name}</div>
                        <div class="evac-item-desc">Глубина перелива: <strong style="color:#ef4444;">${r.depth_m} м</strong> | Протяженность: ${r.length_m} м</div>
                        <div style="color:#f87171;font-size:10px;font-weight:600;margin-top:2px;">${r.status}</div>
                    </div>
                `).join('')}

                <div class="evac-section-title">Отрезанные селения и СНТ (${data.isolated_communities.length})</div>
                ${data.isolated_communities.map(c => `
                    <div class="evac-item-card isolate">
                        <div class="evac-item-title">${c.name}</div>
                        <div class="evac-item-desc">Население: <strong style="color:#ea580c;">${c.population_at_risk} чел.</strong> | ${c.isolation_type}</div>
                        <div style="color:#fdba74;font-size:10px;margin-top:2px;">Транспорт: ${c.required_transport}</div>
                    </div>
                `).join('')}

                <div class="evac-section-title">Безопасные ПВР и авиаплощадки (${data.safe_zones.length})</div>
                ${data.safe_zones.map(z => `
                    <div class="evac-item-card pvr">
                        <div class="evac-item-title">${z.name}</div>
                        <div class="evac-item-desc">${z.capacity_people > 0 ? `Вместимость: ${z.capacity_people} мест | ` : ''}HAND: ${z.hand_m} м (отметка ${z.elevation_bs_m} м БС)</div>
                        <div style="color:#86efac;font-size:10px;font-weight:600;margin-top:2px;">Статус: ${z.status}</div>
                    </div>
                `).join('')}

                <div style="background:#0f172a;border:1px solid #475569;border-radius:6px;padding:8px;margin-top:6px;font-size:11px;line-height:1.3;color:#e2e8f0;">
                    <strong style="color:#38bdf8;">Директива оперативному штабу:</strong> ${s.operational_directive}
                </div>
            `;
        }

    } catch (err) {
        console.error("Failed to load evacuation data:", err);
    }
}

// ==========================================
// 8. ABLATION STUDY MODAL & ZERO-INTERNET BASMAP
// ==========================================
async function openAblationModal() {
    const modal = document.getElementById("ablationModal");
    const body = document.getElementById("ablationBody");
    if (modal) modal.style.display = "flex";

    try {
        const resp = await fetch("/api/v1/ablation");
        if (!resp.ok) throw new Error("Failed to load ablation study");
        const data = await resp.json();

        body.innerHTML = `
            <div style="display:flex;justify-content:space-between;align-items:center;background:#1e293b;border-radius:8px;padding:12px 16px;margin-bottom:12px;">
                <div>
                    <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;">Интегральный соревновательный Score</div>
                    <div style="font-size:24px;font-weight:800;color:#4ade80;font-family:Consolas,monospace;">${data.final_score} <span style="font-size:13px;color:#94a3b8;font-weight:400;">(целевой: &gt;0.85)</span></div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;">Суммарный прирост от бейзлайна</div>
                    <div style="font-size:20px;font-weight:800;color:#38bdf8;font-family:Consolas,monospace;">+${data.total_gain} (+${(data.total_gain / 0.59182 * 100).toFixed(1)}%)</div>
                </div>
            </div>

            <div class="ablation-table-wrap">
                <table class="ablation-table">
                    <thead>
                        <tr>
                            <th style="width:28%;">Этап / Архитектурный модуль</th>
                            <th style="width:18%;">Score (Регламент)</th>
                            <th style="width:12%;">Δ Score</th>
                            <th style="width:10%;">Q flood</th>
                            <th style="width:10%;">Spec base</th>
                            <th style="width:22%;">Вклад в надежность</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.experiments.map(e => {
                            const isFinal = e.step === 6;
                            const scorePct = Math.round(e.score * 100);
                            return `
                                <tr class="${isFinal ? 'final-row' : ''}">
                                    <td>
                                        <strong style="color:${isFinal ? '#4ade80' : '#f8fafc'};">${e.name}</strong>
                                        <div style="font-size:11px;color:#94a3b8;margin-top:2px;">${e.description}</div>
                                    </td>
                                    <td>
                                        <div style="font-family:Consolas,monospace;font-weight:700;font-size:13px;color:${isFinal ? '#4ade80' : '#38bdf8'};">
                                            ${e.score.toFixed(5)}
                                        </div>
                                        <div class="ablation-score-bar-bg">
                                            <div class="ablation-score-bar-fill ${isFinal ? 'final' : ''}" style="width:${scorePct}%;"></div>
                                        </div>
                                    </td>
                                    <td>
                                        ${e.delta_score > 0 ? 
                                            `<span class="delta-badge pos">+${e.delta_score.toFixed(4)}</span>` : 
                                            `<span class="delta-badge neutral">0.0000</span>`
                                        }
                                    </td>
                                    <td style="font-family:Consolas,monospace;">${e.q_flood.toFixed(3)}</td>
                                    <td style="font-family:Consolas,monospace;">${e.spec_base.toFixed(3)}</td>
                                    <td style="font-size:11px;color:#cbd5e1;">${e.contribution}</td>
                                </tr>
                            `;
                        }).join('')}
                    </tbody>
                </table>
            </div>
            
            <div style="margin-top:12px;padding:10px 14px;background:#090d16;border:1px solid #334155;border-radius:6px;font-size:11px;color:#94a3b8;line-height:1.4;">
                <strong style="color:#e2e8f0;">Вывод для экспертной комиссии:</strong> Финальный конвейер объединяет физическое моделирование (подавление 4 доменных ловушек) с глубоким обучением UNet и топологической постобработкой, обеспечивая максимальное качество детекции паводка (<code style="color:#38bdf8;">0.98248</code>), безупречную специфичность на межени (<code style="color:#4ade80;">1.00000</code>) и нулевое расхождение растра и таблицы (<code style="color:#a855f7;">0.0000%</code>).
            </div>
        `;
    } catch (err) {
        body.innerHTML = `<div style="color:#ef4444;font-size:13px;">Ошибка загрузки матрицы абляций: ${err.message}</div>`;
    }
}

function closeAblationModal() {
    const modal = document.getElementById("ablationModal");
    if (modal) modal.style.display = "none";
}

function onModalOverlayClick(e) {
    if (e.target && e.target.id === "ablationModal") {
        closeAblationModal();
    }
}

function toggleOfflineBasemap() {
    isOfflineBasemap = !isOfflineBasemap;
    const btn = document.getElementById("btnToggleOffline");
    const mapEl = document.getElementById("map");

    if (btn) btn.classList.toggle("active", isOfflineBasemap);

    if (isOfflineBasemap) {
        if (currentBaseTileLayer && map.hasLayer(currentBaseTileLayer)) {
            map.removeLayer(currentBaseTileLayer);
        }
        if (mapEl) mapEl.classList.add("offline-grid-pattern");
        if (btn) btn.textContent = "📴 Офлайн: Активен";
    } else {
        if (mapEl) mapEl.classList.remove("offline-grid-pattern");
        if (currentBaseTileLayer) {
            currentBaseTileLayer.addTo(map);
        }
        if (btn) btn.textContent = "📴 Офлайн-карта";
    }
}

// ==========================================
// 10. CLICK-TO-INSPECT POINT HYDROLOGY
// ==========================================
let inspectPopup = null;

async function handlePointInspect(latlng, originalEvent) {
    if (typeof isDraggingSwipe !== "undefined" && isDraggingSwipe) return;
    if (originalEvent && originalEvent.target) {
        if (originalEvent.target.closest('.leaflet-control, .swipe-divider, .drawer, .dock, .modal, .threat-pulse-marker, .gauge-marker-wrap, .evac-pin')) {
            return;
        }
    }
    if (!currentPairId) return;

    const { lat, lng } = latlng;

    const loadingHtml = `
        <div class="inspect-card">
            <div class="inspect-header">
                <span class="inspect-title">🎯 Инспектор точки</span>
                <span class="inspect-coords">${lat.toFixed(4)}°N, ${lng.toFixed(4)}°E</span>
            </div>
            <div style="font-size:11px;color:#94a3b8;padding:8px 0;display:flex;align-items:center;gap:6px;">
                <span>⏳ Запрос данных Copernicus DEM и масок Sentinel...</span>
            </div>
        </div>
    `;

    inspectPopup = L.popup({ className: 'custom-inspect-popup', maxWidth: 320 })
        .setLatLng([lat, lng])
        .setContent(loadingHtml)
        .openOn(map);

    try {
        const resp = await fetch(`/api/v1/pairs/${currentPairId}/inspect?lat=${lat}&lon=${lng}`);
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        const d = await resp.json();

        const latVal = (d.coords && d.coords.lat) || d.lat || lat;
        const lonVal = (d.coords && d.coords.lon) || d.lon || lng;
        const elevVal = d.elevation_bs_m ?? d.elevation_dem_m ?? 0;
        const depthVal = d.water_depth_m ?? d.depth_m ?? 0;
        const statusVal = d.zone_status || d.water_status_ru || "Суша";
        const colorVal = d.color || d.water_status_color || "#10b981";
        const lcVal = d.landcover || d.worldcover_class || "Естественный покров";
        const distVal = d.distance_to_channel_m ?? d.distance_to_river_m ?? 0;
        const datumVal = d.datum_m ?? (elevVal - depthVal);
        const riskVal = d.risk_level || (depthVal > 0 ? "ВЫСОКИЙ (Затопление)" : "БЕЗОПАСНО (Суша)");

        const cardHtml = `
            <div class="inspect-card">
                <div class="inspect-header">
                    <span class="inspect-title">🎯 Инспектор точки</span>
                    <div style="display:flex;align-items:center;gap:5px;">
                        <span class="inspect-coords">${latVal.toFixed(4)}°N, ${lonVal.toFixed(4)}°E</span>
                        <button class="btn-copy-coords" onclick="copyInspectCoords(${latVal.toFixed(6)}, ${lonVal.toFixed(6)}, this)" title="Скопировать точные координаты">📋</button>
                    </div>
                </div>
                <div class="inspect-status-badge" style="background: ${colorVal}22; color: ${colorVal}; border: 1px solid ${colorVal};">
                    <span>●</span>
                    <span>${statusVal}</span>
                </div>
                <div class="inspect-grid">
                    <div class="inspect-item">
                        <div class="inspect-item-label">Отметка DEM (БС)</div>
                        <div class="inspect-item-val">${elevVal} м</div>
                    </div>
                    <div class="inspect-item">
                        <div class="inspect-item-label">Слой воды H</div>
                        <div class="inspect-item-val" style="color: ${depthVal > 0 ? '#ef4444' : '#10b981'};">
                            ${depthVal > 0 ? depthVal.toFixed(1) + ' м' : '0.0 м'}
                        </div>
                    </div>
                    <div class="inspect-item">
                        <div class="inspect-item-label">Уровень русла реки</div>
                        <div class="inspect-item-val">${datumVal} м</div>
                    </div>
                    <div class="inspect-item">
                        <div class="inspect-item-label">Дистанция до русла</div>
                        <div class="inspect-item-val">${Math.round(distVal)} м</div>
                    </div>
                </div>
                <div class="inspect-footer">
                    <strong>Покров (ESA):</strong> ${lcVal}<br>
                    <strong>Оценка риска:</strong> <span style="color:${colorVal};font-weight:600;">${riskVal}</span>
                </div>
            </div>
        `;
        inspectPopup.setContent(cardHtml);
    } catch (err) {
        inspectPopup.setContent(`
            <div class="inspect-card">
                <div class="inspect-header">
                    <span class="inspect-title" style="color:#ef4444;">⚠️ Ошибка инспектора</span>
                </div>
                <div style="font-size:11px;color:#ef4444;">Не удалось получить параметры точки: ${err.message}</div>
            </div>
        `);
    }
}

function copyInspectCoords(lat, lon, btnElem) {
    const text = `${lat.toFixed(6)}, ${lon.toFixed(6)}`;
    const onDone = () => {
        if (btnElem) {
            btnElem.textContent = "✓";
            btnElem.title = "Скопировано!";
            btnElem.classList.add("copied");
            setTimeout(() => {
                btnElem.textContent = "📋";
                btnElem.title = "Скопировать точные координаты";
                btnElem.classList.remove("copied");
            }, 1800);
        }
    };

    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(onDone).catch(() => fallbackCopyCoords(text, onDone));
    } else {
        fallbackCopyCoords(text, onDone);
    }
}

function fallbackCopyCoords(text, cb) {
    const input = document.createElement("input");
    input.value = text;
    document.body.appendChild(input);
    input.select();
    try {
        document.execCommand("copy");
        if (cb) cb();
    } catch (e) {
        console.warn("Clipboard copy failed:", e);
    }
    document.body.removeChild(input);
}

function initPointInspector() {
    if (!map) return;
    map.on('click', (e) => {
        handlePointInspect(e.latlng, e.originalEvent);
    });
}

// ==========================================
// 11. THREAT QUICK-ZOOM & PULSE FOCUS
// ==========================================
let threatPulseMarker = null;

const THREAT_LOCATIONS = {
    "blagoveshchensk": {
        "flood": { lat: 50.2520, lon: 127.6150, zoom: 12, label: "Зона активного паводка: слияние рек Амур и Зея" },
        "roads": { lat: 50.3150, lon: 127.5620, zoom: 13, label: "Участок подтопления: автодорога Благовещенск — Свободный" },
        "settlement": { lat: 50.2980, lon: 127.6320, zoom: 13, label: "Угроза жилому сектору: с. Владимировка / Зазейский" }
    },
    "svobodny": {
        "flood": { lat: 51.3650, lon: 128.1420, zoom: 12, label: "Зона разлива реки Зея в черте г. Свободный" },
        "roads": { lat: 51.3920, lon: 128.0950, zoom: 13, label: "Подтопление подъездных путей к микрорайону Суражевка" },
        "settlement": { lat: 51.3520, lon: 128.1180, zoom: 13, label: "Зона риска: прибрежные жилые кварталы г. Свободный" }
    },
    "konstantinovka": {
        "flood": { lat: 49.6100, lon: 128.0200, zoom: 12, label: "Зона затопления пойменных угодий р. Амур" },
        "roads": { lat: 49.6350, lon: 127.9750, zoom: 13, label: "Перелив региональной автотрассы 10К-044" },
        "settlement": { lat: 49.6200, lon: 127.9900, zoom: 13, label: "Ближайшая жилая застройка: с. Константиновка" }
    },
    "belogorsk": {
        "flood": { lat: 50.9120, lon: 128.4850, zoom: 12, label: "Зона затопления долины реки Томь" },
        "roads": { lat: 50.9320, lon: 128.4550, zoom: 13, label: "Угроза перелива: трасса подъезда к г. Белогорск" },
        "settlement": { lat: 50.9050, lon: 128.4720, zoom: 13, label: "Угроза подтопления: частный сектор г. Белогорск" }
    },
    "poyarkovo": {
        "flood": { lat: 49.6150, lon: 128.6650, zoom: 12, label: "Паводок р. Амур в устье реки Завитая" },
        "roads": { lat: 49.6450, lon: 128.6350, zoom: 13, label: "Подтопление подъездной дороги к портовому узлу Поярково" },
        "settlement": { lat: 49.6300, lon: 128.6500, zoom: 13, label: "Угроза подтопления: пгт Поярково" }
    }
};

function focusThreatTarget(type) {
    if (!currentPairId) return;

    let target = null;
    for (const [aoiKey, threats] of Object.entries(THREAT_LOCATIONS)) {
        if (currentPairId.includes(aoiKey)) {
            target = threats[type];
            break;
        }
    }

    if (!target) {
        for (const [aoiKey, coords] of Object.entries(AOI_CENTERS)) {
            if (currentPairId.includes(aoiKey)) {
                target = { lat: coords[0], lon: coords[1], zoom: 12, label: "Зона мониторинга паводка" };
                break;
            }
        }
    }

    if (!target) return;

    // For road and settlement threats, activate evacuation layer if inactive
    if ((type === 'roads' || type === 'settlement') && typeof evacuationActive !== "undefined" && !evacuationActive) {
        toggleEvacuationLayer();
    }

    map.flyTo([target.lat, target.lon], target.zoom, {
        animate: true,
        duration: 1.2
    });

    const pulseIcon = L.divIcon({
        className: 'threat-pulse-marker',
        html: '<div class="threat-pulse-ring"></div><div class="threat-pulse-dot"></div>',
        iconSize: [42, 42],
        iconAnchor: [21, 21]
    });

    if (threatPulseMarker) {
        map.removeLayer(threatPulseMarker);
    }

    threatPulseMarker = L.marker([target.lat, target.lon], { icon: pulseIcon, zIndexOffset: 1000 }).addTo(map);
    threatPulseMarker.bindTooltip(`<strong>🎯 ${target.label}</strong>`, {
        permanent: true,
        direction: 'top',
        offset: [0, -18]
    }).openTooltip();

    setTimeout(() => {
        if (threatPulseMarker) {
            map.removeLayer(threatPulseMarker);
            threatPulseMarker = null;
        }
    }, 7000);
}

// ==========================================
// 12. OPERATIONAL KEYBOARD SHORTCUTS
// ==========================================
function initKeyboardShortcuts() {
    window.addEventListener("keydown", (e) => {
        const tag = (e.target && e.target.tagName) || "";
        if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || (e.target && e.target.isContentEditable)) {
            return;
        }

        if (e.key === "Tab") {
            e.preventDefault();
            toggleSidebarCollapse();
        } else if (e.key === "h" || e.key === "H" || e.key === "р" || e.key === "Р") {
            fitCurrentAoiBounds();
        } else if (e.key === "1") {
            switchSidebarTab("overview");
        } else if (e.key === "2") {
            switchSidebarTab("forecast");
        } else if (e.key === "s" || e.key === "S" || e.key === "ы" || e.key === "Ы") {
            toggleSwipeMode();
        } else if (e.key === " ") {
            e.preventDefault();
            if (typeof timelapseActive !== "undefined" && timelapseActive) {
                togglePlayTimelapse();
            } else if (typeof toggleTimelapse === "function") {
                toggleTimelapse();
            }
        } else if (e.key === "Escape") {
            const sidebar = document.querySelector(".sidebar");
            if (sidebar && sidebar.classList.contains("collapsed")) {
                toggleSidebarCollapse();
            }
            if (typeof isProfileOpen !== "undefined" && isProfileOpen) {
                toggleProfileDrawer();
            }
            if (typeof timelapseActive !== "undefined" && timelapseActive) {
                toggleTimelapse();
            }
            if (typeof evacuationActive !== "undefined" && evacuationActive) {
                toggleEvacuationLayer();
            }
            closeAblationModal();
            closeExportMenu();
            if (map) map.closePopup();
        }
    });
}

// ==========================================
// 13. SIDEBAR ZEN COLLAPSE & MAP BOUNDS
// ==========================================
function toggleSidebarCollapse() {
    const sidebar = document.querySelector(".sidebar");
    const btnExpand = document.getElementById("btnExpandSidebar");
    if (!sidebar) return;

    const isCollapsed = sidebar.classList.toggle("collapsed");
    if (btnExpand) {
        btnExpand.style.display = isCollapsed ? "flex" : "none";
    }

    setTimeout(() => {
        if (map) map.invalidateSize();
    }, 300);
}

let legendAutoCollapsed = false;

function toggleLegendCollapse() {
    const leg = document.getElementById("mapLegend");
    if (!leg) return;
    leg.classList.toggle("collapsed");
    legendAutoCollapsed = false; // User manually chose the state
}

function autoCollapseLegendForDrawer(open) {
    const leg = document.getElementById("mapLegend");
    if (!leg) return;
    if (open) {
        if (!leg.classList.contains("collapsed")) {
            leg.classList.add("collapsed");
            legendAutoCollapsed = true;
        }
    } else {
        const anyDrawerOpen = (typeof isProfileOpen !== "undefined" && isProfileOpen) ||
                              (typeof timelapseActive !== "undefined" && timelapseActive) ||
                              (typeof evacuationActive !== "undefined" && evacuationActive);
        if (!anyDrawerOpen && legendAutoCollapsed) {
            leg.classList.remove("collapsed");
            legendAutoCollapsed = false;
        }
    }
}

function fitCurrentAoiBounds() {
    if (aoiLayer && aoiLayer.getBounds && aoiLayer.getBounds().isValid()) {
        map.fitBounds(aoiLayer.getBounds(), { padding: [30, 30] });
    } else if (currentAoiBounds && currentAoiBounds.isValid()) {
        map.fitBounds(currentAoiBounds, { padding: [30, 30] });
    } else if (currentPairId) {
        for (const [aoiKey, coords] of Object.entries(AOI_CENTERS)) {
            if (currentPairId.includes(aoiKey)) {
                map.setView(coords, 10);
                break;
            }
        }
    }
}




