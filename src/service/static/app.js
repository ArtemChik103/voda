// Web-GIS Dashboard Application Logic

let map;
let geojsonLayer = null;
let currentPairId = null;
let currentLayerMode = 'all';

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
    loadPairsList();

    document.getElementById("eventSelect").addEventListener("change", (e) => {
        onSelectPair(e.target.value);
    });
});

function initMap() {
    // Initial view on Amur Region, Blagoveshchensk
    map = L.map('map', {
        center: [50.28, 127.54],
        zoom: 10,
        zoomControl: true,
    });

    // 1. High-Resolution Satellite imagery (Esri World Imagery) - No API Key Needed!
    const esriSatellite = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics',
        maxZoom: 18
    });

    // 2. Pure Dark Hydrological Map (OSM with dark filter) - No API Key Needed!
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
    esriSatellite.addTo(map);

    const baseLayers = {
        "🛰️ Спутник (Esri Satellite)": esriSatellite,
        "🗺️ Тёмная карта (OSM Dark)": osmDark,
        "🗺️ Топографическая (OSM)": osmLight
    };

    L.control.layers(baseLayers, null, { position: 'topright' }).addTo(map);
}

async function loadPairsList() {
    try {
        const response = await fetch("/api/v1/pairs");
        const data = await response.json();
        const select = document.getElementById("eventSelect");
        select.innerHTML = "";

        data.pairs.forEach((p, idx) => {
            const opt = document.createElement("option");
            opt.value = p.pair_id;
            const eventBadge = p.event_type === "flood" ? "🌊 Паводок" : "🌾 Межень";
            opt.textContent = `[${eventBadge}] ${p.pair_id}`;
            select.appendChild(opt);
        });

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

    } catch (err) {
        console.error("Error loading pair data:", err);
    }
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

    // Land cover bars
    const barsContainer = document.getElementById("landcoverBars");
    barsContainer.innerHTML = "";

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

function renderGeoJSON(geoData, pairId) {
    if (geojsonLayer) {
        map.removeLayer(geojsonLayer);
    }

    geojsonLayer = L.geoJSON(geoData, {
        style: (feature) => {
            const type = feature.properties.type;
            if (type === 'flood') {
                return { color: '#ef4444', weight: 1.5, fillOpacity: 0.65, fillColor: '#ef4444' };
            } else if (type === 'water_peak') {
                return { color: '#60a5fa', weight: 1.0, fillOpacity: 0.45, fillColor: '#60a5fa' };
            } else {
                return { color: '#0284c7', weight: 1.0, fillOpacity: 0.50, fillColor: '#0284c7' };
            }
        },
        onEachFeature: (feature, layer) => {
            const p = feature.properties;
            layer.bindPopup(`
                <div style="font-family: sans-serif; font-size: 13px;">
                    <strong style="color: #0284c7;">${p.name || 'Водный объект'}</strong><br/>
                    Тип: <strong>${p.type === 'flood' ? 'Зона затопления' : 'Водное зеркало'}</strong><br/>
                    Площадь: <strong>${p.area_ha} га</strong>
                </div>
            `);
        }
    }).addTo(map);

    // Zoom to features bounds or AOI center
    const bounds = geojsonLayer.getBounds();
    if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [30, 30] });
    } else {
        for (const [aoiKey, coords] of Object.entries(AOI_CENTERS)) {
            if (pairId.includes(aoiKey)) {
                map.setView(coords, 10);
                break;
            }
        }
    }
}

function setLayerMode(mode) {
    currentLayerMode = mode;
    ['btnLayerAll', 'btnLayerFlood', 'btnLayerPre'].forEach(id => {
        document.getElementById(id).classList.remove('active');
    });

    if (mode === 'all') document.getElementById('btnLayerAll').classList.add('active');
    if (mode === 'flood') document.getElementById('btnLayerFlood').classList.add('active');
    if (mode === 'pre') document.getElementById('btnLayerPre').classList.add('active');

    if (!geojsonLayer) return;

    geojsonLayer.eachLayer(layer => {
        const type = layer.feature.properties.type;
        if (mode === 'all') {
            layer.setStyle({ opacity: 1.0, fillOpacity: 0.6 });
        } else if (mode === 'flood') {
            const isF = (type === 'flood');
            layer.setStyle({ opacity: isF ? 1.0 : 0.0, fillOpacity: isF ? 0.7 : 0.0 });
        } else if (mode === 'pre') {
            const isP = (type === 'water_pre');
            layer.setStyle({ opacity: isP ? 1.0 : 0.0, fillOpacity: isP ? 0.7 : 0.0 });
        }
    });
}

function updateSidebarLoading() {
    document.getElementById("valFloodHa").textContent = "Расчет...";
    document.getElementById("valFloodKm2").textContent = "Загрузка...";
    document.getElementById("valPeakHa").textContent = "Расчет...";
    document.getElementById("valPreHa").textContent = "Расчет...";
}

function exportGeoJSON() {
    if (!currentPairId) return;
    window.open(`/api/v1/pairs/${currentPairId}/geojson?download=true`, '_blank');
}

function openReport() {
    if (!currentPairId) return;
    window.open(`/api/v1/pairs/${currentPairId}/report/html`, '_blank');
}
