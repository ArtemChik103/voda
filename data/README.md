# Каталог данных: Структура и соглашения

Данный каталог предназначен для хранения геопространственных растров, векторных слоев и таблиц метаданных в соответствии с требованиями регламента соревнований «КосмоХакатон 2026».

## Ожидаемая структура файлов

```
data/
├── vectors/                           # Векторные слои
│   ├── amur_region_boundary.geojson   # Границы Амурской области
│   ├── aoi_polygons.geojson           # Полигоны 11 районов интереса (AOI)
│   ├── rivers_osm.geojson             # Речная гидросеть OSM
│   └── hydrobasins_amur.geojson       # Водосборные бассейны HydroSHEDS
├── tables/                            # Таблицы метаданных и реестров сцен
│   ├── pairs_registry.csv             # Список 11 пар с датами и углами падения
│   └── scene_metadata.csv             # Метаданные сенсоров Sentinel-1 и Sentinel-2
├── rasters/                           # Спутниковые растры по событиям и AOI
│   └── <event_id>/<aoi>/
│       ├── S1_pre.tif                 # SAR до события (VV, VH, VV/VH в дБ, 10м)
│       ├── S1_peak.tif                # SAR на пике (VV, VH, VV/VH в дБ, 10м)
│       ├── SENTINEL2_pre.tif          # Оптика до (B3, B4, B8, B11, NDWI, MNDWI, NDVI, AWEIsh)
│       ├── SENTINEL2_peak.tif         # Оптика на пике
│       └── AUX_terrain_gsw.tif        # ЦМР/HAND (slope, hand, occurrence, seasonality, builtup)
├── reference_masks/                   # 5-канальные эталонные маски
│   └── <pair_id>_reference.tif
└── synthetic_benchmark/               # Синтетический бенчмарк для модульных тестов и CI
    ├── ground_truth.csv
    ├── sample_submission.csv
    └── reference_masks/
```

## Требования к растровым геоданным
- **Система координат:** EPSG:32652 (WGS 84 / UTM zone 52N).
- **Разрешение:** 10 м $\times$ 10 м на пиксель ($0.01$ га).
- **Тип данных масок:** `uint8` со значениями строго $\{0, 1\}$.
