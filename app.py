from datetime import datetime
from pathlib import Path
import json
import re
import threading
import time
import traceback
import unicodedata
import uuid

import folium
from folium.plugins import Draw
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from owslib.wcs import WebCoverageService
from rasterio.crs import CRS
from rasterio.errors import RasterioIOError
from rasterio.features import shapes
from rasterio.io import MemoryFile
from rasterio.mask import mask as raster_mask
from rasterio.windows import from_bounds
from shapely.geometry import mapping, shape
import streamlit as st
from streamlit_folium import st_folium

from experimental_sentinel_wosis import (
    DEFAULT_TRAINING_SOURCE,
    TRAINING_SOURCE_OPTIONS,
    read_sentinel_wosis_fraction_rasters,
)


USDA_CLASSES = {
    1: "Arenosa (Sand)",
    2: "Arenosa-franca (Loamy sand)",
    3: "Franco-arenosa (Sandy loam)",
    4: "Franca (Loam)",
    5: "Franco-limosa (Silt loam)",
    6: "Limosa (Silt)",
    7: "Franco-areno-arcillosa (Sandy clay loam)",
    8: "Franco-arcillosa (Clay loam)",
    9: "Franco-limo-arcillosa (Silty clay loam)",
    10: "Areno-arcillosa (Sandy clay)",
    11: "Limo-arcillosa (Silty clay)",
    12: "Arcillosa (Clay)",
}

NODATA_CLASS = 0

USDA_COLORS = {
    1: "#d4c27b",
    2: "#c9bb85",
    3: "#b5a670",
    4: "#a39665",
    5: "#91875c",
    6: "#807753",
    7: "#cf8e55",
    8: "#ba804c",
    9: "#a67244",
    10: "#b85849",
    11: "#a65042",
    12: "#94473b",
    NODATA_CLASS: "#cccccc",
}

OPENLANDMAP_FRACTION_COGS = {
    "sand": {
        "label": "arena",
        "urls": {
            "mean": "https://s3.opengeohub.org/global-soil/global_soil_props_v20250523/sand.tot_iso.11277.2020.wpct_m_120m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250523.tif",
            "p16": "https://s3.opengeohub.org/global-soil/global_soil_props_v20250523/sand.tot_iso.11277.2020.wpct_p16_120m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250523.tif",
            "p84": "https://s3.opengeohub.org/global-soil/global_soil_props_v20250523/sand.tot_iso.11277.2020.wpct_p84_120m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250523.tif",
        },
    },
    "silt": {
        "label": "limo",
        "urls": {
            "mean": "https://s3.opengeohub.org/global-soil/global_soil_props_v20250523/silt.tot_iso.11277.2020.wpct_m_120m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250523.tif",
            "p16": "https://s3.opengeohub.org/global-soil/global_soil_props_v20250523/silt.tot_iso.11277.2020.wpct_p16_120m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250523.tif",
            "p84": "https://s3.opengeohub.org/global-soil/global_soil_props_v20250523/silt.tot_iso.11277.2020.wpct_p84_120m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250523.tif",
        },
    },
    "clay": {
        "label": "arcilla",
        "urls": {
            "mean": "https://s3.opengeohub.org/global-soil/global_soil_props_v20250523/clay.tot_iso.11277.2020.wpct_m_120m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250523.tif",
            "p16": "https://s3.opengeohub.org/global-soil/global_soil_props_v20250523/clay.tot_iso.11277.2020.wpct_p16_120m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250523.tif",
            "p84": "https://s3.opengeohub.org/global-soil/global_soil_props_v20250523/clay.tot_iso.11277.2020.wpct_p84_120m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250523.tif",
        },
    },
}

OPENLANDMAP_INTERVAL_CONFIDENCE_PERCENT = 68
OPENLANDMAP_MIN_CERTAINTY_PERCENT = 60
OPENLANDMAP_SOURCE_DESCRIPTION = (
    "OpenLandMap-soildb COGs: fracciones arena/limo/arcilla, media "
    "2020-2022, profundidad 0-30 cm, resolucion 120 m, EPSG:4326, "
    "con intervalos p0.16/p0.84 de 68% (>60%)."
)
OPENLANDMAP_CATALOG_URL = (
    "https://raw.githubusercontent.com/openlandmap/soildb/main/tables/"
    "OpenLandMap_soildb_COGS.csv"
)
SOILGRIDS_SOURCE_DESCRIPTION = (
    "SoilGrids250m 2.0 / ISRIC WCS: fracciones arena/limo/arcilla, "
    "promedio ponderado 0-30 cm desde 0-5, 5-15 y 15-30 cm, resolucion 250 m."
)
SOILGRIDS_CATALOG_URL = "https://docs.isric.org/globaldata/soilgrids/wcs.html"
SENTINEL_WOSIS_SOURCE_DESCRIPTION = (
    "Modelo experimental Sentinel-2 L2A + perfiles locales: entrena Random Forest "
    "con WoSIS/ISRIC y/o calicatas Costa Rica (arena/limo/arcilla 0-30 cm) y "
    "covariables multitemporales de suelo descubierto Sentinel-2 para predecir "
    "arena/limo/arcilla a 20 m."
)
SENTINEL_WOSIS_CATALOG_URL = (
    "https://docs.isric.org/globaldata/wosis/; "
    "https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a"
)
SOILGRIDS_CRS = "ESRI:54052"
SOILGRIDS_CRS_WKT = (
    'PROJCS["World_Goode_Homolosine_Land",'
    'GEOGCS["WGS 84",'
    'DATUM["World Geodetic System 1984",'
    'SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Interrupted_Goode_Homolosine"],'
    'PARAMETER["central_meridian",0],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1],'
    'AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["ESRI","54052"]]'
)
SOILGRIDS_WCS_CRS = "urn:ogc:def:crs:EPSG::152160"
SOILGRIDS_DEPTH_INTERVALS = [
    ("0-5cm", 5),
    ("5-15cm", 10),
    ("15-30cm", 15),
]
SOILGRIDS_LAYER_TEMPLATE = "https://maps.isric.org/mapserv?map=/map/{property}.map"
DATA_SOURCES = {
    "openlandmap": {
        "name": "OpenLandMap-soildb 120 m (PI 68%)",
        "description": OPENLANDMAP_SOURCE_DESCRIPTION,
        "catalog": OPENLANDMAP_CATALOG_URL,
        "layers": OPENLANDMAP_FRACTION_COGS,
        "resolution_label": "120 m",
        "resolution_slug": "120m",
        "certainty_threshold": OPENLANDMAP_MIN_CERTAINTY_PERCENT,
        "interval_confidence": OPENLANDMAP_INTERVAL_CONFIDENCE_PERCENT,
        "network_host": "s3.opengeohub.org",
    },
    "soilgrids": {
        "name": "SoilGrids250m / ISRIC WCS",
        "description": SOILGRIDS_SOURCE_DESCRIPTION,
        "catalog": SOILGRIDS_CATALOG_URL,
        "layers": {
            "sand": {"label": "arena", "coverage_prefix": "sand"},
            "silt": {"label": "limo", "coverage_prefix": "silt"},
            "clay": {"label": "arcilla", "coverage_prefix": "clay"},
        },
        "resolution_label": "250 m",
        "resolution_slug": "250m",
        "network_host": "maps.isric.org",
    },
    "sentinel_wosis": {
        "name": "Experimental Sentinel-2 + perfiles (20 m)",
        "description": SENTINEL_WOSIS_SOURCE_DESCRIPTION,
        "catalog": SENTINEL_WOSIS_CATALOG_URL,
        "layers": {
            "training": "WoSIS y/o calicatas Costa Rica sand/silt/clay 0-30 cm",
            "imagery": "Sentinel-2 L2A multitemporal bare-soil composite",
            "model": "RandomForestRegressor experimental",
        },
        "resolution_label": "20 m",
        "resolution_slug": "20m_sentinel_wosis",
        "network_host": "maps.isric.org y planetarycomputer.microsoft.com",
        "experimental": True,
    },
}
DEFAULT_SOURCE_KEY = "openlandmap"
OUTPUT_ROOT = Path("salidas")
SAVED_POLYGONS_DIR = OUTPUT_ROOT / "poligonos"
MAX_PIXELS = 2_500_000
SQM_PER_HA = 10_000.0
RECENT_JOB_LIMIT = 5
GDAL_HTTP_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "50000000",
}


@st.cache_resource
def processing_registry():
    return {"lock": threading.RLock(), "jobs": {}}


def timestamp_label():
    return datetime.now().isoformat(timespec="seconds")


def job_snapshot(job):
    return {
        key: value
        for key, value in job.items()
        if key != "thread"
    }


def update_processing_job_in_registry(registry, job_id, **updates):
    with registry["lock"]:
        job = registry["jobs"].get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = timestamp_label()


def update_processing_job(job_id, **updates):
    update_processing_job_in_registry(processing_registry(), job_id, **updates)


class BackgroundStatus:
    def __init__(self, registry, job_id):
        self.registry = registry
        self.job_id = job_id

    def info(self, message):
        update_processing_job_in_registry(self.registry, self.job_id, message=message)


def run_processing_job(registry, job_id, geometry, source_key, training_source=None):
    status = BackgroundStatus(registry, job_id)
    try:
        result = process_sampling(
            geometry,
            source_key=source_key,
            status_box=status,
            training_source=training_source,
        )
    except Exception as exc:
        update_processing_job_in_registry(
            registry,
            job_id,
            state="error",
            message=str(exc),
            error=str(exc),
            traceback=traceback.format_exc(),
            finished_at=timestamp_label(),
        )
        return

    update_processing_job_in_registry(
        registry,
        job_id,
        state="complete",
        message="Muestreo procesado y archivos creados.",
        result=result,
        finished_at=timestamp_label(),
    )


def start_processing_job(geometry, source_key, training_source=None):
    source_config = get_source_config(source_key)
    job_id = uuid.uuid4().hex[:12]
    job_geometry = json.loads(json.dumps(geometry))
    resolved_training_source = None
    if source_config.get("experimental"):
        resolved_training_source = training_source or DEFAULT_TRAINING_SOURCE
        if resolved_training_source not in TRAINING_SOURCE_OPTIONS:
            raise ValueError(
                f"Fuente de entrenamiento no soportada: {resolved_training_source}"
            )
    job = {
        "id": job_id,
        "source_key": source_key,
        "source_name": source_config["name"],
        "training_source": resolved_training_source,
        "training_source_label": (
            TRAINING_SOURCE_OPTIONS.get(resolved_training_source)
            if resolved_training_source
            else None
        ),
        "state": "running",
        "message": "Iniciando procesamiento...",
        "error": None,
        "traceback": None,
        "result": None,
        "started_at": timestamp_label(),
        "updated_at": timestamp_label(),
        "finished_at": None,
    }
    registry = processing_registry()
    thread = threading.Thread(
        target=run_processing_job,
        args=(registry, job_id, job_geometry, source_key, resolved_training_source),
        name=f"soil-sampler-job-{job_id}",
        daemon=True,
    )
    job["thread"] = thread

    with registry["lock"]:
        registry["jobs"][job_id] = job
    thread.start()
    return job_id


def get_processing_job(job_id):
    if not job_id:
        return None
    registry = processing_registry()
    with registry["lock"]:
        job = registry["jobs"].get(job_id)
        return job_snapshot(job) if job else None


def list_processing_jobs(limit=RECENT_JOB_LIMIT):
    registry = processing_registry()
    with registry["lock"]:
        jobs = [job_snapshot(job) for job in registry["jobs"].values()]
    jobs.sort(key=lambda job: job.get("started_at") or "", reverse=True)
    return jobs[:limit]


def rerun_app():
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun:
        rerun()


def extract_geometry(geojson_data):
    geojson_type = geojson_data.get("type")
    if geojson_type == "FeatureCollection":
        features = geojson_data.get("features", [])
        if not features:
            raise ValueError("El GeoJSON no contiene features.")
        return features[0].get("geometry")
    if geojson_type == "Feature":
        return geojson_data.get("geometry")
    if geojson_type in {"Polygon", "MultiPolygon"}:
        return geojson_data
    raise ValueError("Sube un GeoJSON con geometria Polygon o MultiPolygon.")


def validate_polygon(geometry):
    if not geometry:
        raise ValueError("No se encontro una geometria valida.")
    poly_geom = shape(geometry)
    if poly_geom.is_empty:
        raise ValueError("El poligono esta vacio.")
    if poly_geom.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("La geometria debe ser Polygon o MultiPolygon.")
    if not poly_geom.is_valid:
        poly_geom = poly_geom.buffer(0)
    if poly_geom.is_empty or not poly_geom.is_valid:
        raise ValueError("El poligono no es valido.")
    return poly_geom


def geometries_equivalent(left, right):
    if left == right:
        return True
    if not left or not right:
        return False
    try:
        return shape(left).equals(shape(right))
    except Exception:
        return False


def update_polygon_geometry(geometry):
    geometry_changed = not geometries_equivalent(
        st.session_state.polygon_geojson,
        geometry,
    )
    st.session_state.polygon_geojson = geometry
    if geometry_changed:
        st.session_state.result = None
    return geometry_changed


def iter_polygon_parts(geometry):
    if geometry is None or geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
        return
    if geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for part in geometry.geoms:
            yield from iter_polygon_parts(part)


def sanitize_geojson_filename(raw_name):
    name = (raw_name or "").strip().replace("\\", "/").split("/")[-1]
    lower_name = name.lower()
    if lower_name.endswith(".geojson"):
        name = name[:-8]
    elif lower_name.endswith(".json"):
        name = name[:-5]

    ascii_name = (
        unicodedata.normalize("NFKD", name)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_name).strip("_")
    return f"{safe_name or 'poligono'}.geojson"


def build_polygon_geojson_bytes(geometry, polygon_name):
    poly_geom = validate_polygon(geometry)
    display_name = (polygon_name or "").strip() or "poligono"
    feature_collection = {
        "type": "FeatureCollection",
        "name": display_name,
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "nombre": display_name,
                    "creado_en": datetime.now().isoformat(timespec="seconds"),
                },
                "geometry": mapping(poly_geom),
            }
        ],
    }
    return json.dumps(feature_collection, indent=2).encode("utf-8")


def unique_saved_polygon_path(file_name):
    SAVED_POLYGONS_DIR.mkdir(parents=True, exist_ok=True)
    candidate = SAVED_POLYGONS_DIR / file_name
    if not candidate.exists():
        return candidate

    suffix = 1
    while True:
        candidate = SAVED_POLYGONS_DIR / f"{Path(file_name).stem}_{suffix}.geojson"
        if not candidate.exists():
            return candidate
        suffix += 1


def save_polygon_geojson(geometry, polygon_name):
    file_name = sanitize_geojson_filename(polygon_name)
    payload = build_polygon_geojson_bytes(geometry, polygon_name)
    path = unique_saved_polygon_path(file_name)
    path.write_bytes(payload)
    return path


def estimate_window_pixels(dataset, poly_geom):
    minx, miny, maxx, maxy = poly_geom.bounds
    bounds = dataset.bounds
    left = max(minx, bounds.left)
    right = min(maxx, bounds.right)
    bottom = max(miny, bounds.bottom)
    top = min(maxy, bounds.top)
    if left >= right or bottom >= top:
        raise ValueError("El poligono no intersecta la cobertura del dataset.")

    window = from_bounds(left, bottom, right, top, transform=dataset.transform)
    width = int(np.ceil(abs(window.width)))
    height = int(np.ceil(abs(window.height)))
    return width * height, width, height


def get_source_config(source_key):
    if source_key not in DATA_SOURCES:
        raise ValueError(f"Fuente de datos no soportada: {source_key}")
    return DATA_SOURCES[source_key]


def check_pixel_limit(pixel_count, width, height):
    if pixel_count > MAX_PIXELS:
        raise ValueError(
            "El poligono cubre demasiados pixeles para una corrida "
            f"interactiva ({pixel_count:,}; limite {MAX_PIXELS:,}). "
            "Prueba con un poligono mas pequeno o sube el limite en app.py."
        )
    if width <= 0 or height <= 0:
        raise ValueError("El poligono es demasiado pequeno.")


def read_openlandmap_fraction_rasters(poly_geom, status_box=None):
    arrays = {}
    intervals = {}
    out_transform = None
    out_crs = None
    out_shape = None
    pixel_count = None

    with rasterio.Env(**GDAL_HTTP_OPTIONS):
        for key, layer in OPENLANDMAP_FRACTION_COGS.items():
            intervals[key] = {}

            for stat_key, url in layer["urls"].items():
                if status_box:
                    status_box.info(
                        f"Leyendo {layer['label']} {stat_key} desde OpenLandMap-soildb 120 m..."
                    )

                try:
                    with rasterio.open(url) as src:
                        if pixel_count is None:
                            pixel_count, width, height = estimate_window_pixels(src, poly_geom)
                            check_pixel_limit(pixel_count, width, height)

                        data, transform = raster_mask(
                            src,
                            [mapping(poly_geom)],
                            crop=True,
                            filled=False,
                            all_touched=True,
                        )
                        band = np.ma.masked_invalid(data[0].astype("float32"))

                        if out_shape is None:
                            out_shape = band.shape
                            out_transform = transform
                            out_crs = src.crs
                        elif band.shape != out_shape or not transform.almost_equals(out_transform):
                            raise RuntimeError(
                                "Los rasters de arena, limo y arcilla no estan alineados."
                            )

                        intervals[key][stat_key] = band
                except RasterioIOError as exc:
                    raise RuntimeError(
                        "No se pudo abrir el raster remoto de "
                        f"{layer['label']} {stat_key}. Revisa la conexion a internet o "
                        f"el acceso a s3.opengeohub.org. Detalle tecnico: {exc}"
                    ) from exc

            arrays[key] = intervals[key]["mean"]

    mean_texture, _ = classify_usda_texture(
        intervals["sand"]["mean"],
        intervals["silt"]["mean"],
        intervals["clay"]["mean"],
    )
    lower_texture, _ = classify_usda_texture(
        intervals["sand"]["p16"],
        intervals["silt"]["p16"],
        intervals["clay"]["p16"],
    )
    upper_texture, _ = classify_usda_texture(
        intervals["sand"]["p84"],
        intervals["silt"]["p84"],
        intervals["clay"]["p84"],
    )
    valid = mean_texture > 0
    stable = valid & (lower_texture == mean_texture) & (upper_texture == mean_texture)
    stable_pct = float(np.count_nonzero(stable) / np.count_nonzero(valid) * 100.0) if np.any(valid) else 0.0
    auxiliary_rasters = {
        "certainty_mask": {
            "filename": "consistencia_intervalo_68_120m.tif",
            "array": stable.astype("uint8"),
            "dtype": "uint8",
            "nodata": 0,
            "description": (
                "Mascara derivada: 1 cuando la clase USDA calculada con media "
                "coincide con las clases calculadas usando p0.16 y p0.84. "
                "El intervalo fuente es 68% (>60%); no es una probabilidad "
                "oficial de clase."
            ),
            "summary": {
                "interval_confidence_percent": OPENLANDMAP_INTERVAL_CONFIDENCE_PERCENT,
                "minimum_requested_certainty_percent": OPENLANDMAP_MIN_CERTAINTY_PERCENT,
                "valid_pixels": int(np.count_nonzero(valid)),
                "stable_pixels": int(np.count_nonzero(stable)),
                "stable_pixels_percent": round(stable_pct, 2),
            },
        }
    }

    return arrays, out_transform, out_crs, pixel_count, auxiliary_rasters


def estimate_projected_pixels(bounds, res):
    minx, miny, maxx, maxy = bounds
    width = int(np.ceil((maxx - minx) / res))
    height = int(np.ceil((maxy - miny) / res))
    return width * height, width, height


def transform_polygon(poly_geom, target_crs):
    return gpd.GeoSeries([poly_geom], crs="EPSG:4326").to_crs(target_crs).iloc[0]


def fetch_soilgrids_coverage(wcs, property_name, depth_label, poly_source_geom, bbox):
    coverage_id = f"{property_name}_{depth_label}_mean"
    response = wcs.getCoverage(
        identifier=coverage_id,
        crs=SOILGRIDS_WCS_CRS,
        bbox=bbox,
        resx=250,
        resy=250,
        format="GEOTIFF_INT16",
    )
    payload = response.read()
    if payload.lstrip().startswith(b"<"):
        detail = payload[:800].decode("utf-8", errors="replace")
        raise RuntimeError(f"SoilGrids devolvio XML en vez de GeoTIFF: {detail}")

    with MemoryFile(payload) as memfile:
        with memfile.open() as src:
            data, transform = raster_mask(
                src,
                [mapping(poly_source_geom)],
                crop=True,
                filled=False,
                all_touched=True,
            )
            crs = src.crs or CRS.from_wkt(SOILGRIDS_CRS_WKT)

    band = np.ma.masked_invalid(data[0].astype("float32"))
    band = np.ma.masked_where(band <= -30000, band)
    return band, transform, crs


def weighted_depth_average(weighted_bands):
    weighted_sum = None
    weight_total = None

    for band, thickness in weighted_bands:
        values = np.ma.filled(band, np.nan).astype("float32")
        valid = np.isfinite(values)
        if weighted_sum is None:
            weighted_sum = np.zeros(values.shape, dtype="float32")
            weight_total = np.zeros(values.shape, dtype="float32")
        weighted_sum[valid] += values[valid] * thickness
        weight_total[valid] += thickness

    with np.errstate(divide="ignore", invalid="ignore"):
        averaged = np.where(weight_total > 0, weighted_sum / weight_total, np.nan)
    return np.ma.masked_invalid(averaged.astype("float32"))


def read_soilgrids_fraction_rasters(poly_geom, status_box=None):
    arrays = {}
    out_transform = None
    out_crs = None
    out_shape = None
    source_geom = transform_polygon(poly_geom, SOILGRIDS_CRS)
    bbox = source_geom.bounds
    pixel_count, width, height = estimate_projected_pixels(bbox, 250)
    check_pixel_limit(pixel_count, width, height)

    for key, layer in DATA_SOURCES["soilgrids"]["layers"].items():
        property_name = layer["coverage_prefix"]
        if status_box:
            status_box.info(f"Leyendo {layer['label']} desde SoilGrids250m / ISRIC...")

        try:
            wcs = WebCoverageService(
                SOILGRIDS_LAYER_TEMPLATE.format(property=property_name),
                version="1.0.0",
                timeout=60,
            )
            weighted_bands = []
            for depth_label, thickness in SOILGRIDS_DEPTH_INTERVALS:
                if status_box:
                    status_box.info(
                        f"Leyendo {layer['label']} {depth_label} desde SoilGrids250m..."
                    )
                band, transform, crs = fetch_soilgrids_coverage(
                    wcs,
                    property_name,
                    depth_label,
                    source_geom,
                    bbox,
                )

                if out_shape is None:
                    out_shape = band.shape
                    out_transform = transform
                    out_crs = crs
                elif band.shape != out_shape or not transform.almost_equals(out_transform):
                    raise RuntimeError(
                        "Las coberturas SoilGrids descargadas no estan alineadas."
                    )

                weighted_bands.append((band, thickness))

            arrays[key] = weighted_depth_average(weighted_bands)
        except Exception as exc:
            raise RuntimeError(
                "No se pudo leer SoilGrids para "
                f"{layer['label']}. Revisa la conexion a internet o el acceso a "
                f"maps.isric.org. Detalle tecnico: {exc}"
            ) from exc

    return arrays, out_transform, out_crs, pixel_count, {}


def read_soil_fraction_rasters(
    poly_geom,
    source_key,
    status_box=None,
    training_source=None,
):
    if source_key == "openlandmap":
        return read_openlandmap_fraction_rasters(poly_geom, status_box)
    if source_key == "soilgrids":
        return read_soilgrids_fraction_rasters(poly_geom, status_box)
    if source_key == "sentinel_wosis":
        return read_sentinel_wosis_fraction_rasters(
            poly_geom,
            status_box=status_box,
            max_pixels=MAX_PIXELS,
            training_source=training_source or DEFAULT_TRAINING_SOURCE,
        )
    raise ValueError(f"Fuente de datos no soportada: {source_key}")


def classify_usda_texture(sand, silt, clay):
    sand_raw = np.ma.filled(sand, np.nan).astype("float32")
    silt_raw = np.ma.filled(silt, np.nan).astype("float32")
    clay_raw = np.ma.filled(clay, np.nan).astype("float32")
    total = sand_raw + silt_raw + clay_raw
    valid = np.isfinite(total) & (total > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        sand_pct = np.where(valid, sand_raw / total * 100.0, np.nan)
        silt_pct = np.where(valid, silt_raw / total * 100.0, np.nan)
        clay_pct = np.where(valid, clay_raw / total * 100.0, np.nan)

    conditions = [
        valid & (silt_pct + 1.5 * clay_pct < 15),
        valid & (silt_pct + 1.5 * clay_pct >= 15) & (silt_pct + 2 * clay_pct < 30),
        valid
        & (
            (
                (clay_pct >= 7)
                & (clay_pct < 20)
                & (sand_pct > 52)
                & (silt_pct + 2 * clay_pct >= 30)
            )
            | ((clay_pct < 7) & (silt_pct < 50) & (sand_pct > 43))
        ),
        valid
        & (clay_pct >= 7)
        & (clay_pct < 27)
        & (silt_pct >= 28)
        & (silt_pct < 50)
        & (sand_pct <= 52),
        valid
        & (
            ((silt_pct >= 50) & (clay_pct >= 12) & (clay_pct < 27))
            | ((silt_pct >= 50) & (silt_pct < 80) & (clay_pct < 12))
        ),
        valid & (silt_pct >= 80) & (clay_pct < 12),
        valid & (clay_pct >= 20) & (clay_pct < 35) & (silt_pct < 28) & (sand_pct > 45),
        valid
        & (clay_pct >= 27)
        & (clay_pct < 40)
        & (sand_pct > 20)
        & (sand_pct <= 45),
        valid & (clay_pct >= 27) & (clay_pct < 40) & (sand_pct <= 20),
        valid & (clay_pct >= 35) & (sand_pct > 45),
        valid & (clay_pct >= 40) & (silt_pct >= 40),
        valid & (clay_pct >= 40) & (sand_pct <= 45) & (silt_pct < 40),
    ]
    texture_grid = np.select(
        conditions,
        list(range(1, 13)),
        default=0,
    ).astype("int16")
    return texture_grid, int(np.count_nonzero(texture_grid))


def build_texture_zones(texture_grid, transform, crs, poly_geom):
    raster_shapes = shapes(
        texture_grid,
        mask=texture_grid > 0,
        transform=transform,
        connectivity=4,
    )

    polygons = []
    classes = []
    for geom, value in raster_shapes:
        value = int(value)
        if value > 0:
            polygons.append(shape(geom))
            classes.append(value)

    if not polygons:
        raise RuntimeError("No se generaron zonas texturales desde el raster.")

    raster_crs = crs or "EPSG:4326"
    texture_gdf = gpd.GeoDataFrame(
        {"texture_id": classes},
        geometry=polygons,
        crs=raster_crs,
    )
    polygon_gdf = gpd.GeoDataFrame(geometry=[poly_geom], crs="EPSG:4326")
    if texture_gdf.crs and polygon_gdf.crs != texture_gdf.crs:
        polygon_gdf = polygon_gdf.to_crs(texture_gdf.crs)

    clipped = gpd.clip(texture_gdf, polygon_gdf)
    if clipped.empty:
        raise RuntimeError("No quedaron zonas texturales dentro del poligono.")

    clipped["texture_id"] = clipped["texture_id"].astype(int)
    zone_records = []
    for _, row in clipped.iterrows():
        texture_id = int(row["texture_id"])
        for part in iter_polygon_parts(row.geometry):
            if part.is_empty or part.area <= 0:
                continue
            zone_records.append(
                {
                    "texture_id": texture_id,
                    "Textura": USDA_CLASSES[texture_id],
                    "geometry": part,
                }
            )

    if not zone_records:
        raise RuntimeError("No quedaron poligonos texturales validos dentro del poligono.")

    zones = gpd.GeoDataFrame(zone_records, geometry="geometry", crs=texture_gdf.crs)
    zones["zona_id"] = np.arange(1, len(zones) + 1)
    return zones[["zona_id", "texture_id", "Textura", "geometry"]]


def estimate_area_crs(gdf):
    gdf_wgs84 = gdf.to_crs("EPSG:4326") if gdf.crs else gdf.set_crs("EPSG:4326")
    try:
        estimated_crs = gdf_wgs84.estimate_utm_crs()
        if estimated_crs:
            return estimated_crs
    except RuntimeError:
        pass
    return "EPSG:6933"


def build_sampling_points(zones):
    area_crs = estimate_area_crs(zones)
    zones_metric = zones.to_crs(area_crs) if zones.crs else zones.set_crs("EPSG:4326").to_crs(area_crs)
    point_records = []
    unit_records = []

    for _, zone in zones_metric.iterrows():
        zone_area_ha = zone.geometry.area / SQM_PER_HA
        base_record = {
            "zona_id": int(zone["zona_id"]),
            "texture_id": int(zone["texture_id"]),
            "Textura": zone["Textura"],
            "area_zona_ha": round(zone_area_ha, 4),
        }
        unit_records.append({**base_record, "geometry": zone.geometry})
        point_records.append(
            {**base_record, "metodo_punto": "centroide", "geometry": zone.geometry.centroid}
        )

    sampling_units = gpd.GeoDataFrame(unit_records, geometry="geometry", crs=area_crs)
    points = gpd.GeoDataFrame(point_records, geometry="geometry", crs=area_crs)
    points = points.to_crs("EPSG:4326")
    sampling_units = sampling_units.to_crs("EPSG:4326")
    points["ID_Punto"] = np.arange(1, len(points) + 1)
    sampling_units["ID_Unidad"] = points["ID_Punto"].to_numpy()
    points["Lon"] = points.geometry.x
    points["Lat"] = points.geometry.y

    point_columns = [
        "ID_Punto",
        "zona_id",
        "texture_id",
        "Textura",
        "area_zona_ha",
        "metodo_punto",
        "Lat",
        "Lon",
        "geometry",
    ]
    unit_columns = [
        "ID_Unidad",
        "zona_id",
        "texture_id",
        "Textura",
        "area_zona_ha",
        "geometry",
    ]
    return points[point_columns], sampling_units[unit_columns]


def next_output_dir():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / f"muestreo_{stamp}"
    suffix = 1
    while output_dir.exists():
        output_dir = OUTPUT_ROOT / f"muestreo_{stamp}_{suffix}"
        suffix += 1
    output_dir.mkdir(parents=True)
    return output_dir


def write_single_band_raster(path, array, transform, crs, dtype, nodata, tags=None):
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": dtype,
        "crs": crs or "EPSG:4326",
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
        if tags:
            dst.update_tags(**tags)


def write_outputs(
    texture_grid,
    transform,
    crs,
    zones,
    points,
    sampling_units,
    pixel_count,
    source_config,
    auxiliary_rasters=None,
):
    output_dir = next_output_dir()
    raster_path = output_dir / f"raster_textura_{source_config['resolution_slug']}.tif"
    zones_path = output_dir / "zonas_texturales.geojson"
    sampling_units_path = output_dir / "poligonos_muestreo.geojson"
    points_geojson_path = output_dir / "puntos_muestreo.geojson"
    points_csv_path = output_dir / "puntos_muestreo.csv"
    classes_path = output_dir / "tabla_clases_textura.csv"
    metadata_path = output_dir / "metadata_fuente.json"

    write_single_band_raster(
        raster_path,
        texture_grid,
        transform,
        crs,
        "int16",
        NODATA_CLASS,
        tags={"source": source_config["description"]},
    )

    zones_out = zones.to_crs("EPSG:4326") if zones.crs else zones.set_crs("EPSG:4326")
    points_out = points.to_crs("EPSG:4326") if points.crs else points.set_crs("EPSG:4326")
    sampling_units_out = (
        sampling_units.to_crs("EPSG:4326")
        if sampling_units.crs
        else sampling_units.set_crs("EPSG:4326")
    )

    zones_path.write_text(zones_out.to_json(), encoding="utf-8")
    sampling_units_path.write_text(sampling_units_out.to_json(), encoding="utf-8")
    points_geojson_path.write_text(points_out.to_json(), encoding="utf-8")
    points_out.drop(columns="geometry").to_csv(points_csv_path, index=False, encoding="utf-8")

    class_table = pd.DataFrame(
        [{"texture_id": key, "Textura": value} for key, value in USDA_CLASSES.items()]
    )
    class_table.to_csv(classes_path, index=False, encoding="utf-8")

    metadata = {
        "source": source_config["name"],
        "description": source_config["description"],
        "catalog": source_config["catalog"],
        "resolution": source_config["resolution_label"],
        "layers": source_config["layers"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pixel_count_bbox": pixel_count,
        "nodata_value": NODATA_CLASS,
        "valid_texture_pixels": int(np.count_nonzero(texture_grid)),
        "zone_count": int(len(zones_out)),
        "sampling_unit_count": int(len(sampling_units_out)),
        "sampling_rule": "un_punto_centroide_por_poligono_textural",
        "point_count": int(len(points_out)),
        "outputs": {
            "raster": raster_path.name,
            "zones_geojson": zones_path.name,
            "sampling_units_geojson": sampling_units_path.name,
            "points_geojson": points_geojson_path.name,
            "points_csv": points_csv_path.name,
            "classes_csv": classes_path.name,
        },
    }
    files = {
        "raster": raster_path,
        "zones_geojson": zones_path,
        "sampling_units_geojson": sampling_units_path,
        "points_geojson": points_geojson_path,
        "points_csv": points_csv_path,
        "classes_csv": classes_path,
        "metadata": metadata_path,
    }

    auxiliary_rasters = auxiliary_rasters or {}
    auxiliary_metadata = {}
    for key, raster in auxiliary_rasters.items():
        path = output_dir / raster["filename"]
        write_single_band_raster(
            path,
            raster["array"],
            transform,
            crs,
            raster["dtype"],
            raster["nodata"],
            tags={"source": source_config["description"], "description": raster["description"]},
        )
        files[key] = path
        metadata["outputs"][key] = path.name
        auxiliary_metadata[key] = {
            "description": raster["description"],
            "summary": raster.get("summary", {}),
        }
    if auxiliary_metadata:
        metadata["auxiliary_rasters"] = auxiliary_metadata

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return files, output_dir


def process_sampling(
    geometry,
    source_key=DEFAULT_SOURCE_KEY,
    status_box=None,
    training_source=None,
):
    source_config = get_source_config(source_key)
    poly_geom = validate_polygon(geometry)
    fractions, transform, crs, pixel_count, auxiliary_rasters = read_soil_fraction_rasters(
        poly_geom,
        source_key,
        status_box,
        training_source=training_source,
    )

    if status_box:
        status_box.info("Clasificando textura USDA desde arena/limo/arcilla reales...")
    texture_grid, valid_pixels = classify_usda_texture(
        fractions["sand"],
        fractions["silt"],
        fractions["clay"],
    )
    if valid_pixels == 0:
        raise RuntimeError("El dataset no devolvio pixeles validos para el poligono.")

    if status_box:
        status_box.info("Vectorizando zonas texturales y creando puntos...")
    zones = build_texture_zones(texture_grid, transform, crs, poly_geom)
    points, sampling_units = build_sampling_points(zones)
    files, output_dir = write_outputs(
        texture_grid,
        transform,
        crs,
        zones,
        points,
        sampling_units,
        pixel_count,
        source_config,
        auxiliary_rasters,
    )

    return {
        "zones": zones,
        "points": points,
        "sampling_units": sampling_units,
        "files": files,
        "output_dir": output_dir,
        "valid_pixels": valid_pixels,
        "pixel_count": pixel_count,
        "source_key": source_key,
        "source_name": source_config["name"],
        "resolution_label": source_config["resolution_label"],
        "training_source": training_source,
        "training_source_label": (
            TRAINING_SOURCE_OPTIONS.get(training_source) if training_source else None
        ),
        "auxiliary_rasters": auxiliary_rasters,
    }


def add_satellite_tiles(map_obj):
    esri_tiles = (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    )
    esri_attr = (
        "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, "
        "GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
    )
    folium.TileLayer(tiles=esri_tiles, attr=esri_attr, name="Esri Satellite").add_to(map_obj)


def render_input_map():
    map_obj = folium.Map(location=[9.9281, -84.0907], zoom_start=7, tiles=None)
    add_satellite_tiles(map_obj)

    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "rectangle": True,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "polygon": True,
        },
        edit_options={"edit": False},
    ).add_to(map_obj)

    if st.session_state.polygon_geojson:
        folium.GeoJson(st.session_state.polygon_geojson, name="Poligono").add_to(map_obj)
        poly_shape = shape(st.session_state.polygon_geojson)
        map_obj.fit_bounds(
            [
                [poly_shape.bounds[1], poly_shape.bounds[0]],
                [poly_shape.bounds[3], poly_shape.bounds[2]],
            ]
        )

    return st_folium(
        map_obj,
        width=700,
        height=500,
        returned_objects=["last_active_drawing"],
    )


def render_polygon_export_controls():
    st.markdown("### Guardar poligono")
    polygon_name = st.text_input(
        "Nombre del GeoJSON",
        value="poligono_campo",
        key="polygon_export_name",
    )

    if not st.session_state.polygon_geojson:
        st.info("Dibuja o sube un poligono para guardarlo como GeoJSON.")
        return

    try:
        file_name = sanitize_geojson_filename(polygon_name)
        payload = build_polygon_geojson_bytes(st.session_state.polygon_geojson, polygon_name)
    except ValueError as exc:
        st.error(f"No se pudo preparar el GeoJSON: {exc}")
        return

    save_col, download_col = st.columns(2)
    with save_col:
        if st.button("Guardar GeoJSON", key="save_input_polygon"):
            saved_path = save_polygon_geojson(st.session_state.polygon_geojson, polygon_name)
            st.success(f"Guardado en `{saved_path}`.")
    with download_col:
        st.download_button(
            label="Descargar GeoJSON",
            data=payload,
            file_name=file_name,
            mime="application/geo+json",
            key="download_input_polygon",
            on_click="ignore",
        )


def render_result_map(result):
    zones = result["zones"]
    zones_wgs84 = zones.to_crs("EPSG:4326") if zones.crs else zones.set_crs("EPSG:4326")
    sampling_units = result["sampling_units"]
    sampling_units_wgs84 = (
        sampling_units.to_crs("EPSG:4326")
        if sampling_units.crs
        else sampling_units.set_crs("EPSG:4326")
    )
    points = result["points"]
    center = [points["Lat"].mean(), points["Lon"].mean()]
    result_map = folium.Map(location=center, zoom_start=15, tiles=None)
    add_satellite_tiles(result_map)

    for _, row in zones_wgs84.iterrows():
        color = USDA_COLORS.get(row["texture_id"], "#cccccc")
        geojson = gpd.GeoSeries([row["geometry"]], crs="EPSG:4326").to_json()
        folium.GeoJson(
            geojson,
            style_function=lambda feature, color=color: {
                "fillColor": color,
                "color": "white",
                "weight": 2,
                "fillOpacity": 0.5,
            },
            tooltip=f"Zona {row['zona_id']}: {row['Textura']}",
        ).add_to(result_map)

    for _, row in sampling_units_wgs84.iterrows():
        geojson = gpd.GeoSeries([row["geometry"]], crs="EPSG:4326").to_json()
        folium.GeoJson(
            geojson,
            style_function=lambda feature: {
                "fillColor": "transparent",
                "color": "#111827",
                "weight": 1,
                "dashArray": "4,4",
                "fillOpacity": 0,
            },
            tooltip=(
                f"Poligono {row['ID_Unidad']} | Zona {row['zona_id']} | "
                f"{row['area_zona_ha']} ha"
            ),
        ).add_to(result_map)

    for _, row in points.iterrows():
        folium.Marker(
            location=[row["Lat"], row["Lon"]],
            popup=(
                f"<b>Punto {row['ID_Punto']}</b><br>"
                f"Zona: {row['zona_id']}<br>"
                f"Textura: {row['Textura']}<br>"
                f"Area poligono: {row['area_zona_ha']} ha<br>"
                f"Metodo: {row['metodo_punto']}"
            ),
            icon=folium.Icon(color="green", icon="info-sign"),
        ).add_to(result_map)

    bounds = zones_wgs84.total_bounds
    result_map.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    st_folium(result_map, width=700, height=500, returned_objects=[])


def render_downloads(files, result):
    st.markdown("### Archivos generados")
    download_items = [
        (f"Raster textura {result['resolution_label']} (TIF)", "raster", "image/tiff"),
        ("Zonas texturales (GeoJSON)", "zones_geojson", "application/geo+json"),
        ("Poligonos de muestreo (GeoJSON)", "sampling_units_geojson", "application/geo+json"),
        ("Puntos de muestreo (GeoJSON)", "points_geojson", "application/geo+json"),
        ("Puntos de muestreo (CSV)", "points_csv", "text/csv"),
        ("Tabla de clases (CSV)", "classes_csv", "text/csv"),
        ("Metadata de fuente (JSON)", "metadata", "application/json"),
    ]
    if "certainty_mask" in files:
        download_items.insert(
            1,
            ("Consistencia intervalo 68% (TIF)", "certainty_mask", "image/tiff"),
        )
    if "sentinel_bare_observations" in files:
        download_items.insert(
            1,
            (
                "Observaciones suelo descubierto Sentinel-2 (TIF)",
                "sentinel_bare_observations",
                "image/tiff",
            ),
        )
    if "sentinel_model_uncertainty" in files:
        download_items.insert(
            2,
            (
                "Incertidumbre modelo Sentinel-WoSIS (TIF)",
                "sentinel_model_uncertainty",
                "image/tiff",
            ),
        )
    if "sentinel_bare_score" in files:
        download_items.insert(
            3,
            ("Score suelo descubierto Sentinel-2 (TIF)", "sentinel_bare_score", "image/tiff"),
        )

    for label, key, mime_type in download_items:
        path = files[key]
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime_type,
            key=f"download_{key}",
            on_click="ignore",
        )


def load_completed_job_result(job):
    result = job.get("result")
    if not result:
        return False
    st.session_state.result = result
    st.session_state.processing_job_id = None
    return True


def render_current_processing_job():
    job = get_processing_job(st.session_state.get("processing_job_id"))
    if not job:
        return False

    state = job["state"]
    if state == "running":
        st.info(job["message"])
        st.caption(
            f"Proceso en segundo plano: {job['source_name']} | "
            f"Inicio: {job['started_at']} | Ultima actualizacion: {job['updated_at']}"
        )
        if st.button("Actualizar estado", key="refresh_processing_job"):
            rerun_app()
        time.sleep(2)
        rerun_app()
        return True

    if state == "complete":
        if load_completed_job_result(job):
            st.success(job["message"])
        else:
            st.error("El proceso termino, pero no se encontro el resultado en memoria.")
        return False

    if state == "error":
        st.session_state.processing_job_id = None
        st.session_state.result = None
        st.error(job["message"])
        with st.expander("Detalle tecnico"):
            st.code(job.get("traceback") or job["message"])
        return False

    return False


def render_recent_processing_jobs():
    jobs = list_processing_jobs()
    if not jobs:
        return

    current_job_id = st.session_state.get("processing_job_id")
    visible_jobs = [job for job in jobs if job["id"] != current_job_id]
    if not visible_jobs:
        return

    running_jobs = [job for job in visible_jobs if job["state"] == "running"]
    with st.expander("Procesos recientes", expanded=bool(running_jobs)):
        for job in visible_jobs:
            st.write(
                f"`{job['id']}` | {job['source_name']} | {job['state']} | "
                f"Inicio: {job['started_at']} | {job['message']}"
            )
            if job["state"] == "complete":
                if st.button("Cargar resultado", key=f"load_job_{job['id']}"):
                    if load_completed_job_result(job):
                        rerun_app()
            elif job["state"] == "running":
                if st.button("Seguir proceso", key=f"follow_job_{job['id']}"):
                    st.session_state.processing_job_id = job["id"]
                    rerun_app()
            elif job["state"] == "error":
                with st.expander(f"Detalle tecnico {job['id']}"):
                    st.code(job.get("traceback") or job["message"])


def main():
    st.set_page_config(page_title="Soil Point Sampler", layout="wide")

    if "polygon_geojson" not in st.session_state:
        st.session_state.polygon_geojson = None
    if "result" not in st.session_state:
        st.session_state.result = None
    if "source_key" not in st.session_state:
        st.session_state.source_key = DEFAULT_SOURCE_KEY
    if "training_source" not in st.session_state:
        st.session_state.training_source = DEFAULT_TRAINING_SOURCE
    if "processing_job_id" not in st.session_state:
        st.session_state.processing_job_id = None

    st.title("Metodologia Establecimiento Puntos de Muestreo de Suelos")
    st.markdown(
        "Define puntos preliminares de muestreo dentro de un poligono usando "
        "textura USDA calculada desde fracciones reales de arena, limo y arcilla."
    )
    st.caption(
        "Fuentes reales disponibles: OpenLandMap-soildb 120 m, SoilGrids250m / "
        "ISRIC WCS y modo experimental Sentinel-2 + WoSIS. No se generan datos simulados."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Definir poligono")
        uploaded_file = st.file_uploader("Subir poligono (GeoJSON)", type=["geojson", "json"])
        if uploaded_file is not None:
            try:
                uploaded_file.seek(0)
                geometry = extract_geometry(json.load(uploaded_file))
                validate_polygon(geometry)
                if update_polygon_geometry(geometry):
                    st.success("Poligono cargado correctamente.")
            except (json.JSONDecodeError, ValueError) as exc:
                st.error(f"No se pudo cargar el poligono: {exc}")

        output = render_input_map()
        if output.get("last_active_drawing"):
            new_geom = output["last_active_drawing"]["geometry"]
            if not geometries_equivalent(st.session_state.polygon_geojson, new_geom):
                try:
                    validate_polygon(new_geom)
                    if update_polygon_geometry(new_geom):
                        st.success("Poligono dibujado registrado.")
                except ValueError as exc:
                    st.error(f"El dibujo no es valido: {exc}")

        render_polygon_export_controls()

    with col2:
        st.subheader("2. Procesar y descargar")
        source_options = list(DATA_SOURCES.keys())
        selected_source = st.selectbox(
            "Fuente de datos",
            source_options,
            index=source_options.index(st.session_state.source_key),
            format_func=lambda key: DATA_SOURCES[key]["name"],
        )
        if selected_source != st.session_state.source_key:
            st.session_state.source_key = selected_source
            st.session_state.result = None

        source_config = get_source_config(st.session_state.source_key)
        st.caption(source_config["description"])
        if source_config.get("experimental"):
            training_options = list(TRAINING_SOURCE_OPTIONS.keys())
            selected_training = st.selectbox(
                "Perfiles de entrenamiento",
                training_options,
                index=training_options.index(st.session_state.training_source)
                if st.session_state.training_source in TRAINING_SOURCE_OPTIONS
                else training_options.index(DEFAULT_TRAINING_SOURCE),
                format_func=lambda key: TRAINING_SOURCE_OPTIONS[key],
                help=(
                    "Las calicatas Costa Rica aportan ~1,600 perfiles 0-30 cm con "
                    "arena/limo/arcilla. Se filtran al buffer de 250 km y se usan "
                    "hasta ~400 perfiles cercanos al poligono."
                ),
            )
            if selected_training != st.session_state.training_source:
                st.session_state.training_source = selected_training
                st.session_state.result = None
            st.warning(
                "Modo experimental: entrena un modelo local con perfiles WoSIS y/o "
                "calicatas Costa Rica, mas compuestos Sentinel-2 de suelo descubierto. "
                "Puede ser lento, depende de que existan suficientes perfiles y pixeles "
                "descubiertos, y sus predicciones no sustituyen muestreo ni cartografia local."
            )
            st.caption(
                "Recomendacion: usarlo para comparacion exploratoria contra "
                "OpenLandMap/SoilGrids y revisar la incertidumbre exportada."
            )
        st.info(
            f"La corrida necesita conexion a {source_config['network_host']}. Si la fuente real no "
            "responde, el proceso se detiene en lugar de inventar datos."
        )

        current_job = get_processing_job(st.session_state.processing_job_id)
        job_running = bool(current_job and current_job["state"] == "running")
        if st.button("Procesar muestreo", type="primary", disabled=job_running):
            if not st.session_state.polygon_geojson:
                st.error("Dibuja o sube un poligono primero.")
            else:
                try:
                    validate_polygon(st.session_state.polygon_geojson)
                    training_source = (
                        st.session_state.training_source
                        if source_config.get("experimental")
                        else None
                    )
                    st.session_state.processing_job_id = start_processing_job(
                        st.session_state.polygon_geojson,
                        st.session_state.source_key,
                        training_source=training_source,
                    )
                    st.session_state.result = None
                    rerun_app()
                except Exception as exc:
                    st.error(str(exc))

        render_current_processing_job()
        render_recent_processing_jobs()

        if st.session_state.result:
            result = st.session_state.result
            st.write(f"Carpeta de salida: `{result['output_dir']}`")
            source_line = (
                f"Fuente: {result['source_name']} | "
                f"Resolucion: {result['resolution_label']} | "
                f"Pixeles validos: {result['valid_pixels']:,} | "
                f"Zonas: {len(result['zones']):,} | "
                f"Poligonos de muestreo: {len(result['sampling_units']):,} | "
                f"Puntos: {len(result['points']):,}"
            )
            if result.get("training_source_label"):
                source_line += f" | Entrenamiento: {result['training_source_label']}"
            st.write(source_line)
            certainty = result["auxiliary_rasters"].get("certainty_mask")
            if certainty:
                summary = certainty["summary"]
                st.write(
                    "Consistencia intervalo 68%: "
                    f"{summary['stable_pixels']:,} de {summary['valid_pixels']:,} pixeles "
                    f"({summary['stable_pixels_percent']}%)."
                )
            sentinel_uncertainty = result["auxiliary_rasters"].get("sentinel_model_uncertainty")
            sentinel_bare = result["auxiliary_rasters"].get("sentinel_bare_observations")
            if sentinel_uncertainty and sentinel_bare:
                uncertainty_summary = sentinel_uncertainty["summary"]
                bare_summary = sentinel_bare["summary"]
                training_by_source = uncertainty_summary.get("training_profiles_by_source") or {}
                source_bits = []
                if training_by_source.get("wosis"):
                    source_bits.append(f"WoSIS={training_by_source['wosis']:,}")
                if training_by_source.get("calicatas_cr"):
                    source_bits.append(
                        f"calicatas CR={training_by_source['calicatas_cr']:,}"
                    )
                profiles_label = (
                    f"{uncertainty_summary['training_profiles_with_bare_sentinel']:,} "
                    "perfiles con suelo descubierto"
                )
                if source_bits:
                    profiles_label += f" ({', '.join(source_bits)})"
                sentinel_details = [
                    profiles_label,
                    (
                        f"{bare_summary['bare_soil_pixels_percent']}% del poligono con "
                        "compuesto Sentinel-2 descubierto"
                    ),
                ]
                if "mean_bare_observations_per_prediction_pixel" in bare_summary:
                    sentinel_details.append(
                        "media "
                        f"{bare_summary['mean_bare_observations_per_prediction_pixel']} "
                        "observaciones descubiertas por pixel"
                    )
                if "spatial_cv_mae_mean_fraction" in uncertainty_summary:
                    sentinel_details.append(
                        "MAE CV espacial medio "
                        f"{uncertainty_summary['spatial_cv_mae_mean_fraction']} puntos porcentuales"
                    )
                if uncertainty_summary.get("training_source_label"):
                    sentinel_details.insert(
                        0,
                        f"entrenamiento={uncertainty_summary['training_source_label']}",
                    )
                st.write(
                    "Sentinel experimental: "
                    + "; ".join(sentinel_details)
                    + "."
                )
                if uncertainty_summary.get("model_quality_warning"):
                    st.warning(uncertainty_summary["model_quality_warning"])
            render_result_map(result)
            render_downloads(result["files"], result)


if __name__ == "__main__":
    main()
