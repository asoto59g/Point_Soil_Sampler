from datetime import datetime, timezone
import json
import os
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, urlunsplit

try:
    from pyproj import datadir as pyproj_datadir

    proj_data_dir = pyproj_datadir.get_data_dir()
    os.environ["PROJ_DATA"] = proj_data_dir
    os.environ["PROJ_LIB"] = proj_data_dir
except Exception:
    pass

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import Point, box, mapping, shape


SENTINEL_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SENTINEL_COLLECTION = "sentinel-2-l2a"
SENTINEL_DATETIME_START = "2016-01-01"
SENTINEL_TARGET_RESOLUTION_M = 20
SENTINEL_CLOUD_COVER_LT = 60
SENTINEL_MIN_WOSIS_SAMPLES = 30
SENTINEL_WOSIS_BUFFER_KM = 250
SENTINEL_MAX_TRAINING_PROFILES = 400
SENTINEL_MAX_TRAINING_ITEMS = 120
SENTINEL_MAX_PREDICTION_ITEMS = 90
SENTINEL_MIN_TRAINING_ITEMS = 25
SENTINEL_TRAINING_TARGET_VALID_SAMPLES = 180
SENTINEL_MIN_PREDICTION_ITEMS = 25
SENTINEL_TARGET_BARE_PIXEL_PERCENT = 85.0
SENTINEL_TARGET_MEAN_BARE_OBSERVATIONS = 3.0
SENTINEL_TRAINING_TARGET_MEAN_BARE_OBSERVATIONS = 2.0
# Costa Rica / Pacific Central America dry season (prefer bare-soil scenes).
SENTINEL_DRY_SEASON_MONTHS = (12, 1, 2, 3, 4)
SENTINEL_DRY_SEASON_SCORE_BONUS = 0.12
# Stricter bare-soil spectral gates (SCL 5 primary, SCL 7 fallback).
SENTINEL_BARE_SCL5_NDVI_MAX = 0.25
SENTINEL_BARE_SCL5_NDWI_MAX = 0.05
SENTINEL_BARE_SCL5_BSI_MIN = -0.10
SENTINEL_BARE_SCL5_RED_MIN = 0.03
SENTINEL_BARE_SCL7_NDVI_MAX = 0.18
SENTINEL_BARE_SCL7_NDWI_MAX = 0.03
SENTINEL_BARE_SCL7_BSI_MIN = 0.00
SENTINEL_BARE_SCL7_RED_MIN = 0.04
SENTINEL_BARE_NDVI_MIN = -0.05
SENTINEL_FRACTION_SMOOTHING_RADIUS_PIXELS = 2
SENTINEL_MIN_SMOOTHING_NEIGHBORS = 5
SENTINEL_MAX_ITEM_FAILURES = 20
SENTINEL_SIGNED_URL_MIN_TTL_SECONDS = 10 * 60
SENTINEL_RF_TREES = 300
SENTINEL_RF_RANDOM_STATE = 42
SENTINEL_GDAL_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_CONNECTTIMEOUT": "20",
    "GDAL_HTTP_TIMEOUT": "90",
    "GDAL_HTTP_MAX_RETRY": "2",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "50000000",
}
WOSIS_WFS_URL = "https://maps.isric.org/mapserv"
WOSIS_REQUEST_TIMEOUT_SECONDS = 120
WOSIS_REQUEST_RETRIES = 3
WOSIS_TILE_SIZE_DEGREES = 2.0
WOSIS_MIN_TILE_SIZE_DEGREES = 0.25
WOSIS_PROPERTIES = {
    "sand": "wosis_latest_sand",
    "silt": "wosis_latest_silt",
    "clay": "wosis_latest_clay",
}
TRAINING_SOURCE_WOSIS = "wosis"
TRAINING_SOURCE_CALICATAS_CR = "calicatas_cr"
TRAINING_SOURCE_BOTH = "ambos"
TRAINING_SOURCE_OPTIONS = {
    TRAINING_SOURCE_WOSIS: "Solo WoSIS/ISRIC",
    TRAINING_SOURCE_CALICATAS_CR: "Solo calicatas Costa Rica",
    TRAINING_SOURCE_BOTH: "WoSIS + calicatas Costa Rica",
}
DEFAULT_TRAINING_SOURCE = TRAINING_SOURCE_BOTH
CALICATAS_CR_CSV_PATH = Path(__file__).resolve().parent / "Calicatas_01_02_21_Costa_Rica.csv"
CALICATAS_CR_PROFILE_ID_OFFSET = 10_000_000
CALICATAS_CR_SOURCE_LABEL = "calicatas_cr"
WOSIS_SOURCE_LABEL = "wosis"
SENTINEL_ASSET_ALIASES = {
    "blue": ("B02", "blue"),
    "green": ("B03", "green"),
    "red": ("B04", "red"),
    "rededge1": ("B05", "rededge1"),
    "rededge2": ("B06", "rededge2"),
    "rededge3": ("B07", "rededge3"),
    "nir": ("B08", "nir"),
    "nir08": ("B8A", "nir08"),
    "swir1": ("B11", "swir16"),
    "swir2": ("B12", "swir22"),
    "scl": ("SCL", "scl"),
}
# Reflectance assets read per scene (excluding SCL).
SENTINEL_REFLECTANCE_BANDS = (
    "blue",
    "green",
    "red",
    "rededge1",
    "rededge2",
    "rededge3",
    "nir",
    "nir08",
    "swir1",
    "swir2",
)
SENTINEL_SPECTRAL_FEATURE_NAMES = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B11",
    "B12",
    "NDVI",
    "SAVI",
    "MSAVI",
    "BSI",
    "CI",
    "NDWI",
    "GEOI",
    "BI",
    "NDRE",
    "NDRE2",
]
SENTINEL_MODEL_FEATURE_NAMES = (
    [f"{feature_name}_best" for feature_name in SENTINEL_SPECTRAL_FEATURE_NAMES]
    + [f"{feature_name}_mean" for feature_name in SENTINEL_SPECTRAL_FEATURE_NAMES]
    + [
        "BARE_OBS",
        "BARE_SCORE",
        "ELEV",
        "SLOPE_DEG",
        "ASPECT_SIN",
        "ASPECT_COS",
        "CURV",
        "VV_DB",
        "VH_DB",
        "VV_VH_DB",
    ]
)


def update_status(status_box, message):
    if status_box:
        status_box.info(message)


def require_experimental_dependencies():
    missing = []
    modules = {}
    for package_name, import_name in [
        ("requests", "requests"),
        ("pystac-client", "pystac_client"),
        ("planetary-computer", "planetary_computer"),
        ("scikit-learn", "sklearn"),
    ]:
        try:
            modules[import_name] = __import__(import_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        raise RuntimeError(
            "El modo experimental Sentinel-2 + WoSIS requiere dependencias "
            f"adicionales no instaladas: {', '.join(missing)}. Ejecuta "
            "`pip install -r requirements.txt` y vuelve a intentar."
        )
    return modules


def sentinel_datetime_range():
    today = datetime.now(timezone.utc).date().isoformat()
    return f"{SENTINEL_DATETIME_START}/{today}"


def estimate_area_crs_for_geometry(poly_geom):
    gdf = gpd.GeoDataFrame(geometry=[poly_geom], crs="EPSG:4326")
    try:
        estimated_crs = gdf.estimate_utm_crs()
        if estimated_crs:
            return estimated_crs
    except RuntimeError:
        pass
    return "EPSG:6933"


def project_geometry(poly_geom, target_crs):
    return gpd.GeoSeries([poly_geom], crs="EPSG:4326").to_crs(target_crs).iloc[0]


def expanded_wgs84_bounds(poly_geom, buffer_km):
    target_crs = estimate_area_crs_for_geometry(poly_geom)
    metric_geom = project_geometry(poly_geom, target_crs)
    buffered = metric_geom.buffer(buffer_km * 1000.0)
    buffered_wgs84 = gpd.GeoSeries([buffered], crs=target_crs).to_crs("EPSG:4326").iloc[0]
    return tuple(float(value) for value in buffered_wgs84.bounds)


def validate_grid_size(pixel_count, width, height, max_pixels):
    if width <= 0 or height <= 0:
        raise ValueError("El poligono es demasiado pequeno para Sentinel-2.")
    if pixel_count > max_pixels:
        raise ValueError(
            "El modo Sentinel-2 + WoSIS genera demasiados pixeles "
            f"({pixel_count:,}; limite {max_pixels:,}) a "
            f"{SENTINEL_TARGET_RESOLUTION_M} m. Prueba con un poligono menor."
        )


def build_prediction_grid(poly_geom, max_pixels):
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin

    target_crs = estimate_area_crs_for_geometry(poly_geom)
    metric_geom = project_geometry(poly_geom, target_crs)
    minx, miny, maxx, maxy = metric_geom.bounds
    resolution = SENTINEL_TARGET_RESOLUTION_M
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    pixel_count = width * height
    validate_grid_size(pixel_count, width, height, max_pixels)

    transform = from_origin(minx, maxy, resolution, resolution)
    mask = geometry_mask(
        [mapping(metric_geom)],
        out_shape=(height, width),
        transform=transform,
        invert=True,
        all_touched=True,
    )
    return {
        "crs": target_crs,
        "geometry": metric_geom,
        "transform": transform,
        "width": width,
        "height": height,
        "pixel_count": pixel_count,
        "mask": mask,
    }


def empty_wosis_property_gdf(property_key):
    return gpd.GeoDataFrame(
        columns=["profile_id", property_key, "geometry"],
        geometry="geometry",
        crs="EPSG:4326",
    )


def split_bounds(bounds, tile_size_degrees=WOSIS_TILE_SIZE_DEGREES):
    minx, miny, maxx, maxy = bounds
    x_steps = max(1, int(np.ceil((maxx - minx) / tile_size_degrees)))
    y_steps = max(1, int(np.ceil((maxy - miny) / tile_size_degrees)))
    x_edges = np.linspace(minx, maxx, x_steps + 1)
    y_edges = np.linspace(miny, maxy, y_steps + 1)

    for x_index in range(x_steps):
        for y_index in range(y_steps):
            yield (
                float(x_edges[x_index]),
                float(y_edges[y_index]),
                float(x_edges[x_index + 1]),
                float(y_edges[y_index + 1]),
            )


def split_bounds_quadrants(bounds):
    minx, miny, maxx, maxy = bounds
    midx = (minx + maxx) / 2.0
    midy = (miny + maxy) / 2.0
    return [
        (minx, miny, midx, midy),
        (midx, miny, maxx, midy),
        (minx, midy, midx, maxy),
        (midx, midy, maxx, maxy),
    ]


def response_excerpt(response, max_chars=500):
    text = response.text.replace("\r", " ").replace("\n", " ").strip()
    return text[:max_chars]


def parse_wosis_geojson_response(response, property_key, tile_bounds):
    content_type = response.headers.get("content-type", "sin content-type")
    excerpt = response_excerpt(response)
    if response.status_code != 200:
        raise RuntimeError(
            f"WoSIS/ISRIC no respondio correctamente para {property_key}: "
            f"HTTP {response.status_code}, {content_type}, tesela {tile_bounds}. "
            f"Respuesta: {excerpt}"
        )
    if response.text.lstrip().startswith("<"):
        raise RuntimeError(
            f"WoSIS/ISRIC devolvio XML/HTML en vez de GeoJSON para {property_key}, "
            f"tesela {tile_bounds}. Respuesta: {excerpt}"
        )
    try:
        payload = json.loads(response.text)
    except ValueError as exc:
        raise RuntimeError(
            f"WoSIS/ISRIC no devolvio GeoJSON valido para {property_key}, "
            f"tesela {tile_bounds}, {content_type}. Respuesta: {excerpt}"
        ) from exc

    if payload.get("type") != "FeatureCollection" or "features" not in payload:
        raise RuntimeError(
            f"WoSIS/ISRIC devolvio una estructura inesperada para {property_key}, "
            f"tesela {tile_bounds}. Respuesta: {str(payload)[:500]}"
        )
    return payload


def request_wosis_tile(requests_client, property_key, tile_bounds):
    layer_name = WOSIS_PROPERTIES[property_key]
    params = {
        "map": "/map/wosis_latest.map",
        "SERVICE": "WFS",
        "VERSION": "1.0.0",
        "REQUEST": "GetFeature",
        "TYPENAME": layer_name,
        "OUTPUTFORMAT": "geojson",
        "SRSNAME": "EPSG:4326",
        "BBOX": ",".join(f"{value:.8f}" for value in tile_bounds),
    }
    last_error = None
    for attempt in range(1, WOSIS_REQUEST_RETRIES + 1):
        try:
            response = requests_client.get(
                WOSIS_WFS_URL,
                params=params,
                timeout=WOSIS_REQUEST_TIMEOUT_SECONDS,
            )
            return parse_wosis_geojson_response(response, property_key, tile_bounds)
        except Exception as exc:
            last_error = exc
            if attempt < WOSIS_REQUEST_RETRIES:
                time.sleep(1.5 * attempt)

    raise RuntimeError(
        f"No se pudo descargar WoSIS {property_key} despues de "
        f"{WOSIS_REQUEST_RETRIES} intentos. Detalle: {last_error}"
    ) from last_error


def request_wosis_tile_recursive(requests_client, property_key, tile_bounds, status_box=None):
    try:
        return [request_wosis_tile(requests_client, property_key, tile_bounds)]
    except RuntimeError:
        minx, miny, maxx, maxy = tile_bounds
        if max(maxx - minx, maxy - miny) <= WOSIS_MIN_TILE_SIZE_DEGREES:
            raise
        update_status(
            status_box,
            f"WoSIS {property_key} devolvio una respuesta no valida; "
            "reintentando con teselas menores...",
        )
        payloads = []
        for smaller_tile in split_bounds_quadrants(tile_bounds):
            payloads.extend(
                request_wosis_tile_recursive(
                    requests_client,
                    property_key,
                    smaller_tile,
                    status_box=status_box,
                )
            )
        return payloads


def fetch_wosis_property(requests_module, property_key, bounds, status_box=None):
    requests_client = requests_module.Session() if hasattr(requests_module, "Session") else requests_module
    tiles = list(split_bounds(bounds))
    frames = []
    for tile_index, tile_bounds in enumerate(tiles, start=1):
        update_status(
            status_box,
            f"Descargando WoSIS {property_key} {tile_index:,}/{len(tiles):,}...",
        )
        payloads = request_wosis_tile_recursive(
            requests_client,
            property_key,
            tile_bounds,
            status_box=status_box,
        )
        for payload in payloads:
            features = payload.get("features", [])
            if features:
                frames.append(gpd.GeoDataFrame.from_features(features, crs="EPSG:4326"))

    if not frames:
        return empty_wosis_property_gdf(property_key)

    combined = pd.concat(frames, ignore_index=True)
    if "layer_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["layer_id"])
    elif "profile_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["profile_id", "upper_depth", "lower_depth"])
    return gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")


def weighted_depth_average_for_property(gdf, property_key):
    if gdf.empty:
        return gpd.GeoDataFrame(columns=["profile_id", property_key, "geometry"], geometry="geometry", crs="EPSG:4326")

    df = gdf.copy()
    for column in ["profile_id", "upper_depth", "lower_depth", "value_avg"]:
        if column not in df.columns:
            raise RuntimeError(f"La capa WoSIS {property_key} no contiene el campo requerido `{column}`.")
    if "organic_surface" in df.columns:
        df = df[df["organic_surface"] != True]  # noqa: E712
    if "licence" in df.columns:
        licenses = df["licence"].fillna("").str.lower()
        public_license = (
            licenses.str.contains("creativecommons.org/licenses/by", regex=False)
            | licenses.str.contains("cc by", regex=False)
            | licenses.str.contains("public domain", regex=False)
        )
        df = df[public_license]

    df["upper_depth"] = pd.to_numeric(df["upper_depth"], errors="coerce")
    df["lower_depth"] = pd.to_numeric(df["lower_depth"], errors="coerce")
    df["value_avg"] = pd.to_numeric(df["value_avg"], errors="coerce")
    df = df.dropna(subset=["profile_id", "upper_depth", "lower_depth", "value_avg", "geometry"])
    df = df[(df["value_avg"] >= 0) & (df["value_avg"] <= 100)]
    df["overlap_cm"] = (
        np.minimum(df["lower_depth"], 30.0) - np.maximum(df["upper_depth"], 0.0)
    ).clip(lower=0)
    df = df[df["overlap_cm"] > 0]
    if df.empty:
        return gpd.GeoDataFrame(columns=["profile_id", property_key, "geometry"], geometry="geometry", crs="EPSG:4326")

    records = []
    for profile_id, group in df.groupby("profile_id", dropna=True):
        weights = group["overlap_cm"].to_numpy(dtype="float64")
        values = group["value_avg"].to_numpy(dtype="float64")
        if weights.sum() <= 0:
            continue
        records.append(
            {
                "profile_id": int(profile_id),
                property_key: float(np.average(values, weights=weights)),
                "geometry": group.geometry.iloc[0],
            }
        )

    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def normalize_training_fractions(samples):
    samples = samples.copy()
    for column in ["sand", "silt", "clay"]:
        samples[column] = pd.to_numeric(samples[column], errors="coerce")
    samples = samples.dropna(subset=["sand", "silt", "clay", "geometry"])
    total = samples[["sand", "silt", "clay"]].sum(axis=1)
    samples = samples[(total >= 60) & (total <= 140)].copy()
    if samples.empty:
        return samples
    total = samples[["sand", "silt", "clay"]].sum(axis=1)
    samples[["sand", "silt", "clay"]] = samples[["sand", "silt", "clay"]].div(total, axis=0) * 100.0
    return samples.reset_index(drop=True)


def fetch_wosis_texture_samples(poly_geom, status_box=None, require_minimum=True):
    requests_module = require_experimental_dependencies()["requests"]
    bounds = expanded_wgs84_bounds(poly_geom, SENTINEL_WOSIS_BUFFER_KM)
    update_status(
        status_box,
        "Descargando observaciones reales WoSIS de arena/limo/arcilla "
        f"en un buffer de {SENTINEL_WOSIS_BUFFER_KM} km...",
    )

    property_tables = {}
    for property_key in WOSIS_PROPERTIES:
        raw = fetch_wosis_property(requests_module, property_key, bounds, status_box=status_box)
        property_tables[property_key] = weighted_depth_average_for_property(raw, property_key)

    samples = property_tables["sand"][["profile_id", "sand", "geometry"]]
    for property_key in ["silt", "clay"]:
        samples = samples.merge(
            property_tables[property_key][["profile_id", property_key]],
            on="profile_id",
            how="inner",
        )

    if samples.empty:
        if require_minimum:
            raise RuntimeError(
                "WoSIS no devolvio perfiles con arena, limo y arcilla 0-30 cm "
                "para el area de entrenamiento. El modelo experimental se detiene."
            )
        return gpd.GeoDataFrame(
            columns=["profile_id", "sand", "silt", "clay", "source", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    samples = gpd.GeoDataFrame(samples, geometry="geometry", crs="EPSG:4326")
    samples = normalize_training_fractions(samples)
    samples["source"] = WOSIS_SOURCE_LABEL

    if require_minimum and len(samples) < SENTINEL_MIN_WOSIS_SAMPLES:
        raise RuntimeError(
            "WoSIS devolvio muy pocas muestras completas para entrenar "
            f"({len(samples)}; minimo {SENTINEL_MIN_WOSIS_SAMPLES}). "
            "Amplia el poligono, combina con calicatas Costa Rica o usa OpenLandMap/SoilGrids."
        )
    return samples.reset_index(drop=True)


def parse_decimal_series(series):
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def find_calicatas_column(columns, candidates):
    normalized = {str(column).strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for column in columns:
        lowered = str(column).strip().lower()
        for candidate in candidates:
            if candidate in lowered:
                return column
    raise RuntimeError(
        "No se encontro una columna requerida en el CSV de calicatas Costa Rica. "
        f"Candidatos: {', '.join(candidates)}"
    )


def load_calicatas_cr_texture_samples(poly_geom, status_box=None, require_minimum=True):
    if not CALICATAS_CR_CSV_PATH.exists():
        raise RuntimeError(
            "No se encontro el archivo de calicatas de Costa Rica "
            f"({CALICATAS_CR_CSV_PATH.name}). Colocalo en la raiz del proyecto."
        )

    update_status(
        status_box,
        "Cargando calicatas Costa Rica con arena/limo/arcilla 0-30 cm "
        f"en un buffer de {SENTINEL_WOSIS_BUFFER_KM} km...",
    )
    raw = pd.read_csv(
        CALICATAS_CR_CSV_PATH,
        sep=";",
        encoding="utf-8",
        low_memory=False,
    )
    id_col = find_calicatas_column(raw.columns, ["id_calicata"])
    lon_col = find_calicatas_column(raw.columns, ["longitud"])
    lat_col = find_calicatas_column(raw.columns, ["latitud"])
    upper_col = find_calicatas_column(raw.columns, ["cmprof. inicial", "prof. inicial"])
    lower_col = find_calicatas_column(raw.columns, ["prof. final"])
    sand_col = find_calicatas_column(raw.columns, ["% areana", "% arena", "areana", "arena"])
    silt_col = find_calicatas_column(raw.columns, ["limo"])
    clay_col = find_calicatas_column(raw.columns, ["arcilla"])
    texture_col = None
    try:
        texture_col = find_calicatas_column(raw.columns, ["clase textural"])
    except RuntimeError:
        texture_col = None

    work = pd.DataFrame(
        {
            "calicata_id": pd.to_numeric(raw[id_col], errors="coerce"),
            "lon": parse_decimal_series(raw[lon_col]),
            "lat": parse_decimal_series(raw[lat_col]),
            "upper_depth": parse_decimal_series(raw[upper_col]),
            "lower_depth": parse_decimal_series(raw[lower_col]),
            "sand": parse_decimal_series(raw[sand_col]),
            "silt": parse_decimal_series(raw[silt_col]),
            "clay": parse_decimal_series(raw[clay_col]),
        }
    )
    if texture_col is not None:
        work["texture_label"] = raw[texture_col].astype(str).str.strip()

    work = work.dropna(
        subset=["calicata_id", "lon", "lat", "upper_depth", "lower_depth", "sand", "silt", "clay"]
    )
    work = work[(work["sand"] >= 0) & (work["silt"] >= 0) & (work["clay"] >= 0)]
    work = work[(work["sand"] <= 100) & (work["silt"] <= 100) & (work["clay"] <= 100)]
    work["overlap_cm"] = (
        np.minimum(work["lower_depth"], 30.0) - np.maximum(work["upper_depth"], 0.0)
    ).clip(lower=0)
    work = work[work["overlap_cm"] > 0]
    if work.empty:
        if require_minimum:
            raise RuntimeError(
                "El CSV de calicatas Costa Rica no contiene horizontes con arena/limo/arcilla "
                "que solapen 0-30 cm."
            )
        return gpd.GeoDataFrame(
            columns=["profile_id", "sand", "silt", "clay", "source", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    records = []
    for calicata_id, group in work.groupby("calicata_id", dropna=True):
        weights = group["overlap_cm"].to_numpy(dtype="float64")
        if weights.sum() <= 0:
            continue
        record = {
            "profile_id": int(CALICATAS_CR_PROFILE_ID_OFFSET + int(calicata_id)),
            "sand": float(np.average(group["sand"], weights=weights)),
            "silt": float(np.average(group["silt"], weights=weights)),
            "clay": float(np.average(group["clay"], weights=weights)),
            "source": CALICATAS_CR_SOURCE_LABEL,
            "geometry": Point(float(group["lon"].iloc[0]), float(group["lat"].iloc[0])),
        }
        if "texture_label" in group.columns:
            labels = group["texture_label"].dropna()
            labels = labels[labels.astype(str).str.strip() != ""]
            if not labels.empty:
                record["texture_label"] = str(labels.mode().iloc[0])
        records.append(record)

    if not records:
        if require_minimum:
            raise RuntimeError(
                "No se pudieron agregar perfiles 0-30 cm desde calicatas Costa Rica."
            )
        return gpd.GeoDataFrame(
            columns=["profile_id", "sand", "silt", "clay", "source", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    samples = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    samples = normalize_training_fractions(samples)

    bounds = expanded_wgs84_bounds(poly_geom, SENTINEL_WOSIS_BUFFER_KM)
    train_area = box(*bounds)
    samples = samples[samples.geometry.within(train_area) | samples.geometry.intersects(train_area)]
    samples = samples.reset_index(drop=True)

    update_status(
        status_box,
        f"Calicatas Costa Rica listas: {len(samples):,} perfiles 0-30 cm "
        f"dentro del buffer de {SENTINEL_WOSIS_BUFFER_KM} km.",
    )

    if require_minimum and len(samples) < SENTINEL_MIN_WOSIS_SAMPLES:
        raise RuntimeError(
            "Calicatas Costa Rica devolvio muy pocas muestras completas para entrenar "
            f"({len(samples)}; minimo {SENTINEL_MIN_WOSIS_SAMPLES}). "
            "Amplia el poligono, combina con WoSIS o usa OpenLandMap/SoilGrids."
        )
    return samples


def combine_training_samples(frames):
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return gpd.GeoDataFrame(
            columns=["profile_id", "sand", "silt", "clay", "source", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
    combined = pd.concat(valid_frames, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
    combined = combined.drop_duplicates(subset=["profile_id"], keep="first")
    return combined.reset_index(drop=True)


def limit_training_samples_near_polygon(samples, poly_geom, max_samples, status_box=None):
    if samples.empty or max_samples <= 0 or len(samples) <= max_samples:
        return samples.reset_index(drop=True)

    target_crs = estimate_area_crs_for_geometry(poly_geom)
    samples_metric = samples.to_crs(target_crs)
    centroid_metric = project_geometry(poly_geom, target_crs).centroid
    ranked = samples.copy().reset_index(drop=True)
    ranked["_distance_to_aoi"] = samples_metric.geometry.distance(centroid_metric).to_numpy()
    ranked = ranked.sort_values(["_distance_to_aoi", "profile_id"], kind="mergesort")

    if "source" in ranked.columns and ranked["source"].nunique() > 1:
        selected_ids = []
        source_groups = {
            source: group for source, group in ranked.groupby("source", sort=False)
        }
        base_quota = max(1, max_samples // len(source_groups))
        for group in source_groups.values():
            selected_ids.extend(group.head(min(len(group), base_quota))["profile_id"].tolist())
        selected = ranked[ranked["profile_id"].isin(selected_ids)]
        if len(selected) < max_samples:
            extra = ranked[~ranked["profile_id"].isin(selected_ids)].head(
                max_samples - len(selected)
            )
            selected = pd.concat([selected, extra], ignore_index=True)
        selected = selected.head(max_samples)
    else:
        selected = ranked.head(max_samples)

    selected = gpd.GeoDataFrame(selected, geometry="geometry", crs=samples.crs)
    selected = selected.drop(columns=["_distance_to_aoi"], errors="ignore")
    update_status(
        status_box,
        "Se limitaron los perfiles de entrenamiento a los "
        f"{len(selected):,} mas cercanos al poligono "
        f"(de {len(samples):,} disponibles) para acotar Sentinel-2.",
    )
    return selected.reset_index(drop=True)


def fetch_training_texture_samples(
    poly_geom,
    training_source=DEFAULT_TRAINING_SOURCE,
    status_box=None,
):
    if training_source not in TRAINING_SOURCE_OPTIONS:
        raise ValueError(
            f"Fuente de entrenamiento no soportada: {training_source}. "
            f"Opciones: {', '.join(TRAINING_SOURCE_OPTIONS)}"
        )

    use_wosis = training_source in {TRAINING_SOURCE_WOSIS, TRAINING_SOURCE_BOTH}
    use_calicatas = training_source in {
        TRAINING_SOURCE_CALICATAS_CR,
        TRAINING_SOURCE_BOTH,
    }
    require_each = training_source != TRAINING_SOURCE_BOTH

    frames = []
    if use_wosis:
        try:
            frames.append(
                fetch_wosis_texture_samples(
                    poly_geom,
                    status_box=status_box,
                    require_minimum=require_each,
                )
            )
        except Exception as exc:
            if require_each:
                raise
            update_status(
                status_box,
                "WoSIS no estuvo disponible; se continuara solo con calicatas Costa Rica. "
                f"Detalle: {exc}",
            )
    if use_calicatas:
        try:
            frames.append(
                load_calicatas_cr_texture_samples(
                    poly_geom,
                    status_box=status_box,
                    require_minimum=require_each,
                )
            )
        except Exception as exc:
            if require_each:
                raise
            update_status(
                status_box,
                "Calicatas Costa Rica no estuvieron disponibles; se continuara solo con WoSIS. "
                f"Detalle: {exc}",
            )

    samples = combine_training_samples(frames)
    if samples.empty:
        raise RuntimeError(
            "No hay perfiles de entrenamiento con arena, limo y arcilla 0-30 cm "
            f"para la fuente `{TRAINING_SOURCE_OPTIONS[training_source]}`. "
            "El modelo experimental se detiene."
        )
    samples = limit_training_samples_near_polygon(
        samples,
        poly_geom,
        SENTINEL_MAX_TRAINING_PROFILES,
        status_box=status_box,
    )
    if len(samples) < SENTINEL_MIN_WOSIS_SAMPLES:
        raise RuntimeError(
            "Hay muy pocas muestras completas para entrenar "
            f"({len(samples)}; minimo {SENTINEL_MIN_WOSIS_SAMPLES}) con "
            f"`{TRAINING_SOURCE_OPTIONS[training_source]}`. "
            "Amplia el poligono o usa OpenLandMap/SoilGrids."
        )

    source_counts = samples["source"].value_counts().to_dict() if "source" in samples.columns else {}
    update_status(
        status_box,
        "Perfiles de entrenamiento listos: "
        f"{len(samples):,} totales"
        + (
            " ("
            + ", ".join(f"{label}={count:,}" for label, count in sorted(source_counts.items()))
            + ")"
            if source_counts
            else ""
        )
        + ".",
    )
    return samples.reset_index(drop=True)


def sentinel_item_cloud_cover(item):
    try:
        return float(item.properties.get("eo:cloud_cover", np.inf))
    except (TypeError, ValueError):
        return np.inf


def sentinel_item_datetime(item):
    return item.properties.get("datetime") or ""


def sentinel_item_month(item):
    value = sentinel_item_datetime(item)
    if not value:
        return None
    try:
        # STAC datetimes look like 2023-01-15T16:12:34.000000Z
        return int(str(value)[5:7])
    except (TypeError, ValueError):
        return None


def sentinel_item_is_dry_season(item):
    month = sentinel_item_month(item)
    return month in SENTINEL_DRY_SEASON_MONTHS if month is not None else False


def order_items_dry_season_first(items):
    """Prefer dry-season scenes, then lower cloud cover, then newer datetime."""
    return sorted(
        items,
        key=lambda item: (
            0 if sentinel_item_is_dry_season(item) else 1,
            sentinel_item_cloud_cover(item),
            # Newer scenes first within the same season/cloud bucket.
            "" if not sentinel_item_datetime(item) else sentinel_item_datetime(item),
        ),
        reverse=False,
    )


def sentinel_item_group_key(item):
    return (
        item.properties.get("s2:mgrs_tile")
        or item.properties.get("grid:code")
        or item.id.rsplit("_", 1)[-1]
    )


def select_sentinel_items(items, max_items, status_box=None, purpose="modelo"):
    if not items:
        return []
    ordered = order_items_dry_season_first(items)
    if not max_items or len(ordered) <= max_items:
        dry_count = sum(1 for item in ordered if sentinel_item_is_dry_season(item))
        update_status(
            status_box,
            "Sentinel-2: "
            f"{len(ordered):,} escenas para {purpose} "
            f"({dry_count:,} en estacion seca {SENTINEL_DRY_SEASON_MONTHS}).",
        )
        return ordered

    selected = ordered[:max_items]
    dry_count = sum(1 for item in selected if sentinel_item_is_dry_season(item))
    update_status(
        status_box,
        "Sentinel-2 devolvio "
        f"{len(items):,} escenas para {purpose}; se usaran las "
        f"{len(selected):,} priorizando estacion seca y menor nubosidad "
        f"({dry_count:,} en meses {SENTINEL_DRY_SEASON_MONTHS}).",
    )
    return selected


def sentinel_item_coverage_count(item, point_geometries):
    try:
        item_geometry = shape(item.geometry)
    except Exception:
        return 0
    return int(point_geometries.intersects(item_geometry).sum())


def select_training_sentinel_items(items, samples, max_items, status_box=None):
    if not max_items or len(items) <= max_items:
        return order_items_dry_season_first(list(items))

    point_geometries = samples.geometry
    groups = {}
    ignored_items = 0
    for item in items:
        coverage_count = sentinel_item_coverage_count(item, point_geometries)
        if coverage_count <= 0:
            ignored_items += 1
            continue
        group_key = sentinel_item_group_key(item)
        groups.setdefault(group_key, []).append((coverage_count, item))

    if not groups:
        update_status(
            status_box,
            "No se pudo calcular cobertura espacial de escenas Sentinel-2; "
            "se usaran escenas priorizando estacion seca y menor nubosidad.",
        )
        return select_sentinel_items(
            items,
            max_items,
            status_box=status_box,
            purpose="entrenamiento de perfiles",
        )

    for group_items in groups.values():
        group_items.sort(
            key=lambda record: (
                -record[0],
                0 if sentinel_item_is_dry_season(record[1]) else 1,
                sentinel_item_cloud_cover(record[1]),
                sentinel_item_datetime(record[1]),
            )
        )

    selected = []
    ordered_group_keys = sorted(
        groups,
        key=lambda group_key: (
            -max(record[0] for record in groups[group_key]),
            group_key,
        ),
    )
    while len(selected) < max_items and ordered_group_keys:
        next_keys = []
        for group_key in ordered_group_keys:
            if not groups[group_key]:
                continue
            selected.append(groups[group_key].pop(0)[1])
            if len(selected) >= max_items:
                break
            if groups[group_key]:
                next_keys.append(group_key)
        ordered_group_keys = next_keys

    selected = order_items_dry_season_first(selected)
    dry_count = sum(1 for item in selected if sentinel_item_is_dry_season(item))
    update_status(
        status_box,
        "Sentinel-2 devolvio "
        f"{len(items):,} escenas para entrenamiento; se usaran "
        f"{len(selected):,} escenas que cubren perfiles "
        f"({dry_count:,} en estacion seca {SENTINEL_DRY_SEASON_MONTHS}) en "
        f"{len(groups):,} tiles. Escenas sin perfiles cubiertos: {ignored_items:,}.",
    )
    return selected


def raise_if_too_many_sentinel_failures(failed_items, phase, exc):
    if len(failed_items) < SENTINEL_MAX_ITEM_FAILURES:
        return
    item_id, detail = failed_items[-1]
    hint = ""
    detail_text = str(detail)
    if "blob.core.windows.net" in detail_text and "not recognized as being in a supported file format" in detail_text:
        hint = (
            " Esto puede ocurrir cuando vence una firma temporal de Planetary "
            "Computer y el servidor devuelve una respuesta de error en vez del GeoTIFF."
        )
    raise RuntimeError(
        f"Demasiadas escenas Sentinel-2 fallaron durante {phase} "
        f"({len(failed_items)} fallos). Ultima escena: {item_id}. "
        f"Detalle tecnico: {detail}{hint}"
    ) from exc


def search_sentinel_items(bounds, status_box=None, purpose="modelo", max_items=None):
    modules = require_experimental_dependencies()
    pystac_client = modules["pystac_client"]

    datetime_range = sentinel_datetime_range()
    max_items_message = (
        f"; se priorizaran hasta {max_items:,} escenas de menor nubosidad"
        if max_items
        else ""
    )
    update_status(
        status_box,
        "Buscando escenas Sentinel-2 L2A disponibles para "
        f"{purpose} ({datetime_range}, nubosidad escena < "
        f"{SENTINEL_CLOUD_COVER_LT}%{max_items_message})...",
    )
    catalog = pystac_client.Client.open(SENTINEL_STAC_URL)
    search = catalog.search(
        collections=[SENTINEL_COLLECTION],
        bbox=list(bounds),
        datetime=datetime_range,
        query={"eo:cloud_cover": {"lt": SENTINEL_CLOUD_COVER_LT}},
    )
    items = list(search.items())
    items.sort(key=lambda item: item.properties.get("datetime") or "")
    if not items:
        raise RuntimeError(
            "Planetary Computer no devolvio escenas Sentinel-2 L2A para el area. "
            "El modelo experimental se detiene."
        )
    return select_sentinel_items(items, max_items, status_box, purpose)


def unsigned_asset_href(href):
    parsed = urlsplit(href)
    query = parse_qs(parsed.query)
    if {"st", "se", "sp"} & set(query):
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", parsed.fragment))
    return href


def signed_url_expires_at(href):
    expires_values = parse_qs(urlsplit(href).query).get("se")
    if not expires_values:
        return None
    try:
        return datetime.fromisoformat(expires_values[0].replace("Z", "+00:00"))
    except ValueError:
        return None


def signed_url_has_enough_ttl(href):
    expires_at = signed_url_expires_at(href)
    if not expires_at:
        return True
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    ttl_seconds = (expires_at - datetime.now(timezone.utc)).total_seconds()
    return ttl_seconds > SENTINEL_SIGNED_URL_MIN_TTL_SECONDS


def clear_planetary_computer_token_cache(planetary_computer):
    sas_module = getattr(planetary_computer, "sas", None)
    if sas_module is None:
        try:
            sas_module = __import__("planetary_computer.sas", fromlist=["TOKEN_CACHE"])
        except ImportError:
            return
    token_cache = getattr(sas_module, "TOKEN_CACHE", None)
    if hasattr(token_cache, "clear"):
        token_cache.clear()


def sign_planetary_computer_href(planetary_computer, href):
    sign_url = getattr(planetary_computer, "sign_url", None)
    if callable(sign_url):
        signed = sign_url(href)
        return getattr(signed, "href", signed)
    sign = getattr(planetary_computer, "sign", None)
    if callable(sign):
        signed = sign(href)
        return getattr(signed, "href", signed)
    raise RuntimeError(
        "La dependencia planetary-computer instalada no expone una funcion "
        "compatible para firmar assets Sentinel-2."
    )


def sign_asset_href(href):
    planetary_computer = require_experimental_dependencies()["planetary_computer"]
    unsigned_href = unsigned_asset_href(href)
    signed_href = sign_planetary_computer_href(planetary_computer, unsigned_href)
    if signed_url_has_enough_ttl(signed_href):
        return signed_href

    clear_planetary_computer_token_cache(planetary_computer)
    signed_href = sign_planetary_computer_href(planetary_computer, unsigned_href)
    if signed_url_has_enough_ttl(signed_href):
        return signed_href

    expires_at = signed_url_expires_at(signed_href)
    raise RuntimeError(
        "Planetary Computer devolvio una firma Sentinel-2 con vigencia "
        f"insuficiente (vence: {expires_at}). Reintenta en unos minutos."
    )


def item_asset_href(item, logical_name):
    for asset_name in SENTINEL_ASSET_ALIASES[logical_name]:
        if asset_name in item.assets:
            return sign_asset_href(item.assets[asset_name].href)
    raise RuntimeError(f"La escena Sentinel-2 {item.id} no contiene el asset {logical_name}.")


def read_asset_grid(href, target_crs, transform, width, height, resampling):
    from rasterio.vrt import WarpedVRT

    with rasterio.open(href) as src:
        with WarpedVRT(
            src,
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            resampling=resampling,
            nodata=src.nodata,
        ) as vrt:
            return vrt.read(1, masked=True)


def reflectance(values):
    array = np.ma.filled(values, np.nan).astype("float32")
    array = np.where(array > 1.5, array / 10000.0, array)
    return np.where((array >= 0) & (array <= 1.5), array, np.nan).astype("float32")


def safe_ratio(numerator, denominator):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(denominator) > 1e-6, numerator / denominator, np.nan).astype("float32")


def read_item_reflectance_bands_grid(item, grid, resampling):
    return {
        logical_name: reflectance(
            read_asset_grid(
                item_asset_href(item, logical_name),
                grid["crs"],
                grid["transform"],
                grid["width"],
                grid["height"],
                resampling,
            )
        )
        for logical_name in SENTINEL_REFLECTANCE_BANDS
    }


def read_item_reflectance_bands_samples(item, samples):
    return {
        logical_name: reflectance(
            read_asset_samples(item_asset_href(item, logical_name), samples)
        )
        for logical_name in SENTINEL_REFLECTANCE_BANDS
    }


def derive_sentinel_features(bands):
    blue = bands["blue"]
    green = bands["green"]
    red = bands["red"]
    rededge1 = bands["rededge1"]
    rededge2 = bands["rededge2"]
    rededge3 = bands["rededge3"]
    nir = bands["nir"]
    nir08 = bands["nir08"]
    swir1 = bands["swir1"]
    swir2 = bands["swir2"]

    ndvi = safe_ratio(nir - red, nir + red)
    savi = safe_ratio(1.5 * (nir - red), nir + red + 0.5)
    msavi = (2 * nir + 1 - np.sqrt(np.maximum((2 * nir + 1) ** 2 - 8 * (nir - red), 0))) / 2
    bsi = safe_ratio((swir1 + red) - (nir + blue), (swir1 + red) + (nir + blue))
    ci = safe_ratio(swir1, swir2)
    ndwi = safe_ratio(green - nir, green + nir)
    geoi = safe_ratio(swir1 - swir2, swir1 + swir2)
    bi = np.sqrt(np.maximum(blue**2 + green**2 + red**2, 0)).astype("float32")
    # Red-edge indices (B8A narrow NIR vs B05/B06) for residual vegetation / soil contrast.
    ndre = safe_ratio(nir08 - rededge1, nir08 + rededge1)
    ndre2 = safe_ratio(nir08 - rededge2, nir08 + rededge2)

    return {
        "B02": blue,
        "B03": green,
        "B04": red,
        "B05": rededge1,
        "B06": rededge2,
        "B07": rededge3,
        "B08": nir,
        "B8A": nir08,
        "B11": swir1,
        "B12": swir2,
        "NDVI": ndvi,
        "SAVI": savi.astype("float32"),
        "MSAVI": msavi.astype("float32"),
        "BSI": bsi,
        "CI": ci,
        "NDWI": ndwi,
        "GEOI": geoi,
        "BI": bi,
        "NDRE": ndre,
        "NDRE2": ndre2,
    }


def bare_soil_mask_and_score(features, scl, dry_season=False):
    valid_features = np.ones_like(features["NDVI"], dtype=bool)
    for feature_name in SENTINEL_SPECTRAL_FEATURE_NAMES:
        valid_features &= np.isfinite(features[feature_name])

    not_vegetated_scl = scl == 5
    strict_bare = (
        valid_features
        & not_vegetated_scl
        & (features["NDVI"] >= SENTINEL_BARE_NDVI_MIN)
        & (features["NDVI"] <= SENTINEL_BARE_SCL5_NDVI_MAX)
        & (features["NDWI"] < SENTINEL_BARE_SCL5_NDWI_MAX)
        & (features["B04"] > SENTINEL_BARE_SCL5_RED_MIN)
        & (features["BSI"] > SENTINEL_BARE_SCL5_BSI_MIN)
    )
    spectral_bare = (
        valid_features
        & (scl == 7)
        & (features["NDVI"] >= SENTINEL_BARE_NDVI_MIN)
        & (features["NDVI"] <= SENTINEL_BARE_SCL7_NDVI_MAX)
        & (features["NDWI"] < SENTINEL_BARE_SCL7_NDWI_MAX)
        & (features["B04"] > SENTINEL_BARE_SCL7_RED_MIN)
        & (features["BSI"] > SENTINEL_BARE_SCL7_BSI_MIN)
    )
    bare_mask = strict_bare | spectral_bare
    score = (
        features["BSI"]
        - np.abs(features["NDVI"]) * 0.75
        - np.maximum(features["NDWI"], 0) * 0.45
        + features["BI"] * 0.03
        + (SENTINEL_DRY_SEASON_SCORE_BONUS if dry_season else 0.0)
    ).astype("float32")
    return bare_mask, score


def empty_spectral_accumulators(shape):
    best_features = {
        feature_name: np.full(shape, np.nan, dtype="float32")
        for feature_name in SENTINEL_SPECTRAL_FEATURE_NAMES
    }
    sum_features = {
        feature_name: np.zeros(shape, dtype="float32")
        for feature_name in SENTINEL_SPECTRAL_FEATURE_NAMES
    }
    return best_features, sum_features


def accumulate_bare_spectral_features(
    best_features,
    sum_features,
    bare_count,
    best_score,
    features,
    bare_mask,
    score,
):
    if not np.any(bare_mask):
        return

    bare_count[bare_mask] = np.minimum(
        bare_count[bare_mask] + 1,
        np.iinfo("uint16").max,
    )
    for feature_name in SENTINEL_SPECTRAL_FEATURE_NAMES:
        values = features[feature_name]
        sum_features[feature_name][bare_mask] += values[bare_mask]

    update_mask = bare_mask & (score > best_score)
    if np.any(update_mask):
        best_score[update_mask] = score[update_mask]
        for feature_name in SENTINEL_SPECTRAL_FEATURE_NAMES:
            best_features[feature_name][update_mask] = features[feature_name][update_mask]


def model_feature_arrays_from_accumulators(
    best_features,
    sum_features,
    bare_count,
    best_score,
    valid_mask,
    lon=None,
    lat=None,
):
    count = bare_count.astype("float32")
    has_observations = valid_mask & (bare_count > 0)
    feature_arrays = {}

    for feature_name in SENTINEL_SPECTRAL_FEATURE_NAMES:
        best_key = f"{feature_name}_best"
        mean_key = f"{feature_name}_mean"
        feature_arrays[best_key] = np.where(
            valid_mask,
            best_features[feature_name],
            np.nan,
        ).astype("float32")
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_values = sum_features[feature_name] / count
        feature_arrays[mean_key] = np.where(
            has_observations,
            mean_values,
            np.nan,
        ).astype("float32")

    feature_arrays["BARE_OBS"] = np.where(has_observations, count, np.nan).astype("float32")
    feature_arrays["BARE_SCORE"] = np.where(valid_mask, best_score, np.nan).astype("float32")
    return feature_arrays


def attach_dem_features(feature_arrays, dem_features, valid_mask=None):
    for feature_name, values in dem_features.items():
        array = np.asarray(values, dtype="float32")
        if valid_mask is not None:
            array = np.where(valid_mask, array, np.nan).astype("float32")
        feature_arrays[feature_name] = array
    return feature_arrays


def grid_wgs84_coordinate_arrays(grid):
    from rasterio.warp import transform as transform_coordinates

    shape = (grid["height"], grid["width"])
    lon = np.full(shape, np.nan, dtype="float32")
    lat = np.full(shape, np.nan, dtype="float32")
    cols = np.arange(grid["width"], dtype="float64") + 0.5
    chunk_rows = 512

    for row_start in range(0, grid["height"], chunk_rows):
        row_end = min(row_start + chunk_rows, grid["height"])
        rows = np.arange(row_start, row_end, dtype="float64") + 0.5
        col_grid, row_grid = np.meshgrid(cols, rows)
        xs = (
            grid["transform"].c
            + grid["transform"].a * col_grid
            + grid["transform"].b * row_grid
        )
        ys = (
            grid["transform"].f
            + grid["transform"].d * col_grid
            + grid["transform"].e * row_grid
        )
        lons, lats = transform_coordinates(
            grid["crs"],
            "EPSG:4326",
            xs.ravel().tolist(),
            ys.ravel().tolist(),
        )
        lon[row_start:row_end, :] = np.asarray(lons, dtype="float32").reshape(row_end - row_start, grid["width"])
        lat[row_start:row_end, :] = np.asarray(lats, dtype="float32").reshape(row_end - row_start, grid["width"])

    return lon, lat


def mean_bare_observations(bare_count, valid_mask):
    valid_counts = bare_count[valid_mask]
    if valid_counts.size == 0:
        return 0.0
    return float(np.mean(valid_counts))


def build_sentinel_bare_soil_composite(poly_geom, status_box=None, max_pixels=2_500_000):
    from rasterio.enums import Resampling

    grid = build_prediction_grid(poly_geom, max_pixels)
    bounds = tuple(float(value) for value in poly_geom.bounds)
    items = search_sentinel_items(
        bounds,
        status_box,
        purpose="prediccion del poligono",
        max_items=SENTINEL_MAX_PREDICTION_ITEMS,
    )

    shape = (grid["height"], grid["width"])
    best_score = np.full(shape, -np.inf, dtype="float32")
    bare_count = np.zeros(shape, dtype="uint16")
    best_features, sum_features = empty_spectral_accumulators(shape)
    failed_items = []
    processed_items = 0
    dry_season_items_used = 0

    with rasterio.Env(**SENTINEL_GDAL_OPTIONS):
        for index, item in enumerate(items, start=1):
            bare_pixels = int(np.count_nonzero(np.isfinite(best_score) & grid["mask"]))
            bare_percent = bare_pixels / max(np.count_nonzero(grid["mask"]), 1) * 100.0
            mean_observations = mean_bare_observations(bare_count, np.isfinite(best_score) & grid["mask"])
            update_status(
                status_box,
                "Componiendo suelo descubierto Sentinel-2 "
                f"{index:,}/{len(items):,} escenas; pixeles con suelo descubierto "
                f"{bare_percent:.1f}%; media de observaciones {mean_observations:.1f} "
                f"(prioridad estacion seca {SENTINEL_DRY_SEASON_MONTHS}).",
            )
            try:
                bands = read_item_reflectance_bands_grid(item, grid, Resampling.bilinear)
                scl = np.ma.filled(
                    read_asset_grid(
                        item_asset_href(item, "scl"),
                        grid["crs"],
                        grid["transform"],
                        grid["width"],
                        grid["height"],
                        Resampling.nearest,
                    ),
                    -1,
                ).astype("int16")
            except Exception as exc:
                failed_items.append((item.id, str(exc)))
                update_status(
                    status_box,
                    "Se omitio una escena Sentinel-2 por error de lectura "
                    f"({len(failed_items):,}/{SENTINEL_MAX_ITEM_FAILURES:,} fallos).",
                )
                raise_if_too_many_sentinel_failures(failed_items, "la prediccion", exc)
                continue

            processed_items += 1
            dry_season = sentinel_item_is_dry_season(item)
            if dry_season:
                dry_season_items_used += 1
            features = derive_sentinel_features(bands)
            bare_mask, score = bare_soil_mask_and_score(
                features,
                scl,
                dry_season=dry_season,
            )
            bare_mask &= grid["mask"]
            accumulate_bare_spectral_features(
                best_features,
                sum_features,
                bare_count,
                best_score,
                features,
                bare_mask,
                score,
            )

            bare_pixels = int(np.count_nonzero(np.isfinite(best_score) & grid["mask"]))
            bare_percent = bare_pixels / max(np.count_nonzero(grid["mask"]), 1) * 100.0
            mean_observations = mean_bare_observations(bare_count, np.isfinite(best_score) & grid["mask"])
            if (
                processed_items >= SENTINEL_MIN_PREDICTION_ITEMS
                and bare_percent >= SENTINEL_TARGET_BARE_PIXEL_PERCENT
                and mean_observations >= SENTINEL_TARGET_MEAN_BARE_OBSERVATIONS
            ):
                update_status(
                    status_box,
                    "Compuesto Sentinel-2 completo con "
                    f"{bare_percent:.1f}% de pixeles de suelo descubierto y "
                    f"{mean_observations:.1f} observaciones medias por pixel.",
                )
                break

    valid_bare = np.isfinite(best_score) & grid["mask"]
    if not np.any(valid_bare):
        raise RuntimeError(
            "No se detectaron pixeles de suelo descubierto Sentinel-2 dentro del poligono. "
            "El modelo experimental se detiene."
        )
    feature_arrays = model_feature_arrays_from_accumulators(
        best_features,
        sum_features,
        bare_count,
        best_score,
        valid_bare,
    )
    from dem_covariates import terrain_feature_arrays_for_grid

    dem_features, dem_meta = terrain_feature_arrays_for_grid(
        poly_geom,
        grid,
        status_callback=lambda message: update_status(status_box, message),
    )
    feature_arrays = attach_dem_features(feature_arrays, dem_features, valid_mask=valid_bare)

    from s1_covariates import s1_feature_arrays_for_grid

    s1_features, s1_meta = s1_feature_arrays_for_grid(
        poly_geom,
        grid,
        status_callback=lambda message: update_status(status_box, message),
    )
    feature_arrays = attach_dem_features(feature_arrays, s1_features, valid_mask=valid_bare)
    s1_ok = (
        np.isfinite(feature_arrays["VV_DB"])
        & np.isfinite(feature_arrays["VH_DB"])
        & np.isfinite(feature_arrays["VV_VH_DB"])
    )
    valid_bare = valid_bare & s1_ok
    if not np.any(valid_bare):
        raise RuntimeError(
            "Hay suelo descubierto Sentinel-2, pero no hay cobertura Sentinel-1 RTC "
            "valida (VV/VH) dentro del poligono. El modelo experimental se detiene."
        )
    for feature_name in list(feature_arrays):
        feature_arrays[feature_name] = np.where(
            valid_bare,
            feature_arrays[feature_name],
            np.nan,
        ).astype("float32")

    summary = {
        "sentinel_collection": SENTINEL_COLLECTION,
        "sentinel_datetime": sentinel_datetime_range(),
        "sentinel_items_used": int(processed_items),
        "sentinel_items_failed": int(len(failed_items)),
        "target_resolution_m": SENTINEL_TARGET_RESOLUTION_M,
        "cloud_cover_scene_threshold_percent": SENTINEL_CLOUD_COVER_LT,
        "aoi_pixels": int(np.count_nonzero(grid["mask"])),
        "bare_soil_pixels": int(np.count_nonzero(valid_bare)),
        "bare_soil_pixels_percent": round(
            np.count_nonzero(valid_bare) / max(np.count_nonzero(grid["mask"]), 1) * 100.0,
            2,
        ),
        "mean_bare_observations_per_prediction_pixel": round(
            mean_bare_observations(bare_count, valid_bare),
            2,
        ),
        "bare_soil_policy": "strict_scl5_fallback_scl7_dry_season_priority",
        "dry_season_months": list(SENTINEL_DRY_SEASON_MONTHS),
        "dry_season_items_used": int(dry_season_items_used),
        "dry_season_items_fraction": round(
            dry_season_items_used / max(processed_items, 1),
            3,
        ),
        "bare_soil_ndvi_max_scl5": SENTINEL_BARE_SCL5_NDVI_MAX,
        "bare_soil_ndvi_max_scl7": SENTINEL_BARE_SCL7_NDVI_MAX,
        **dem_meta,
        **s1_meta,
    }
    return feature_arrays, valid_bare, bare_count, best_score, grid, summary


def read_asset_samples(href, points_wgs84):
    from rasterio.warp import transform as transform_coordinates

    lons = points_wgs84.geometry.x.to_numpy()
    lats = points_wgs84.geometry.y.to_numpy()
    with rasterio.open(href) as src:
        xs, ys = transform_coordinates("EPSG:4326", src.crs, lons.tolist(), lats.tolist())
        sampled = []
        for value in src.sample(zip(xs, ys), masked=True):
            if len(value) == 0 or np.any(np.ma.getmaskarray(value)):
                sampled.append(np.nan)
            else:
                sample_value = float(value[0])
                sampled.append(sample_value if np.isfinite(sample_value) else np.nan)
    return np.asarray(sampled, dtype="float32")


def extract_training_features_from_sentinel(samples, status_box=None, poly_geom=None):
    bounds = tuple(float(value) for value in samples.total_bounds)
    items = search_sentinel_items(
        bounds,
        status_box,
        purpose="entrenamiento de perfiles",
    )
    items = select_training_sentinel_items(items, samples, SENTINEL_MAX_TRAINING_ITEMS, status_box)
    best_score = np.full(len(samples), -np.inf, dtype="float32")
    best_features, sum_features = empty_spectral_accumulators(len(samples))
    bare_observation_count = np.zeros(len(samples), dtype="uint16")
    failed_items = []
    processed_items = 0
    dry_season_items_used = 0
    target_valid_samples = min(
        len(samples),
        max(SENTINEL_MIN_WOSIS_SAMPLES, SENTINEL_TRAINING_TARGET_VALID_SAMPLES),
    )

    with rasterio.Env(**SENTINEL_GDAL_OPTIONS):
        for index, item in enumerate(items, start=1):
            valid_so_far = int(np.count_nonzero(np.isfinite(best_score)))
            mean_observations = mean_bare_observations(bare_observation_count, np.isfinite(best_score))
            update_status(
                status_box,
                "Extrayendo Sentinel-2 para entrenamiento "
                f"{index:,}/{len(items):,} escenas; perfiles validos "
                f"{valid_so_far:,}/{len(samples):,}; media de observaciones "
                f"{mean_observations:.1f} "
                f"(prioridad estacion seca {SENTINEL_DRY_SEASON_MONTHS}).",
            )
            try:
                bands = read_item_reflectance_bands_samples(item, samples)
                scl = read_asset_samples(item_asset_href(item, "scl"), samples).astype("int16")
            except Exception as exc:
                failed_items.append((item.id, str(exc)))
                update_status(
                    status_box,
                    "Se omitio una escena Sentinel-2 por error de lectura "
                    f"({len(failed_items):,}/{SENTINEL_MAX_ITEM_FAILURES:,} fallos).",
                )
                raise_if_too_many_sentinel_failures(failed_items, "el entrenamiento", exc)
                continue

            processed_items += 1
            dry_season = sentinel_item_is_dry_season(item)
            if dry_season:
                dry_season_items_used += 1
            features = derive_sentinel_features(bands)
            bare_mask, score = bare_soil_mask_and_score(
                features,
                scl,
                dry_season=dry_season,
            )
            accumulate_bare_spectral_features(
                best_features,
                sum_features,
                bare_observation_count,
                best_score,
                features,
                bare_mask,
                score,
            )

            valid_so_far = int(np.count_nonzero(np.isfinite(best_score)))
            mean_observations = mean_bare_observations(bare_observation_count, np.isfinite(best_score))
            if (
                processed_items >= SENTINEL_MIN_TRAINING_ITEMS
                and valid_so_far >= target_valid_samples
                and mean_observations >= SENTINEL_TRAINING_TARGET_MEAN_BARE_OBSERVATIONS
            ):
                update_status(
                    status_box,
                    "Entrenamiento Sentinel-2 listo con "
                    f"{valid_so_far:,} perfiles validos y "
                    f"{mean_observations:.1f} observaciones medias por perfil.",
                )
                break

    valid_best = np.isfinite(best_score)
    feature_values = model_feature_arrays_from_accumulators(
        best_features,
        sum_features,
        bare_observation_count,
        best_score,
        valid_best,
    )
    from dem_covariates import terrain_feature_arrays_for_points

    dem_features, dem_meta = terrain_feature_arrays_for_points(
        samples,
        poly_geom=poly_geom,
        status_callback=lambda message: update_status(status_box, message),
    )
    feature_values = attach_dem_features(feature_values, dem_features)

    from s1_covariates import s1_feature_arrays_for_points

    s1_features, s1_meta = s1_feature_arrays_for_points(
        samples,
        poly_geom=poly_geom,
        status_callback=lambda message: update_status(status_box, message),
    )
    feature_values = attach_dem_features(feature_values, s1_features)
    feature_frame = pd.DataFrame(feature_values)
    valid = np.isfinite(best_score)
    valid &= feature_frame.replace([np.inf, -np.inf], np.nan).notna().all(axis=1).to_numpy()
    valid_count = int(np.count_nonzero(valid))
    if valid_count < SENTINEL_MIN_WOSIS_SAMPLES:
        raise RuntimeError(
            "No hay suficientes perfiles de entrenamiento con observaciones Sentinel-2 "
            f"de suelo descubierto y Sentinel-1 RTC ({valid_count}; minimo "
            f"{SENTINEL_MIN_WOSIS_SAMPLES}). "
            f"Se procesaron {processed_items} escenas Sentinel-2 para "
            f"{len(samples)} perfiles completos y fallaron {len(failed_items)} escenas. "
            "El modelo experimental se detiene. Prueba con un poligono en una zona con "
            "mas suelo descubierto, amplia el buffer/periodo de entrenamiento en el codigo "
            "o usa OpenLandMap/SoilGrids para esta corrida."
        )

    valid_samples = samples.loc[valid].reset_index(drop=True)
    x_train = feature_frame.loc[valid, SENTINEL_MODEL_FEATURE_NAMES].to_numpy(dtype="float32")
    y_train = valid_samples[["sand", "silt", "clay"]].to_numpy(dtype="float32")
    source_counts = (
        valid_samples["source"].value_counts().to_dict()
        if "source" in valid_samples.columns
        else {}
    )
    summary = {
        "training_complete_profiles": int(len(samples)),
        "wosis_complete_profiles": int(source_counts.get(WOSIS_SOURCE_LABEL, 0)),
        "calicatas_cr_complete_profiles": int(
            source_counts.get(CALICATAS_CR_SOURCE_LABEL, 0)
        ),
        "training_profiles_with_bare_sentinel": int(len(valid_samples)),
        "training_profiles_by_source": {
            str(key): int(value) for key, value in source_counts.items()
        },
        "training_sentinel_items_used": int(processed_items),
        "training_sentinel_items_failed": int(len(failed_items)),
        "training_dry_season_items_used": int(dry_season_items_used),
        "training_dry_season_items_fraction": round(
            dry_season_items_used / max(processed_items, 1),
            3,
        ),
        "training_bare_soil_policy": "strict_scl5_fallback_scl7_dry_season_priority",
        "training_dry_season_months": list(SENTINEL_DRY_SEASON_MONTHS),
        "mean_bare_observations_per_training_profile": round(
            float(np.mean(bare_observation_count[valid])),
            2,
        ),
        **{f"training_{key}": value for key, value in dem_meta.items()},
        **{f"training_{key}": value for key, value in s1_meta.items()},
    }
    return x_train, y_train, valid_samples, summary


def normalize_texture_fractions(values):
    """Clip to [0, 100] and rescale rows so sand+silt+clay = 100.

    Rows with non-positive totals after clipping become equal thirds (100/3).
    """
    values = np.asarray(values, dtype="float64")
    if values.ndim == 1:
        values = values.reshape(1, -1)
    clipped = np.clip(values, 0.0, 100.0)
    totals = clipped.sum(axis=1, keepdims=True)
    normalized = np.empty_like(clipped, dtype="float64")
    valid = totals[:, 0] > 0
    normalized[valid] = clipped[valid] / totals[valid] * 100.0
    normalized[~valid] = 100.0 / 3.0
    return normalized.astype("float32")


def neighborhood_mean(values, valid_mask, radius, min_neighbors):
    values = np.ma.filled(values, np.nan).astype("float32")
    finite = valid_mask & np.isfinite(values)
    if radius <= 0 or not np.any(finite):
        return np.where(finite, values, np.nan).astype("float32")

    weighted_values = np.where(finite, values, 0.0).astype("float32")
    counts = finite.astype("float32")
    padded_values = np.pad(weighted_values, radius, mode="constant", constant_values=0.0)
    padded_counts = np.pad(counts, radius, mode="constant", constant_values=0.0)
    smoothed_sum = np.zeros_like(values, dtype="float32")
    smoothed_count = np.zeros_like(values, dtype="float32")
    height, width = values.shape
    window_size = radius * 2 + 1

    for row_offset in range(window_size):
        for col_offset in range(window_size):
            smoothed_sum += padded_values[
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ]
            smoothed_count += padded_counts[
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ]

    with np.errstate(divide="ignore", invalid="ignore"):
        smoothed = smoothed_sum / smoothed_count
    return np.where(
        finite & (smoothed_count >= min_neighbors),
        smoothed,
        np.where(finite, values, np.nan),
    ).astype("float32")


def smooth_texture_fraction_rasters(fractions, valid_mask):
    radius = SENTINEL_FRACTION_SMOOTHING_RADIUS_PIXELS
    if radius <= 0:
        return fractions, {
            "fraction_smoothing_applied": False,
            "fraction_smoothing_radius_pixels": 0,
        }

    smoothed_arrays = {
        fraction_name: neighborhood_mean(
            fractions[fraction_name],
            valid_mask,
            radius,
            SENTINEL_MIN_SMOOTHING_NEIGHBORS,
        )
        for fraction_name in ["sand", "silt", "clay"]
    }
    stack = np.stack([smoothed_arrays[name] for name in ["sand", "silt", "clay"]], axis=-1)
    valid = valid_mask & np.isfinite(stack).all(axis=-1)
    normalized = np.full_like(stack, np.nan, dtype="float32")
    normalized[valid] = normalize_texture_fractions(stack[valid])

    smoothed_fractions = {
        "sand": np.ma.masked_invalid(normalized[..., 0]),
        "silt": np.ma.masked_invalid(normalized[..., 1]),
        "clay": np.ma.masked_invalid(normalized[..., 2]),
    }
    return smoothed_fractions, {
        "fraction_smoothing_applied": True,
        "fraction_smoothing_radius_pixels": int(radius),
        "fraction_smoothing_window_pixels": int(radius * 2 + 1),
        "fraction_smoothing_min_neighbors": int(SENTINEL_MIN_SMOOTHING_NEIGHBORS),
    }


def spatial_cv_groups(samples):
    lon = samples.geometry.x.to_numpy()
    lat = samples.geometry.y.to_numpy()
    return np.char.add(
        np.floor(lon * 2).astype("int32").astype(str),
        np.char.add("_", np.floor(lat * 2).astype("int32").astype(str)),
    )


class TextureFractionEnsemble:
    """Three independent RFs (sand/silt/clay) with post-normalization to 100%."""

    TARGET_NAMES = ("sand", "silt", "clay")

    def __init__(self, models):
        if len(models) != 3:
            raise ValueError("TextureFractionEnsemble requiere exactamente 3 modelos.")
        self.models = list(models)

    def predict(self, x):
        raw = np.column_stack([model.predict(x) for model in self.models])
        return normalize_texture_fractions(raw)

    def predict_with_tree_uncertainty(self, x):
        tree_fraction_stacks = []
        n_trees = min(len(model.estimators_) for model in self.models)
        for tree_index in range(n_trees):
            raw = np.column_stack(
                [model.estimators_[tree_index].predict(x) for model in self.models]
            )
            tree_fraction_stacks.append(normalize_texture_fractions(raw))
        tree_predictions = np.stack(tree_fraction_stacks, axis=0)
        mean_prediction = normalize_texture_fractions(np.nanmean(tree_predictions, axis=0))
        uncertainty = np.nanmean(np.nanstd(tree_predictions, axis=0), axis=1)
        return mean_prediction, uncertainty

    @property
    def feature_importances_(self):
        stacked = np.vstack([model.feature_importances_ for model in self.models])
        return np.mean(stacked, axis=0)


def _fit_fraction_models(x_train, y_train, n_estimators, random_state):
    from sklearn.ensemble import RandomForestRegressor

    models = []
    for target_index in range(3):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=3,
            max_features="sqrt",
            random_state=random_state + target_index,
            n_jobs=-1,
        )
        model.fit(x_train, y_train[:, target_index])
        models.append(model)
    return TextureFractionEnsemble(models)


def train_texture_model(x_train, y_train, samples, status_box=None):
    require_experimental_dependencies()
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold

    update_status(
        status_box,
        "Entrenando 3 Random Forest (arena/limo/arcilla) con Sentinel-2 y DEM...",
    )
    metrics = {
        "model": "RandomForestRegressor_x3_normalized",
        "training_samples": int(len(x_train)),
        "features": SENTINEL_MODEL_FEATURE_NAMES,
        "training_mean_sand": round(float(np.mean(y_train[:, 0])), 2),
        "training_mean_silt": round(float(np.mean(y_train[:, 1])), 2),
        "training_mean_clay": round(float(np.mean(y_train[:, 2])), 2),
        "fraction_normalization": "sum_to_100",
    }

    groups = spatial_cv_groups(samples)
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 3:
        fold_count = min(5, len(unique_groups))
        fold_mae = []
        fold_r2 = []
        for train_index, test_index in GroupKFold(n_splits=fold_count).split(x_train, y_train, groups):
            fold_ensemble = _fit_fraction_models(
                x_train[train_index],
                y_train[train_index],
                n_estimators=120,
                random_state=SENTINEL_RF_RANDOM_STATE,
            )
            predictions = fold_ensemble.predict(x_train[test_index])
            fold_mae.append(
                mean_absolute_error(y_train[test_index], predictions, multioutput="raw_values")
            )
            try:
                fold_r2.append(
                    r2_score(y_train[test_index], predictions, multioutput="raw_values")
                )
            except ValueError:
                pass
        mean_mae = np.mean(np.vstack(fold_mae), axis=0)
        mean_mae_all = float(np.mean(mean_mae))
        metrics.update(
            {
                "spatial_cv_folds": int(fold_count),
                "spatial_cv_mae_sand": round(float(mean_mae[0]), 2),
                "spatial_cv_mae_silt": round(float(mean_mae[1]), 2),
                "spatial_cv_mae_clay": round(float(mean_mae[2]), 2),
                "spatial_cv_mae_mean_fraction": round(mean_mae_all, 2),
            }
        )
        if mean_mae_all > 15:
            metrics["model_quality_warning"] = (
                "La validacion espacial del modelo Sentinel-WoSIS tiene MAE alto; "
                "use esta capa como apoyo exploratorio y contraste con SoilGrids o muestras locales."
            )
        if fold_r2:
            mean_r2 = np.nanmean(np.vstack(fold_r2), axis=0)
            metrics.update(
                {
                    "spatial_cv_r2_sand": round(float(mean_r2[0]), 3),
                    "spatial_cv_r2_silt": round(float(mean_r2[1]), 3),
                    "spatial_cv_r2_clay": round(float(mean_r2[2]), 3),
                }
            )
    else:
        metrics["spatial_cv_warning"] = "No se calculo validacion espacial: grupos insuficientes."

    ensemble = _fit_fraction_models(
        x_train,
        y_train,
        n_estimators=SENTINEL_RF_TREES,
        random_state=SENTINEL_RF_RANDOM_STATE,
    )
    metrics["feature_importance"] = {
        feature_name: round(float(importance), 5)
        for feature_name, importance in zip(
            SENTINEL_MODEL_FEATURE_NAMES, ensemble.feature_importances_
        )
    }
    metrics["feature_importance_by_fraction"] = {
        target_name: {
            feature_name: round(float(importance), 5)
            for feature_name, importance in zip(
                SENTINEL_MODEL_FEATURE_NAMES, model.feature_importances_
            )
        }
        for target_name, model in zip(TextureFractionEnsemble.TARGET_NAMES, ensemble.models)
    }
    return ensemble, metrics


def predict_texture_fractions(model, feature_arrays, valid_bare, status_box=None):
    feature_stack = np.stack([feature_arrays[name] for name in SENTINEL_MODEL_FEATURE_NAMES], axis=-1)
    valid = valid_bare & np.isfinite(feature_stack).all(axis=-1)
    if not np.any(valid):
        raise RuntimeError(
            "El compuesto Sentinel-2 no contiene pixeles validos para predecir textura."
        )

    shape = valid.shape
    x_predict = feature_stack[valid].astype("float32")
    predictions = np.full((len(x_predict), 3), np.nan, dtype="float32")
    uncertainty = np.full(len(x_predict), np.nan, dtype="float32")
    chunk_size = 50_000
    for start in range(0, len(x_predict), chunk_size):
        end = min(start + chunk_size, len(x_predict))
        update_status(
            status_box,
            f"Prediciendo arena/limo/arcilla Sentinel-WoSIS {end:,}/{len(x_predict):,} pixeles...",
        )
        chunk = x_predict[start:end]
        if hasattr(model, "predict_with_tree_uncertainty"):
            chunk_pred, chunk_uncertainty = model.predict_with_tree_uncertainty(chunk)
            predictions[start:end] = chunk_pred
            uncertainty[start:end] = chunk_uncertainty
        else:
            predictions[start:end] = normalize_texture_fractions(model.predict(chunk))
            tree_predictions = np.stack(
                [normalize_texture_fractions(tree.predict(chunk)) for tree in model.estimators_],
                axis=0,
            )
            uncertainty[start:end] = np.nanmean(np.nanstd(tree_predictions, axis=0), axis=1)

    sand = np.full(shape, np.nan, dtype="float32")
    silt = np.full(shape, np.nan, dtype="float32")
    clay = np.full(shape, np.nan, dtype="float32")
    uncertainty_grid = np.full(shape, -9999.0, dtype="float32")
    sand[valid] = predictions[:, 0]
    silt[valid] = predictions[:, 1]
    clay[valid] = predictions[:, 2]
    uncertainty_grid[valid] = uncertainty
    return {
        "sand": np.ma.masked_invalid(sand),
        "silt": np.ma.masked_invalid(silt),
        "clay": np.ma.masked_invalid(clay),
    }, uncertainty_grid, valid


def read_sentinel_wosis_fraction_rasters(
    poly_geom,
    status_box=None,
    max_pixels=2_500_000,
    training_source=DEFAULT_TRAINING_SOURCE,
):
    require_experimental_dependencies()
    samples = fetch_training_texture_samples(
        poly_geom,
        training_source=training_source,
        status_box=status_box,
    )
    x_train, y_train, valid_samples, training_summary = extract_training_features_from_sentinel(
        samples,
        status_box,
        poly_geom=poly_geom,
    )
    model, model_metrics = train_texture_model(x_train, y_train, valid_samples, status_box)
    feature_arrays, valid_bare, bare_count, best_score, grid, composite_summary = (
        build_sentinel_bare_soil_composite(poly_geom, status_box, max_pixels=max_pixels)
    )
    fractions, uncertainty_grid, valid_prediction_pixels = predict_texture_fractions(
        model,
        feature_arrays,
        valid_bare,
        status_box,
    )
    fractions, smoothing_summary = smooth_texture_fraction_rasters(
        fractions,
        valid_prediction_pixels,
    )

    bare_count_out = np.where(grid["mask"], bare_count, 0).astype("uint16")
    bare_score_out = np.where(np.isfinite(best_score), best_score, -9999.0).astype("float32")
    source_counts = (
        samples["source"].value_counts().to_dict() if "source" in samples.columns else {}
    )
    training_summary = {
        **training_summary,
        "training_source": training_source,
        "training_source_label": TRAINING_SOURCE_OPTIONS[training_source],
        "training_profiles_available": int(len(samples)),
        "training_profiles_available_by_source": {
            str(key): int(value) for key, value in source_counts.items()
        },
        "calicatas_cr_csv": CALICATAS_CR_CSV_PATH.name,
    }
    auxiliary_rasters = {
        "sentinel_bare_observations": {
            "filename": "sentinel_suelo_descubierto_observaciones.tif",
            "array": bare_count_out,
            "dtype": "uint16",
            "nodata": 0,
            "description": (
                "Cantidad de escenas Sentinel-2 L2A donde el pixel fue clasificado "
                "como suelo descubierto para el compuesto experimental."
            ),
            "summary": composite_summary,
        },
        "sentinel_bare_score": {
            "filename": "sentinel_suelo_descubierto_score.tif",
            "array": bare_score_out,
            "dtype": "float32",
            "nodata": -9999.0,
            "description": (
                "Puntaje interno usado para escoger la observacion Sentinel-2 mas "
                "representativa de suelo descubierto por pixel."
            ),
            "summary": {
                "valid_score_pixels": int(np.count_nonzero(np.isfinite(best_score))),
            },
        },
        "sentinel_model_uncertainty": {
            "filename": "incertidumbre_modelo_sentinel_wosis.tif",
            "array": uncertainty_grid,
            "dtype": "float32",
            "nodata": -9999.0,
            "description": (
                "Incertidumbre experimental: desviacion promedio entre arboles del "
                "Random Forest para arena/limo/arcilla. No es error certificado."
            ),
            "summary": {
                **training_summary,
                **model_metrics,
                **smoothing_summary,
                "predicted_pixels": int(np.count_nonzero(valid_prediction_pixels)),
            },
        },
    }
    return (
        fractions,
        grid["transform"],
        grid["crs"],
        grid["pixel_count"],
        auxiliary_rasters,
    )
