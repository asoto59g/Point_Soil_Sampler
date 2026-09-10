from datetime import datetime, timezone
import os

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
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as transform_coordinates
from shapely.geometry import mapping


SENTINEL_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SENTINEL_COLLECTION = "sentinel-2-l2a"
SENTINEL_DATETIME_START = "2016-01-01"
SENTINEL_TARGET_RESOLUTION_M = 20
SENTINEL_CLOUD_COVER_LT = 80
SENTINEL_MIN_WOSIS_SAMPLES = 30
SENTINEL_WOSIS_BUFFER_KM = 250
SENTINEL_RF_TREES = 300
SENTINEL_RF_RANDOM_STATE = 42
SENTINEL_GDAL_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "50000000",
}
WOSIS_WFS_URL = "https://maps.isric.org/mapserv"
WOSIS_PROPERTIES = {
    "sand": "wosis_latest_sand",
    "silt": "wosis_latest_silt",
    "clay": "wosis_latest_clay",
}
SENTINEL_ASSET_ALIASES = {
    "blue": ("B02", "blue"),
    "green": ("B03", "green"),
    "red": ("B04", "red"),
    "nir": ("B08", "nir"),
    "swir1": ("B11", "swir16"),
    "swir2": ("B12", "swir22"),
    "scl": ("SCL", "scl"),
}
SENTINEL_FEATURE_NAMES = [
    "B02",
    "B03",
    "B04",
    "B08",
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
]


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


def fetch_wosis_property(requests_module, property_key, bounds):
    layer_name = WOSIS_PROPERTIES[property_key]
    params = {
        "map": "/map/wosis_latest.map",
        "SERVICE": "WFS",
        "VERSION": "1.0.0",
        "REQUEST": "GetFeature",
        "TYPENAME": layer_name,
        "OUTPUTFORMAT": "geojson",
        "SRSNAME": "EPSG:4326",
        "BBOX": ",".join(f"{value:.8f}" for value in bounds),
    }
    response = requests_module.get(WOSIS_WFS_URL, params=params, timeout=120)
    if response.status_code != 200:
        raise RuntimeError(
            f"WoSIS/ISRIC no respondio correctamente para {property_key}: "
            f"HTTP {response.status_code}."
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"WoSIS/ISRIC no devolvio GeoJSON valido para {property_key}.") from exc

    features = payload.get("features", [])
    if not features:
        return gpd.GeoDataFrame(columns=["profile_id", property_key, "geometry"], geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")


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


def fetch_wosis_texture_samples(poly_geom, status_box=None):
    requests_module = require_experimental_dependencies()["requests"]
    bounds = expanded_wgs84_bounds(poly_geom, SENTINEL_WOSIS_BUFFER_KM)
    update_status(
        status_box,
        "Descargando observaciones reales WoSIS de arena/limo/arcilla "
        f"en un buffer de {SENTINEL_WOSIS_BUFFER_KM} km...",
    )

    property_tables = {}
    for property_key in WOSIS_PROPERTIES:
        raw = fetch_wosis_property(requests_module, property_key, bounds)
        property_tables[property_key] = weighted_depth_average_for_property(raw, property_key)

    samples = property_tables["sand"][["profile_id", "sand", "geometry"]]
    for property_key in ["silt", "clay"]:
        samples = samples.merge(
            property_tables[property_key][["profile_id", property_key]],
            on="profile_id",
            how="inner",
        )

    if samples.empty:
        raise RuntimeError(
            "WoSIS no devolvio perfiles con arena, limo y arcilla 0-30 cm "
            "para el area de entrenamiento. El modelo experimental se detiene."
        )

    samples = gpd.GeoDataFrame(samples, geometry="geometry", crs="EPSG:4326")
    for column in ["sand", "silt", "clay"]:
        samples[column] = pd.to_numeric(samples[column], errors="coerce")
    samples = samples.dropna(subset=["sand", "silt", "clay", "geometry"])
    total = samples[["sand", "silt", "clay"]].sum(axis=1)
    samples = samples[(total >= 60) & (total <= 140)]
    total = samples[["sand", "silt", "clay"]].sum(axis=1)
    samples[["sand", "silt", "clay"]] = samples[["sand", "silt", "clay"]].div(total, axis=0) * 100.0

    if len(samples) < SENTINEL_MIN_WOSIS_SAMPLES:
        raise RuntimeError(
            "WoSIS devolvio muy pocas muestras completas para entrenar "
            f"({len(samples)}; minimo {SENTINEL_MIN_WOSIS_SAMPLES}). "
            "Amplia el poligono o usa OpenLandMap/SoilGrids."
        )
    return samples.reset_index(drop=True)


def search_sentinel_items(bounds, status_box=None, purpose="modelo"):
    modules = require_experimental_dependencies()
    pystac_client = modules["pystac_client"]
    planetary_computer = modules["planetary_computer"]

    datetime_range = sentinel_datetime_range()
    update_status(
        status_box,
        "Buscando todas las escenas Sentinel-2 L2A disponibles para "
        f"{purpose} ({datetime_range}, nubosidad escena < {SENTINEL_CLOUD_COVER_LT}%)...",
    )
    catalog = pystac_client.Client.open(
        SENTINEL_STAC_URL,
        modifier=planetary_computer.sign_inplace,
    )
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
    return items


def item_asset_href(item, logical_name):
    for asset_name in SENTINEL_ASSET_ALIASES[logical_name]:
        if asset_name in item.assets:
            return item.assets[asset_name].href
    raise RuntimeError(f"La escena Sentinel-2 {item.id} no contiene el asset {logical_name}.")


def read_asset_grid(href, target_crs, transform, width, height, resampling):
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


def derive_sentinel_features(bands):
    blue = bands["blue"]
    green = bands["green"]
    red = bands["red"]
    nir = bands["nir"]
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

    return {
        "B02": blue,
        "B03": green,
        "B04": red,
        "B08": nir,
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
    }


def bare_soil_mask_and_score(features, scl):
    valid_features = np.ones_like(features["NDVI"], dtype=bool)
    for feature_name in SENTINEL_FEATURE_NAMES:
        valid_features &= np.isfinite(features[feature_name])

    not_vegetated_scl = scl == 5
    bare_mask = (
        valid_features
        & not_vegetated_scl
        & (features["NDVI"] >= -0.05)
        & (features["NDVI"] <= 0.35)
        & (features["NDWI"] < 0.10)
        & (features["B04"] > 0.02)
    )
    score = (
        features["BSI"]
        - np.abs(features["NDVI"]) * 0.5
        + features["BI"] * 0.05
    ).astype("float32")
    return bare_mask, score


def build_sentinel_bare_soil_composite(poly_geom, status_box=None, max_pixels=2_500_000):
    grid = build_prediction_grid(poly_geom, max_pixels)
    bounds = tuple(float(value) for value in poly_geom.bounds)
    items = search_sentinel_items(bounds, status_box, purpose="prediccion del poligono")

    shape = (grid["height"], grid["width"])
    best_score = np.full(shape, -np.inf, dtype="float32")
    bare_count = np.zeros(shape, dtype="uint16")
    feature_arrays = {
        feature_name: np.full(shape, np.nan, dtype="float32")
        for feature_name in SENTINEL_FEATURE_NAMES
    }

    with rasterio.Env(**SENTINEL_GDAL_OPTIONS):
        for index, item in enumerate(items, start=1):
            update_status(status_box, f"Componiendo suelo descubierto Sentinel-2 {index:,}/{len(items):,}...")
            bands = {
                "blue": reflectance(
                    read_asset_grid(
                        item_asset_href(item, "blue"),
                        grid["crs"],
                        grid["transform"],
                        grid["width"],
                        grid["height"],
                        Resampling.bilinear,
                    )
                ),
                "green": reflectance(
                    read_asset_grid(
                        item_asset_href(item, "green"),
                        grid["crs"],
                        grid["transform"],
                        grid["width"],
                        grid["height"],
                        Resampling.bilinear,
                    )
                ),
                "red": reflectance(
                    read_asset_grid(
                        item_asset_href(item, "red"),
                        grid["crs"],
                        grid["transform"],
                        grid["width"],
                        grid["height"],
                        Resampling.bilinear,
                    )
                ),
                "nir": reflectance(
                    read_asset_grid(
                        item_asset_href(item, "nir"),
                        grid["crs"],
                        grid["transform"],
                        grid["width"],
                        grid["height"],
                        Resampling.bilinear,
                    )
                ),
                "swir1": reflectance(
                    read_asset_grid(
                        item_asset_href(item, "swir1"),
                        grid["crs"],
                        grid["transform"],
                        grid["width"],
                        grid["height"],
                        Resampling.bilinear,
                    )
                ),
                "swir2": reflectance(
                    read_asset_grid(
                        item_asset_href(item, "swir2"),
                        grid["crs"],
                        grid["transform"],
                        grid["width"],
                        grid["height"],
                        Resampling.bilinear,
                    )
                ),
            }
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
            features = derive_sentinel_features(bands)
            bare_mask, score = bare_soil_mask_and_score(features, scl)
            bare_mask &= grid["mask"]
            bare_count[bare_mask] = np.minimum(bare_count[bare_mask] + 1, np.iinfo("uint16").max)
            update_mask = bare_mask & (score > best_score)
            if np.any(update_mask):
                best_score[update_mask] = score[update_mask]
                for feature_name in SENTINEL_FEATURE_NAMES:
                    feature_arrays[feature_name][update_mask] = features[feature_name][update_mask]

    valid_bare = np.isfinite(best_score) & grid["mask"]
    if not np.any(valid_bare):
        raise RuntimeError(
            "No se detectaron pixeles de suelo descubierto Sentinel-2 dentro del poligono. "
            "El modelo experimental se detiene."
        )
    for feature_name in SENTINEL_FEATURE_NAMES:
        feature_arrays[feature_name][~valid_bare] = np.nan

    summary = {
        "sentinel_collection": SENTINEL_COLLECTION,
        "sentinel_datetime": sentinel_datetime_range(),
        "sentinel_items_used": len(items),
        "target_resolution_m": SENTINEL_TARGET_RESOLUTION_M,
        "cloud_cover_scene_threshold_percent": SENTINEL_CLOUD_COVER_LT,
        "aoi_pixels": int(np.count_nonzero(grid["mask"])),
        "bare_soil_pixels": int(np.count_nonzero(valid_bare)),
        "bare_soil_pixels_percent": round(
            np.count_nonzero(valid_bare) / max(np.count_nonzero(grid["mask"]), 1) * 100.0,
            2,
        ),
    }
    return feature_arrays, valid_bare, bare_count, best_score, grid, summary


def read_asset_samples(href, points_wgs84):
    lons = points_wgs84.geometry.x.to_numpy()
    lats = points_wgs84.geometry.y.to_numpy()
    with rasterio.open(href) as src:
        xs, ys = transform_coordinates("EPSG:4326", src.crs, lons.tolist(), lats.tolist())
        sampled = []
        for value in src.sample(zip(xs, ys), masked=True):
            if np.ma.is_masked(value) or len(value) == 0:
                sampled.append(np.nan)
            else:
                sampled.append(float(value[0]))
    return np.asarray(sampled, dtype="float32")


def extract_training_features_from_sentinel(samples, status_box=None):
    bounds = tuple(float(value) for value in samples.total_bounds)
    items = search_sentinel_items(bounds, status_box, purpose="entrenamiento WoSIS")
    best_score = np.full(len(samples), -np.inf, dtype="float32")
    feature_values = {
        feature_name: np.full(len(samples), np.nan, dtype="float32")
        for feature_name in SENTINEL_FEATURE_NAMES
    }
    bare_observation_count = np.zeros(len(samples), dtype="uint16")

    with rasterio.Env(**SENTINEL_GDAL_OPTIONS):
        for index, item in enumerate(items, start=1):
            update_status(status_box, f"Extrayendo Sentinel-2 en perfiles WoSIS {index:,}/{len(items):,}...")
            bands = {
                "blue": reflectance(read_asset_samples(item_asset_href(item, "blue"), samples)),
                "green": reflectance(read_asset_samples(item_asset_href(item, "green"), samples)),
                "red": reflectance(read_asset_samples(item_asset_href(item, "red"), samples)),
                "nir": reflectance(read_asset_samples(item_asset_href(item, "nir"), samples)),
                "swir1": reflectance(read_asset_samples(item_asset_href(item, "swir1"), samples)),
                "swir2": reflectance(read_asset_samples(item_asset_href(item, "swir2"), samples)),
            }
            scl = read_asset_samples(item_asset_href(item, "scl"), samples).astype("int16")
            features = derive_sentinel_features(bands)
            bare_mask, score = bare_soil_mask_and_score(features, scl)
            bare_observation_count[bare_mask] = np.minimum(
                bare_observation_count[bare_mask] + 1,
                np.iinfo("uint16").max,
            )
            update_mask = bare_mask & (score > best_score)
            if np.any(update_mask):
                best_score[update_mask] = score[update_mask]
                for feature_name in SENTINEL_FEATURE_NAMES:
                    feature_values[feature_name][update_mask] = features[feature_name][update_mask]

    feature_frame = pd.DataFrame(feature_values)
    valid = np.isfinite(best_score)
    valid &= feature_frame.replace([np.inf, -np.inf], np.nan).notna().all(axis=1).to_numpy()
    if np.count_nonzero(valid) < SENTINEL_MIN_WOSIS_SAMPLES:
        raise RuntimeError(
            "No hay suficientes perfiles WoSIS con observaciones Sentinel-2 de suelo descubierto "
            f"({np.count_nonzero(valid)}; minimo {SENTINEL_MIN_WOSIS_SAMPLES}). "
            "El modelo experimental se detiene."
        )

    valid_samples = samples.loc[valid].reset_index(drop=True)
    x_train = feature_frame.loc[valid, SENTINEL_FEATURE_NAMES].to_numpy(dtype="float32")
    y_train = valid_samples[["sand", "silt", "clay"]].to_numpy(dtype="float32")
    summary = {
        "wosis_complete_profiles": int(len(samples)),
        "training_profiles_with_bare_sentinel": int(len(valid_samples)),
        "training_sentinel_items_used": int(len(items)),
        "mean_bare_observations_per_training_profile": round(
            float(np.mean(bare_observation_count[valid])),
            2,
        ),
    }
    return x_train, y_train, valid_samples, summary


def normalize_texture_fractions(values):
    values = np.asarray(values, dtype="float32")
    clipped = np.clip(values, 0, 100)
    totals = clipped.sum(axis=1)
    normalized = np.full_like(clipped, np.nan, dtype="float32")
    valid = totals > 0
    normalized[valid] = clipped[valid] / totals[valid, None] * 100.0
    return normalized


def spatial_cv_groups(samples):
    lon = samples.geometry.x.to_numpy()
    lat = samples.geometry.y.to_numpy()
    return np.char.add(
        np.floor(lon * 2).astype("int32").astype(str),
        np.char.add("_", np.floor(lat * 2).astype("int32").astype(str)),
    )


def train_texture_model(x_train, y_train, samples, status_box=None):
    require_experimental_dependencies()
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import GroupKFold

    update_status(status_box, "Entrenando Random Forest con perfiles WoSIS y covariables Sentinel-2...")
    model = RandomForestRegressor(
        n_estimators=SENTINEL_RF_TREES,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=SENTINEL_RF_RANDOM_STATE,
        n_jobs=-1,
    )
    metrics = {
        "model": "RandomForestRegressor",
        "training_samples": int(len(x_train)),
        "features": SENTINEL_FEATURE_NAMES,
    }

    groups = spatial_cv_groups(samples)
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 3:
        fold_count = min(5, len(unique_groups))
        fold_mae = []
        fold_r2 = []
        for train_index, test_index in GroupKFold(n_splits=fold_count).split(x_train, y_train, groups):
            fold_model = RandomForestRegressor(
                n_estimators=120,
                min_samples_leaf=3,
                max_features="sqrt",
                random_state=SENTINEL_RF_RANDOM_STATE,
                n_jobs=-1,
            )
            fold_model.fit(x_train[train_index], y_train[train_index])
            predictions = normalize_texture_fractions(fold_model.predict(x_train[test_index]))
            fold_mae.append(mean_absolute_error(y_train[test_index], predictions, multioutput="raw_values"))
            try:
                fold_r2.append(r2_score(y_train[test_index], predictions, multioutput="raw_values"))
            except ValueError:
                pass
        mean_mae = np.mean(np.vstack(fold_mae), axis=0)
        metrics.update(
            {
                "spatial_cv_folds": int(fold_count),
                "spatial_cv_mae_sand": round(float(mean_mae[0]), 2),
                "spatial_cv_mae_silt": round(float(mean_mae[1]), 2),
                "spatial_cv_mae_clay": round(float(mean_mae[2]), 2),
            }
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

    model.fit(x_train, y_train)
    metrics["feature_importance"] = {
        feature_name: round(float(importance), 5)
        for feature_name, importance in zip(SENTINEL_FEATURE_NAMES, model.feature_importances_)
    }
    return model, metrics


def predict_texture_fractions(model, feature_arrays, valid_bare, status_box=None):
    feature_stack = np.stack([feature_arrays[name] for name in SENTINEL_FEATURE_NAMES], axis=-1)
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
        update_status(status_box, f"Prediciendo arena/limo/arcilla Sentinel-WoSIS {end:,}/{len(x_predict):,} pixeles...")
        chunk = x_predict[start:end]
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


def read_sentinel_wosis_fraction_rasters(poly_geom, status_box=None, max_pixels=2_500_000):
    require_experimental_dependencies()
    samples = fetch_wosis_texture_samples(poly_geom, status_box)
    x_train, y_train, valid_samples, training_summary = extract_training_features_from_sentinel(
        samples,
        status_box,
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

    bare_count_out = np.where(grid["mask"], bare_count, 0).astype("uint16")
    bare_score_out = np.where(np.isfinite(best_score), best_score, -9999.0).astype("float32")
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
