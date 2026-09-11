"""Sentinel-1 RTC (VV/VH) covariates for the experimental texture model."""

from __future__ import annotations

from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as transform_coordinates


S1_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S1_COLLECTION = "sentinel-1-rtc"
S1_DATETIME_START = "2019-01-01"
S1_MAX_ITEMS = int(__import__("os").getenv("S1_MAX_ITEMS", "30"))
S1_MAX_ITEM_FAILURES = 12
S1_LINEAR_FLOOR = 1e-7
S1_FEATURE_NAMES = ("VV_DB", "VH_DB", "VV_VH_DB")


def _require_planetary_deps():
    missing = []
    modules = {}
    for package_name, import_name in [
        ("pystac-client", "pystac_client"),
        ("planetary-computer", "planetary_computer"),
    ]:
        try:
            modules[import_name] = __import__(import_name)
        except ImportError:
            missing.append(package_name)
    if missing:
        raise RuntimeError(
            "Sentinel-1 RTC requiere dependencias no instaladas: "
            f"{', '.join(missing)}."
        )
    return modules


def _sign_href(href: str) -> str:
    planetary_computer = _require_planetary_deps()["planetary_computer"]
    sign_url = getattr(planetary_computer, "sign_url", None)
    if callable(sign_url):
        signed = sign_url(href)
        return getattr(signed, "href", signed)
    signed = planetary_computer.sign(href)
    return getattr(signed, "href", signed)


def _datetime_range() -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    return f"{S1_DATETIME_START}/{today}"


def _item_datetime(item):
    value = item.properties.get("datetime") or item.properties.get("start_datetime")
    return str(value) if value else ""


def _item_has_vv_vh(item) -> bool:
    assets = item.assets
    return ("vv" in assets or "VV" in assets) and ("vh" in assets or "VH" in assets)


def _asset_href(item, polarization: str) -> str:
    for key in (polarization.lower(), polarization.upper()):
        if key in item.assets:
            return _sign_href(item.assets[key].href)
    raise RuntimeError(
        f"La escena Sentinel-1 {item.id} no contiene el asset {polarization}."
    )


def search_sentinel1_rtc_items(bounds_wgs84, max_items=S1_MAX_ITEMS, status_callback=None):
    modules = _require_planetary_deps()
    pystac_client = modules["pystac_client"]
    if status_callback:
        status_callback("Buscando escenas Sentinel-1 RTC (VV/VH) en Planetary Computer...")
    catalog = pystac_client.Client.open(S1_STAC_URL)
    search = catalog.search(
        collections=[S1_COLLECTION],
        bbox=list(bounds_wgs84),
        datetime=_datetime_range(),
    )
    items = [item for item in search.items() if _item_has_vv_vh(item)]
    items.sort(key=_item_datetime, reverse=True)
    selected = items[: max(1, int(max_items))]
    if not selected:
        raise RuntimeError(
            "Planetary Computer no devolvio escenas Sentinel-1 RTC con VV/VH "
            f"para el area {tuple(float(v) for v in bounds_wgs84)}."
        )
    if status_callback:
        status_callback(
            f"Sentinel-1 RTC: {len(selected):,} escenas seleccionadas "
            f"(de {len(items):,} con VV/VH)."
        )
    return selected


def linear_to_db(linear):
    linear = np.asarray(linear, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 10.0 * np.log10(np.maximum(linear, S1_LINEAR_FLOOR))
    return np.where(np.isfinite(linear) & (linear > 0), db, np.nan).astype("float32")


def _sanitize_linear_backscatter(values, nodata=None):
    array = np.ma.filled(values, np.nan).astype("float32")
    invalid = ~np.isfinite(array) | (array <= 0)
    if nodata is not None and np.isfinite(nodata):
        invalid |= np.isclose(array, float(nodata))
    invalid |= np.isclose(array, -32768.0)
    return np.where(invalid, np.nan, array).astype("float32")


def _read_linear_grid(href, target_crs, transform, width, height):
    with rasterio.open(href) as src:
        with WarpedVRT(
            src,
            crs=target_crs,
            transform=transform,
            width=width,
            height=height,
            resampling=Resampling.bilinear,
            nodata=src.nodata,
        ) as vrt:
            data = vrt.read(1, masked=True)
            return _sanitize_linear_backscatter(data, nodata=src.nodata)


def _read_linear_samples(href, points_wgs84):
    points = (
        points_wgs84.to_crs("EPSG:4326")
        if getattr(points_wgs84, "crs", None)
        else gpd.GeoDataFrame(points_wgs84).set_crs("EPSG:4326")
    )
    lons = points.geometry.x.to_numpy()
    lats = points.geometry.y.to_numpy()
    with rasterio.open(href) as src:
        xs, ys = transform_coordinates("EPSG:4326", src.crs, lons.tolist(), lats.tolist())
        sampled = []
        for value in src.sample(zip(xs, ys), masked=True):
            if len(value) == 0 or np.any(np.ma.getmaskarray(value)):
                sampled.append(np.nan)
                continue
            sample_value = float(value[0])
            if (
                not np.isfinite(sample_value)
                or sample_value <= 0
                or np.isclose(sample_value, -32768.0)
                or (
                    src.nodata is not None
                    and np.isfinite(src.nodata)
                    and np.isclose(sample_value, float(src.nodata))
                )
            ):
                sampled.append(np.nan)
            else:
                sampled.append(sample_value)
    return np.asarray(sampled, dtype="float32")


def _features_from_linear_median(vv_median, vh_median):
    vv_db = linear_to_db(vv_median)
    vh_db = linear_to_db(vh_median)
    return {
        "VV_DB": vv_db,
        "VH_DB": vh_db,
        "VV_VH_DB": (vv_db - vh_db).astype("float32"),
    }


def _nanmedian_stack(stack):
    if not stack:
        raise RuntimeError("No hay observaciones Sentinel-1 RTC validas para componer.")
    with np.errstate(all="ignore"):
        return np.nanmedian(np.stack(stack, axis=0), axis=0).astype("float32")


def s1_feature_arrays_for_grid(poly_geom, grid, status_callback=None, max_items=S1_MAX_ITEMS):
    """Median VV/VH RTC backscatter (dB) on a prediction grid."""
    bounds = tuple(float(value) for value in poly_geom.bounds)
    items = search_sentinel1_rtc_items(bounds, max_items=max_items, status_callback=status_callback)
    vv_stack = []
    vh_stack = []
    failed = []
    used_ids = []

    for index, item in enumerate(items, start=1):
        if status_callback:
            status_callback(
                f"Leyendo Sentinel-1 RTC {index:,}/{len(items):,} para prediccion..."
            )
        try:
            vv = _read_linear_grid(
                _asset_href(item, "vv"),
                grid["crs"],
                grid["transform"],
                grid["width"],
                grid["height"],
            )
            vh = _read_linear_grid(
                _asset_href(item, "vh"),
                grid["crs"],
                grid["transform"],
                grid["width"],
                grid["height"],
            )
        except Exception as exc:
            failed.append((item.id, str(exc)))
            if len(failed) >= S1_MAX_ITEM_FAILURES:
                raise RuntimeError(
                    "Demasiados fallos al leer Sentinel-1 RTC para prediccion "
                    f"({len(failed)}). Ultimo error: {exc}"
                ) from exc
            continue
        if not (np.isfinite(vv).any() and np.isfinite(vh).any()):
            failed.append((item.id, "sin pixeles VV/VH validos"))
            continue
        vv_stack.append(vv)
        vh_stack.append(vh)
        used_ids.append(item.id)

    vv_median = _nanmedian_stack(vv_stack)
    vh_median = _nanmedian_stack(vh_stack)
    features = _features_from_linear_median(vv_median, vh_median)
    valid = np.isfinite(features["VV_DB"]) & np.isfinite(features["VH_DB"])
    if not np.any(valid):
        raise RuntimeError(
            "El compuesto Sentinel-1 RTC no produjo pixeles VV/VH validos en el poligono."
        )

    meta = {
        "s1_collection": S1_COLLECTION,
        "s1_datetime": _datetime_range(),
        "s1_items_used": int(len(used_ids)),
        "s1_items_failed": int(len(failed)),
        "s1_features": list(S1_FEATURE_NAMES),
        "s1_composite": "nanmedian_linear_to_db",
        "s1_valid_pixel_fraction": round(
            float(np.count_nonzero(valid) / max(valid.size, 1)),
            4,
        ),
    }
    return features, meta


def s1_feature_arrays_for_points(
    points_gdf,
    poly_geom=None,
    status_callback=None,
    max_items=S1_MAX_ITEMS,
):
    """Median VV/VH RTC backscatter (dB) sampled at training points."""
    if points_gdf is None or len(points_gdf) == 0:
        empty = {name: np.array([], dtype="float32") for name in S1_FEATURE_NAMES}
        return empty, {
            "s1_collection": S1_COLLECTION,
            "s1_items_used": 0,
            "s1_items_failed": 0,
            "s1_features": list(S1_FEATURE_NAMES),
        }

    points = (
        points_gdf.to_crs("EPSG:4326")
        if getattr(points_gdf, "crs", None)
        else gpd.GeoDataFrame(points_gdf).set_crs("EPSG:4326")
    )
    # Always cover the training point cloud. Using only the prediction AOI
    # leaves distant buffer profiles without VV/VH and drops them from training.
    minx, miny, maxx, maxy = (float(value) for value in points.total_bounds)
    if poly_geom is not None:
        pminx, pminy, pmaxx, pmaxy = (float(value) for value in poly_geom.bounds)
        minx, miny = min(minx, pminx), min(miny, pminy)
        maxx, maxy = max(maxx, pmaxx), max(maxy, pmaxy)
    pad = max((maxx - minx), (maxy - miny), 0.05) * 0.05
    bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    items = search_sentinel1_rtc_items(bounds, max_items=max_items, status_callback=status_callback)
    vv_stack = []
    vh_stack = []
    failed = []
    used_ids = []

    for index, item in enumerate(items, start=1):
        if status_callback:
            status_callback(
                f"Muestreando Sentinel-1 RTC {index:,}/{len(items):,} en perfiles..."
            )
        try:
            vv = _read_linear_samples(_asset_href(item, "vv"), points)
            vh = _read_linear_samples(_asset_href(item, "vh"), points)
        except Exception as exc:
            failed.append((item.id, str(exc)))
            if len(failed) >= S1_MAX_ITEM_FAILURES:
                raise RuntimeError(
                    "Demasiados fallos al leer Sentinel-1 RTC para entrenamiento "
                    f"({len(failed)}). Ultimo error: {exc}"
                ) from exc
            continue
        if not (np.isfinite(vv).any() and np.isfinite(vh).any()):
            failed.append((item.id, "sin muestras VV/VH validas"))
            continue
        vv_stack.append(vv)
        vh_stack.append(vh)
        used_ids.append(item.id)

    vv_median = _nanmedian_stack(vv_stack)
    vh_median = _nanmedian_stack(vh_stack)
    features = _features_from_linear_median(vv_median, vh_median)
    valid_count = int(
        np.count_nonzero(np.isfinite(features["VV_DB"]) & np.isfinite(features["VH_DB"]))
    )
    if valid_count == 0:
        raise RuntimeError(
            "Sentinel-1 RTC no produjo valores VV/VH validos en los perfiles de entrenamiento."
        )

    meta = {
        "s1_collection": S1_COLLECTION,
        "s1_datetime": _datetime_range(),
        "s1_items_used": int(len(used_ids)),
        "s1_items_failed": int(len(failed)),
        "s1_features": list(S1_FEATURE_NAMES),
        "s1_composite": "nanmedian_linear_to_db",
        "s1_valid_training_points": valid_count,
    }
    return features, meta
