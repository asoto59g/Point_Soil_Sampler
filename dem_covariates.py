"""DEM covariates for Sentinel texture model (Costa Rica Drive DEM or Copernicus GLO-30)."""

from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.transform import xy as transform_xy
from rasterio.vrt import WarpedVRT
from shapely.geometry import box

from cr_dem_remote import DEFAULT_DRIVE_DEM_URL, open_raster_source


CR_DEM_DRIVE_URL = DEFAULT_DRIVE_DEM_URL
CR_DEM_CRS = "EPSG:5367"
CR_BBOX_WGS84 = (-86.0, 8.0, -82.5, 11.3)
COP_DEM_COLLECTION = "cop-dem-glo-30"
COP_DEM_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DEM_FEATURE_NAMES = ("ELEV", "SLOPE_DEG", "ASPECT_SIN", "ASPECT_COS", "CURV")
DEM_SOURCE_CR_DRIVE = "cr_drive"
DEM_SOURCE_COP_GLO30 = "copernicus_glo30"


def is_costa_rica_geometry(poly_geom) -> bool:
    centroid = poly_geom.centroid
    minx, miny, maxx, maxy = CR_BBOX_WGS84
    return bool(minx <= centroid.x <= maxx and miny <= centroid.y <= maxy)


def terrain_features_from_elevation(elev, transform):
    """Derive ELEV/SLOPE/ASPECT/CURV from an elevation grid (meters)."""
    elev = np.asarray(elev, dtype="float32")
    finite = np.isfinite(elev)
    elev_filled = np.where(finite, elev, np.nan)

    xres = abs(float(transform.a))
    yres = abs(float(transform.e))
    if not np.isfinite(xres) or xres <= 0:
        xres = 20.0
    if not np.isfinite(yres) or yres <= 0:
        yres = 20.0

    if xres < 0.01:
        mid_row = elev.shape[0] // 2
        mid_col = elev.shape[1] // 2
        _, lat = transform_xy(transform, mid_row, mid_col, offset="center")
        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = 111_320.0 * max(math.cos(math.radians(float(lat))), 0.2)
        xres_m = xres * meters_per_deg_lon
        yres_m = yres * meters_per_deg_lat
    else:
        xres_m = xres
        yres_m = yres

    dz_dy, dz_dx = np.gradient(elev_filled, yres_m, xres_m)
    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    slope_deg = np.degrees(slope_rad).astype("float32")
    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    aspect_sin = np.sin(aspect_rad).astype("float32")
    aspect_cos = np.cos(aspect_rad).astype("float32")

    d2z_dy2, _ = np.gradient(dz_dy, yres_m, xres_m)
    _, d2z_dx2 = np.gradient(dz_dx, yres_m, xres_m)
    curv = (d2z_dx2 + d2z_dy2).astype("float32")

    def _masked(values):
        return np.where(finite, values, np.nan).astype("float32")

    return {
        "ELEV": _masked(elev),
        "SLOPE_DEG": _masked(slope_deg),
        "ASPECT_SIN": _masked(aspect_sin),
        "ASPECT_COS": _masked(aspect_cos),
        "CURV": _masked(curv),
    }


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
            "DEM Copernicus GLO-30 requiere dependencias no instaladas: "
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


def _copernicus_dem_asset_hrefs(bounds_wgs84):
    modules = _require_planetary_deps()
    pystac_client = modules["pystac_client"]
    catalog = pystac_client.Client.open(COP_DEM_STAC_URL)
    search = catalog.search(
        collections=[COP_DEM_COLLECTION],
        bbox=list(bounds_wgs84),
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(
            "Planetary Computer no devolvio teselas Copernicus DEM GLO-30 "
            f"para el area {bounds_wgs84}."
        )
    hrefs = []
    for item in items:
        asset = item.assets.get("data") or item.assets.get("elevation")
        if asset is None:
            for candidate in item.assets.values():
                media = (candidate.media_type or "").lower()
                href = candidate.href or ""
                if "tif" in media or href.lower().endswith((".tif", ".tiff")):
                    asset = candidate
                    break
        if asset is None:
            continue
        hrefs.append(_sign_href(asset.href))
    if not hrefs:
        raise RuntimeError(
            "Las teselas Copernicus DEM GLO-30 no exponen un asset GeoTIFF usable."
        )
    return hrefs


def _read_elevation_from_hrefs(hrefs, target_crs, transform, width, height):
    accum = np.zeros((height, width), dtype="float64")
    counts = np.zeros((height, width), dtype="float64")
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff",
        GDAL_HTTP_MULTIRANGE="YES",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE="50000000",
    ):
        for href in hrefs:
            with rasterio.open(href) as src:
                nodata = src.nodata
                with WarpedVRT(
                    src,
                    crs=target_crs,
                    transform=transform,
                    width=width,
                    height=height,
                    resampling=Resampling.bilinear,
                    nodata=nodata,
                ) as vrt:
                    data = vrt.read(1, masked=True)
            arr = np.ma.filled(data, np.nan).astype("float64")
            if nodata is not None:
                arr[np.isclose(arr, float(nodata))] = np.nan
            valid = np.isfinite(arr)
            accum[valid] += arr[valid]
            counts[valid] += 1.0
    if not np.any(counts > 0):
        raise RuntimeError("No se pudieron leer valores de elevacion DEM en la grilla objetivo.")
    with np.errstate(divide="ignore", invalid="ignore"):
        elev = accum / counts
    elev[counts <= 0] = np.nan
    return elev.astype("float32")


def _load_copernicus_elevation_grid(bounds_wgs84, target_crs, transform, width, height):
    hrefs = _copernicus_dem_asset_hrefs(bounds_wgs84)
    elev = _read_elevation_from_hrefs(hrefs, target_crs, transform, width, height)
    meta = {
        "dem_source": DEM_SOURCE_COP_GLO30,
        "dem_collection": COP_DEM_COLLECTION,
        "dem_tiles": len(hrefs),
        "dem_fallback_used": False,
        "dem_crs": "EPSG:4326",
        "dem_resolution_label": "~30 m",
    }
    return elev, meta


def _cr_subset_bounds_in_dem_crs(poly_geom_wgs84, buffer_m=500.0):
    gdf = gpd.GeoDataFrame(geometry=[poly_geom_wgs84], crs="EPSG:4326").to_crs(CR_DEM_CRS)
    geom = gdf.geometry.iloc[0]
    if buffer_m:
        geom = geom.buffer(float(buffer_m))
    return tuple(float(v) for v in geom.bounds)


def _load_cr_drive_elevation_grid(poly_geom_wgs84, target_crs, transform, width, height):
    subset_bounds = _cr_subset_bounds_in_dem_crs(poly_geom_wgs84)
    with open_raster_source(CR_DEM_DRIVE_URL, subset_bounds=subset_bounds) as src:
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
            elev = np.ma.filled(data, np.nan).astype("float32")
            if src.nodata is not None:
                elev[np.isclose(elev, float(src.nodata))] = np.nan
            source_crs = src.crs.to_string() if src.crs else CR_DEM_CRS
            res = src.res
            res_label = f"~{abs(res[0]):.1f} m" if res and abs(res[0]) > 0.01 else "nativa CR"

    if not np.any(np.isfinite(elev)):
        raise RuntimeError("El DEM de Costa Rica (Google Drive) no devolvio celdas validas.")

    meta = {
        "dem_source": DEM_SOURCE_CR_DRIVE,
        "dem_url": CR_DEM_DRIVE_URL,
        "dem_tiles": 1,
        "dem_fallback_used": False,
        "dem_crs": source_crs,
        "dem_resolution_label": res_label,
    }
    return elev, meta


def load_elevation_grid(poly_geom, target_crs, transform, width, height, status_callback=None):
    """
    Load elevation on a target grid.
    Costa Rica AOI -> Drive DEM (CRTM05); elsewhere -> Copernicus GLO-30.
    Falls back to GLO-30 if the CR Drive DEM fails.
    """
    prefer_cr = is_costa_rica_geometry(poly_geom)
    bounds_wgs84 = tuple(float(v) for v in poly_geom.bounds)
    minx, miny, maxx, maxy = bounds_wgs84
    pad = max((maxx - minx), (maxy - miny), 0.01) * 0.05
    bounds_wgs84 = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    if prefer_cr:
        if status_callback:
            status_callback("Leyendo DEM publico de Costa Rica (Google Drive / CRTM05)...")
        try:
            return _load_cr_drive_elevation_grid(poly_geom, target_crs, transform, width, height)
        except Exception as exc:
            if status_callback:
                status_callback(
                    "DEM Costa Rica no disponible; usando Copernicus GLO-30. "
                    f"Detalle: {exc}"
                )
            elev, meta = _load_copernicus_elevation_grid(
                bounds_wgs84, target_crs, transform, width, height
            )
            meta["dem_fallback_used"] = True
            meta["dem_fallback_reason"] = str(exc)
            return elev, meta

    if status_callback:
        status_callback("Leyendo Copernicus DEM GLO-30 desde Planetary Computer...")
    return _load_copernicus_elevation_grid(bounds_wgs84, target_crs, transform, width, height)


def terrain_feature_arrays_for_grid(poly_geom, grid, status_callback=None):
    elev, meta = load_elevation_grid(
        poly_geom,
        grid["crs"],
        grid["transform"],
        grid["width"],
        grid["height"],
        status_callback=status_callback,
    )
    features = terrain_features_from_elevation(elev, grid["transform"])
    return features, meta


def terrain_feature_arrays_for_points(points_gdf, poly_geom=None, status_callback=None):
    """Build DEM feature vectors for point samples using a local warped elevation patch."""
    if points_gdf.empty:
        empty = {name: np.array([], dtype="float32") for name in DEM_FEATURE_NAMES}
        return empty, {"dem_source": None, "dem_fallback_used": False}

    points = points_gdf.to_crs("EPSG:4326") if points_gdf.crs else points_gdf.set_crs("EPSG:4326")
    ref_geom = poly_geom if poly_geom is not None else box(*points.total_bounds)

    points_metric = gpd.GeoDataFrame(geometry=points.geometry, crs="EPSG:4326")
    try:
        metric_crs = points_metric.estimate_utm_crs() or "EPSG:6933"
    except Exception:
        metric_crs = "EPSG:6933"
    points_m = points_metric.to_crs(metric_crs)
    minx, miny, maxx, maxy = points_m.total_bounds
    pad = 60.0
    minx -= pad
    miny -= pad
    maxx += pad
    maxy += pad
    resolution = 20.0
    width = max(2, int(math.ceil((maxx - minx) / resolution)))
    height = max(2, int(math.ceil((maxy - miny) / resolution)))
    max_dim = 2500
    if width > max_dim or height > max_dim:
        scale = max(width / max_dim, height / max_dim)
        resolution *= scale
        width = max(2, int(math.ceil((maxx - minx) / resolution)))
        height = max(2, int(math.ceil((maxy - miny) / resolution)))

    transform = from_origin(minx, maxy, resolution, resolution)
    elev, meta = load_elevation_grid(
        ref_geom,
        metric_crs,
        transform,
        width,
        height,
        status_callback=status_callback,
    )
    feature_grids = terrain_features_from_elevation(elev, transform)

    xs = points_m.geometry.x.to_numpy(dtype="float64")
    ys = points_m.geometry.y.to_numpy(dtype="float64")
    cols = ((xs - transform.c) / transform.a).astype("int32")
    rows = ((ys - transform.f) / transform.e).astype("int32")
    valid_idx = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    features = {}
    for name in DEM_FEATURE_NAMES:
        values = np.full(len(points), np.nan, dtype="float32")
        grid = feature_grids[name]
        values[valid_idx] = grid[rows[valid_idx], cols[valid_idx]]
        features[name] = values
    return features, meta
