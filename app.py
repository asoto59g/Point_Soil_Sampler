from datetime import datetime
from pathlib import Path
import json

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
}
DEFAULT_SOURCE_KEY = "openlandmap"
OUTPUT_ROOT = Path("salidas")
MAX_PIXELS = 2_500_000
GDAL_HTTP_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "50000000",
}


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


def read_soil_fraction_rasters(poly_geom, source_key, status_box=None):
    if source_key == "openlandmap":
        return read_openlandmap_fraction_rasters(poly_geom, status_box)
    if source_key == "soilgrids":
        return read_soilgrids_fraction_rasters(poly_geom, status_box)
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
    dissolved = clipped.dissolve(by="texture_id", as_index=False)
    zones = dissolved.explode(index_parts=False).reset_index(drop=True)
    zones["zona_id"] = np.arange(1, len(zones) + 1)
    zones["Textura"] = zones["texture_id"].map(USDA_CLASSES)
    return zones[["zona_id", "texture_id", "Textura", "geometry"]]


def build_sampling_points(zones):
    points = zones.copy()
    points["geometry"] = points.geometry.representative_point()
    if points.crs and points.crs.to_string() != "EPSG:4326":
        points = points.to_crs("EPSG:4326")
    points["ID_Punto"] = np.arange(1, len(points) + 1)
    points["Lon"] = points.geometry.x
    points["Lat"] = points.geometry.y
    return points[["ID_Punto", "zona_id", "texture_id", "Textura", "Lat", "Lon", "geometry"]]


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
    pixel_count,
    source_config,
    auxiliary_rasters=None,
):
    output_dir = next_output_dir()
    raster_path = output_dir / f"raster_textura_{source_config['resolution_slug']}.tif"
    zones_path = output_dir / "zonas_texturales.geojson"
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

    zones_path.write_text(zones_out.to_json(), encoding="utf-8")
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
        "point_count": int(len(points_out)),
        "outputs": {
            "raster": raster_path.name,
            "zones_geojson": zones_path.name,
            "points_geojson": points_geojson_path.name,
            "points_csv": points_csv_path.name,
            "classes_csv": classes_path.name,
        },
    }
    files = {
        "raster": raster_path,
        "zones_geojson": zones_path,
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


def process_sampling(geometry, source_key=DEFAULT_SOURCE_KEY, status_box=None):
    source_config = get_source_config(source_key)
    poly_geom = validate_polygon(geometry)
    fractions, transform, crs, pixel_count, auxiliary_rasters = read_soil_fraction_rasters(
        poly_geom,
        source_key,
        status_box,
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
    points = build_sampling_points(zones)
    files, output_dir = write_outputs(
        texture_grid,
        transform,
        crs,
        zones,
        points,
        pixel_count,
        source_config,
        auxiliary_rasters,
    )

    return {
        "zones": zones,
        "points": points,
        "files": files,
        "output_dir": output_dir,
        "valid_pixels": valid_pixels,
        "pixel_count": pixel_count,
        "source_key": source_key,
        "source_name": source_config["name"],
        "resolution_label": source_config["resolution_label"],
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


def render_result_map(result):
    zones = result["zones"]
    zones_wgs84 = zones.to_crs("EPSG:4326") if zones.crs else zones.set_crs("EPSG:4326")
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

    for _, row in points.iterrows():
        folium.Marker(
            location=[row["Lat"], row["Lon"]],
            popup=(
                f"<b>Punto {row['ID_Punto']}</b><br>"
                f"Zona: {row['zona_id']}<br>"
                f"Textura: {row['Textura']}"
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

    for label, key, mime_type in download_items:
        path = files[key]
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime_type,
            key=f"download_{key}",
        )


def main():
    st.set_page_config(page_title="Soil Point Sampler", layout="wide")

    if "polygon_geojson" not in st.session_state:
        st.session_state.polygon_geojson = None
    if "result" not in st.session_state:
        st.session_state.result = None
    if "source_key" not in st.session_state:
        st.session_state.source_key = DEFAULT_SOURCE_KEY

    st.title("Metodologia Establecimiento Puntos de Muestreo de Suelos")
    st.markdown(
        "Define puntos preliminares de muestreo dentro de un poligono usando "
        "textura USDA calculada desde fracciones reales de arena, limo y arcilla."
    )
    st.caption(
        "Fuentes reales disponibles: OpenLandMap-soildb 120 m y SoilGrids250m / "
        "ISRIC WCS. No se generan datos simulados."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Definir poligono")
        uploaded_file = st.file_uploader("Subir poligono (GeoJSON)", type=["geojson", "json"])
        if uploaded_file is not None:
            try:
                geometry = extract_geometry(json.load(uploaded_file))
                validate_polygon(geometry)
                st.session_state.polygon_geojson = geometry
                st.session_state.result = None
                st.success("Poligono cargado correctamente.")
            except (json.JSONDecodeError, ValueError) as exc:
                st.error(f"No se pudo cargar el poligono: {exc}")

        output = render_input_map()
        if output.get("last_active_drawing"):
            new_geom = output["last_active_drawing"]["geometry"]
            if st.session_state.polygon_geojson != new_geom:
                try:
                    validate_polygon(new_geom)
                    st.session_state.polygon_geojson = new_geom
                    st.session_state.result = None
                    st.success("Poligono dibujado registrado.")
                except ValueError as exc:
                    st.error(f"El dibujo no es valido: {exc}")

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
        st.info(
            f"La corrida necesita conexion a {source_config['network_host']}. Si la fuente real no "
            "responde, el proceso se detiene en lugar de inventar datos."
        )

        if st.button("Procesar muestreo", type="primary"):
            if not st.session_state.polygon_geojson:
                st.error("Dibuja o sube un poligono primero.")
            else:
                status_box = st.empty()
                try:
                    with st.spinner(f"Procesando textura desde {source_config['name']}..."):
                        st.session_state.result = process_sampling(
                            st.session_state.polygon_geojson,
                            source_key=st.session_state.source_key,
                            status_box=status_box,
                        )
                    status_box.empty()
                    st.success("Muestreo procesado y archivos creados.")
                except Exception as exc:
                    status_box.empty()
                    st.session_state.result = None
                    st.error(str(exc))

        if st.session_state.result:
            result = st.session_state.result
            st.write(f"Carpeta de salida: `{result['output_dir']}`")
            st.write(
                f"Fuente: {result['source_name']} | "
                f"Resolucion: {result['resolution_label']} | "
                f"Pixeles validos: {result['valid_pixels']:,} | "
                f"Zonas: {len(result['zones']):,} | "
                f"Puntos: {len(result['points']):,}"
            )
            certainty = result["auxiliary_rasters"].get("certainty_mask")
            if certainty:
                summary = certainty["summary"]
                st.write(
                    "Consistencia intervalo 68%: "
                    f"{summary['stable_pixels']:,} de {summary['valid_pixels']:,} pixeles "
                    f"({summary['stable_pixels_percent']}%)."
                )
            render_result_map(result)
            render_downloads(result["files"], result)


if __name__ == "__main__":
    main()
