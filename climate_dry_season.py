"""Resolve dry-season months from WorldClim / CHIRPS for any AOI.

WorldClim 2.1 (10-minute precipitation climatology) is primary.
CHIRPS monthly means (sparse multi-year sample) are the fallback.
If both fail, returns Pacific Central America months (Dec-Apr).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import rasterio
from rasterio.env import Env

DEFAULT_DRY_SEASON_MONTHS = (12, 1, 2, 3, 4)
DEFAULT_N_DRIEST_MONTHS = 5
MONTH_NAMES_ES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
}

WORLDCLIM_PREC_ZIP_URL = (
    "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_prec.zip"
)
WORLDCLIM_PREC_MEMBER = "wc2.1_10m_prec_{month:02d}.tif"
CHIRPS_MONTHLY_URL = (
    "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_monthly/tifs/"
    "chirps-v2.0.{year}.{month:02d}.tif.gz"
)
CHIRPS_CLIMATOLOGY_YEARS = (1995, 2000, 2005, 2010, 2015, 2020)

GDAL_REMOTE_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.zip,.gz",
    "GDAL_HTTP_CONNECTTIMEOUT": "20",
    "GDAL_HTTP_TIMEOUT": "90",
    "GDAL_HTTP_MAX_RETRY": "2",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "50000000",
}


@dataclass(frozen=True)
class DrySeasonResolution:
    months: tuple[int, ...]
    source: str
    source_label: str
    mode: str
    longitude: float | None
    latitude: float | None
    monthly_precip_mm: dict[int, float] | None
    n_driest: int
    notes: str = ""

    def as_dict(self):
        payload = asdict(self)
        payload["months"] = list(self.months)
        payload["month_labels"] = [MONTH_NAMES_ES[m] for m in self.months]
        return payload


def normalize_months(months: Iterable[int]) -> tuple[int, ...]:
    cleaned = []
    seen = set()
    for value in months:
        month = int(value)
        if month < 1 or month > 12:
            raise ValueError(f"Mes invalido: {month}. Usa enteros 1..12.")
        if month not in seen:
            cleaned.append(month)
            seen.add(month)
    if not cleaned:
        raise ValueError("Debes seleccionar al menos un mes seco.")
    return tuple(cleaned)


def format_month_list(months: Sequence[int]) -> str:
    return ", ".join(MONTH_NAMES_ES[int(month)] for month in months)


def polygon_centroid_lonlat(poly_geom):
    centroid = poly_geom.centroid
    return float(centroid.x), float(centroid.y)


def _sample_geotiff(href: str, lon: float, lat: float) -> float:
    with Env(**GDAL_REMOTE_ENV):
        with rasterio.open(href) as dataset:
            if not (
                dataset.bounds.left <= lon <= dataset.bounds.right
                and dataset.bounds.bottom <= lat <= dataset.bounds.top
            ):
                raise ValueError(
                    f"El punto ({lon:.4f}, {lat:.4f}) esta fuera del extent "
                    f"de la capa climatica ({dataset.bounds})."
                )
            value = float(list(dataset.sample([(lon, lat)]))[0][0])
    if not np.isfinite(value) or value < -1000:
        raise ValueError(f"Precipitacion no valida en ({lon:.4f}, {lat:.4f}): {value}")
    return value


def monthly_precip_worldclim(lon: float, lat: float) -> dict[int, float]:
    monthly = {}
    for month in range(1, 13):
        member = WORLDCLIM_PREC_MEMBER.format(month=month)
        href = f"/vsizip//vsicurl/{WORLDCLIM_PREC_ZIP_URL}/{member}"
        monthly[month] = _sample_geotiff(href, lon, lat)
    return monthly


def monthly_precip_chirps(
    lon: float,
    lat: float,
    years: Sequence[int] = CHIRPS_CLIMATOLOGY_YEARS,
) -> dict[int, float]:
    totals = {month: [] for month in range(1, 13)}
    errors = []
    for year in years:
        for month in range(1, 13):
            url = CHIRPS_MONTHLY_URL.format(year=year, month=month)
            href = f"/vsigzip//vsicurl/{url}"
            try:
                totals[month].append(_sample_geotiff(href, lon, lat))
            except Exception as exc:
                errors.append(f"{year}-{month:02d}: {exc}")
    monthly = {}
    for month, values in totals.items():
        if not values:
            raise RuntimeError(
                "CHIRPS no devolvio precipitacion mensual suficiente. "
                f"Ejemplos: {'; '.join(errors[:3]) or 'sin detalle'}"
            )
        monthly[month] = float(np.mean(values))
    return monthly


def driest_months_from_precip(
    monthly_precip_mm: dict[int, float],
    n_driest: int = DEFAULT_N_DRIEST_MONTHS,
) -> tuple[int, ...]:
    if n_driest < 1 or n_driest > 12:
        raise ValueError("n_driest debe estar entre 1 y 12.")
    ordered = sorted(monthly_precip_mm.items(), key=lambda item: (item[1], item[0]))
    return tuple(month for month, _ in ordered[:n_driest])


def resolve_dry_season_months(
    poly_geom=None,
    mode: str = "auto",
    manual_months: Sequence[int] | None = None,
    n_driest: int = DEFAULT_N_DRIEST_MONTHS,
    lon: float | None = None,
    lat: float | None = None,
) -> DrySeasonResolution:
    """Resolve dry-season months for an AOI.

    mode:
      - auto: WorldClim -> CHIRPS -> Central America default
      - manual: user-selected months
      - default_ca: force Pacific Central America months
    """
    mode = (mode or "auto").strip().lower()
    if mode not in {"auto", "manual", "default_ca"}:
        raise ValueError("mode debe ser auto, manual o default_ca.")

    if mode == "manual":
        months = normalize_months(manual_months or ())
        return DrySeasonResolution(
            months=months,
            source="manual",
            source_label="Seleccion manual",
            mode=mode,
            longitude=None,
            latitude=None,
            monthly_precip_mm=None,
            n_driest=len(months),
            notes="Meses secos definidos por el usuario.",
        )

    if mode == "default_ca":
        return DrySeasonResolution(
            months=DEFAULT_DRY_SEASON_MONTHS,
            source="default_ca",
            source_label="Default Centroamerica pacifica",
            mode=mode,
            longitude=None,
            latitude=None,
            monthly_precip_mm=None,
            n_driest=len(DEFAULT_DRY_SEASON_MONTHS),
            notes="Calendario fijo dic-abr (Centroamerica / Pacifico).",
        )

    if lon is None or lat is None:
        if poly_geom is None:
            raise ValueError("Se requiere poly_geom o lon/lat para el modo automatico.")
        lon, lat = polygon_centroid_lonlat(poly_geom)

    errors = []
    try:
        monthly = monthly_precip_worldclim(lon, lat)
        months = driest_months_from_precip(monthly, n_driest=n_driest)
        return DrySeasonResolution(
            months=months,
            source="worldclim_2.1_10m",
            source_label="WorldClim 2.1 (10 min, precipitacion)",
            mode=mode,
            longitude=lon,
            latitude=lat,
            monthly_precip_mm={int(k): round(float(v), 2) for k, v in monthly.items()},
            n_driest=n_driest,
            notes="Meses mas secos de la climatologia WorldClim en el centroide del AOI.",
        )
    except Exception as exc:
        errors.append(f"WorldClim: {exc}")

    try:
        monthly = monthly_precip_chirps(lon, lat)
        months = driest_months_from_precip(monthly, n_driest=n_driest)
        return DrySeasonResolution(
            months=months,
            source="chirps_v2_monthly_mean",
            source_label="CHIRPS v2 (media mensual multi-anual)",
            mode=mode,
            longitude=lon,
            latitude=lat,
            monthly_precip_mm={int(k): round(float(v), 2) for k, v in monthly.items()},
            n_driest=n_driest,
            notes=(
                "Fallback CHIRPS tras fallo de WorldClim. "
                f"Detalle: {errors[-1] if errors else 'n/d'}"
            ),
        )
    except Exception as exc:
        errors.append(f"CHIRPS: {exc}")

    return DrySeasonResolution(
        months=DEFAULT_DRY_SEASON_MONTHS,
        source="default_ca_fallback",
        source_label="Default Centroamerica pacifica (fallback)",
        mode=mode,
        longitude=lon,
        latitude=lat,
        monthly_precip_mm=None,
        n_driest=len(DEFAULT_DRY_SEASON_MONTHS),
        notes="WorldClim y CHIRPS fallaron; se uso dic-abr. " + " | ".join(errors),
    )
