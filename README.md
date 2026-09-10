# Point Soil Sampler

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pointsoilsampler-nzqz5m3sbjzuxxmwyappkrb.streamlit.app/)
![GitHub last commit](https://img.shields.io/github/last-commit/asoto59g/Point_Soil_Sampler)
![GitHub repo size](https://img.shields.io/github/repo-size/asoto59g/Point_Soil_Sampler)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![Data sources](https://img.shields.io/badge/Data-OpenLandMap%20%7C%20SoilGrids%20%7C%20WoSIS%20%7C%20Sentinel--2-2E7D32)

https://pointsoilsampler-nzqz5m3sbjzuxxmwyappkrb.streamlit.app/

Aplicacion Streamlit para definir puntos preliminares de muestreo de suelos dentro de un poligono, usando textura USDA calculada desde fuentes reales de arena, limo y arcilla. La app no genera datos simulados: si una fuente remota no responde, el procesamiento se detiene.

## Fuentes De Datos

### OpenLandMap-soildb 120 m

Fuente predeterminada. Usa COGs globales de OpenLandMap-soildb para arena, limo y arcilla:

- Periodo: 2020-2022
- Profundidad: 0-30 cm
- Resolucion: 120 m
- CRS: EPSG:4326
- Estadisticos usados: `mean`, `p0.16` y `p0.84`
- Intervalo de prediccion: 68%, mayor que el umbral solicitado de 60%

La textura USDA principal se calcula con las capas `mean`. La app tambien lee `p0.16` y `p0.84` y exporta una mascara `consistencia_intervalo_68_120m.tif`: valor `1` cuando la clase USDA calculada con la media coincide con la clase calculada desde ambos extremos del intervalo, y `0` cuando no coincide o no hay dato. Esta mascara es un indicador derivado de consistencia, no una probabilidad oficial de clase.

### SoilGrids250m / ISRIC WCS

Fuente alternativa global para comparar resultados. Usa el servicio WCS de ISRIC SoilGrids:

- Variables: `sand`, `silt`, `clay`
- Profundidades: 0-5 cm, 5-15 cm y 15-30 cm
- Resolucion: 250 m
- Metodo 0-30 cm: promedio ponderado por espesor de cada intervalo

### Experimental Sentinel-2 + WoSIS

Modo experimental para comparar contra las fuentes globales base. Entrena un modelo local `RandomForestRegressor` con observaciones reales WoSIS/ISRIC de arena, limo y arcilla 0-30 cm y covariables Sentinel-2 L2A.

- Entrenamiento: perfiles WoSIS con `sand`, `silt` y `clay`, filtrados a profundidad 0-30 cm y licencias publicas compatibles.
- Descarga WoSIS: consulta el WFS por teselas con reintentos para reducir respuestas grandes o mal formadas.
- Area WoSIS: busca puntos conocidos dentro de un buffer de 250 km alrededor del poligono ingresado.
- Area Sentinel-2 de entrenamiento: usa el extent total de los perfiles WoSIS encontrados para extraer covariables Sentinel-2 en esos puntos conocidos.
- Area de prediccion: estima arena, limo y arcilla solo dentro del poligono original, a 20 m, y unicamente en pixeles Sentinel-2 clasificados como suelo descubierto.
- Imagenes: escenas Sentinel-2 L2A disponibles en Microsoft Planetary Computer.
- Compuesto: usa todas las escenas Sentinel-2 encontradas para el area y selecciona por pixel la observacion mas representativa de suelo descubierto.
- Resolucion de salida: 20 m.
- Covariables: bandas visibles, NIR, SWIR e indices NDVI, SAVI, MSAVI, BSI, CI, NDWI, GEOI y BI.
- Validacion: calcula metricas internas con validacion espacial por grupos cuando hay suficientes perfiles distribuidos.

Advertencias:

- Es una prediccion experimental, no una fuente oficial ni un reemplazo de muestreo de campo.
- Sentinel-2 observa principalmente la superficie; no garantiza representar todo el intervalo 0-30 cm.
- Humedad, rastrojo, sombra, residuos de cultivo, nubosidad y cobertura vegetal pueden sesgar la estimacion.
- Si no hay suficientes perfiles WoSIS completos o pixeles Sentinel-2 de suelo descubierto, el proceso se detiene.
- Para uso operativo, comparar contra OpenLandMap/SoilGrids y revisar la incertidumbre exportada.

## Metodologia

1. El usuario dibuja un poligono sobre el mapa satelital o sube un archivo GeoJSON.
2. Antes de procesar, el poligono activo se puede guardar o descargar como GeoJSON con nombre propio.
3. La app extrae arena, limo y arcilla desde la fuente seleccionada.
4. Las fracciones se normalizan a 100% y se clasifican en textura USDA.
5. Se crea un raster de clase textural, usando `0` solo como nodata.
6. El raster se vectoriza en poligonos continuos de clase textural.
7. Cada poligono continuo de clase textural se mantiene con el tamano resultante de la clasificacion.
8. Para cada poligono continuo de clase textural se genera un solo punto en su centroide.
9. Se crean archivos descargables y una carpeta local de salida por corrida.

## Clases Texturales USDA

La tabla exportada contiene exactamente 12 clases texturales:

| ID | Clase |
| --- | --- |
| 1 | Arenosa (Sand) |
| 2 | Arenosa-franca (Loamy sand) |
| 3 | Franco-arenosa (Sandy loam) |
| 4 | Franca (Loam) |
| 5 | Franco-limosa (Silt loam) |
| 6 | Limosa (Silt) |
| 7 | Franco-areno-arcillosa (Sandy clay loam) |
| 8 | Franco-arcillosa (Clay loam) |
| 9 | Franco-limo-arcillosa (Silty clay loam) |
| 10 | Areno-arcillosa (Sandy clay) |
| 11 | Limo-arcillosa (Silty clay) |
| 12 | Arcillosa (Clay) |

## Archivos De Salida

Cada corrida crea una carpeta en `salidas/muestreo_YYYYMMDD_HHMMSS/` con:

- `raster_textura_120m.tif` para OpenLandMap-soildb
- `raster_textura_250m.tif` para SoilGrids250m
- `raster_textura_20m_sentinel_wosis.tif` para el modo experimental Sentinel-2 + WoSIS
- `consistencia_intervalo_68_120m.tif` cuando se usa OpenLandMap-soildb
- `sentinel_suelo_descubierto_observaciones.tif` cuando se usa Sentinel-2 + WoSIS
- `sentinel_suelo_descubierto_score.tif` cuando se usa Sentinel-2 + WoSIS
- `incertidumbre_modelo_sentinel_wosis.tif` cuando se usa Sentinel-2 + WoSIS
- `zonas_texturales.geojson`
- `poligonos_muestreo.geojson`
- `puntos_muestreo.geojson`
- `puntos_muestreo.csv`
- `tabla_clases_textura.csv`
- `metadata_fuente.json`

La carpeta `salidas/` esta ignorada por Git para evitar subir archivos generados al repositorio.

Los poligonos guardados antes de procesar se escriben en `salidas/poligonos/NOMBRE.geojson`.

## Regla De Muestreo

La app genera un punto por cada poligono continuo de clase textural. No subdivide poligonos mayores de 85 ha y no suma poligonos separados de la misma clase.

Cada punto se ubica en el centroide del poligono textural correspondiente. El CSV de puntos incluye `zona_id`, `texture_id`, `Textura`, `area_zona_ha`, `metodo_punto`, `Lat` y `Lon`.

## Instalacion

```bash
pip install -r requirements.txt
```

## Ejecucion

```bash
streamlit run app.py
```

La app requiere conexion a internet para leer `s3.opengeohub.org`, `maps.isric.org` y, en el modo experimental, `planetarycomputer.microsoft.com`.
