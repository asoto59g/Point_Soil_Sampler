<img width="1667" height="469" alt="preview2" src="https://github.com/user-attachments/assets/91833743-81aa-43ab-bb8e-6ee9654ff37d" />

# Point Soil Sampler

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pointsoilsampler-nzqz5m3sbjzuxxmwyappkrb.streamlit.app/)
![GitHub last commit](https://img.shields.io/github/last-commit/asoto59g/Point_Soil_Sampler)
![GitHub repo size](https://img.shields.io/github/repo-size/asoto59g/Point_Soil_Sampler)
![GitHub License](https://img.shields.io/github/license/asoto59g/Point_Soil_Sampler)
![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![Data sources](https://img.shields.io/badge/Data-OpenLandMap%20%7C%20SoilGrids%20%7C%20WoSIS%20%7C%20Calicatas%20CR%20%7C%20Sentinel--2%20%7C%20Sentinel--1-2E7D32)

https://pointsoilsampler-nzqz5m3sbjzuxxmwyappkrb.streamlit.app/

## Problema Que Resuelve

Point Soil Sampler ayuda a definir puntos preliminares de muestreo de suelos dentro de un poligono agricola. La app estima textura USDA desde fuentes reales de arena, limo y arcilla, vectoriza zonas texturales continuas y genera puntos, tablas y archivos GIS descargables.

La app no genera datos simulados: si una fuente remota no responde o no existen datos suficientes, el procesamiento se detiene.

## Innovacion O Aporte Tecnico

- Integra una interfaz Streamlit con fuentes globales de textura y salidas GIS listas para revision.
- Clasifica textura USDA a partir de fracciones normalizadas de arena, limo y arcilla.
- Compara fuentes globales base: OpenLandMap-soildb 120 m y SoilGrids250m / ISRIC WCS.
- Incluye un modo experimental Sentinel-2/1 + perfiles locales que entrena **3 `RandomForestRegressor` independientes** (arena/limo/arcilla, normalizados a 100%) con WoSIS y/o calicatas Costa Rica, covariables Sentinel-2 L2A (incluye red-edge), Sentinel-1 RTC (VV/VH) y DEM.
- Exporta capas auxiliares de consistencia, observaciones Sentinel-2 de suelo descubierto, score de suelo descubierto e incertidumbre del modelo experimental.
- Reporta validacion espacial por bloques (MAE/R² por fraccion) y las features mas importantes del RF dentro de la UI.

## Mejoras Recientes Y Alcances

Resumen de lo incorporado en el modo experimental y de lo que el sistema **si** / **no** pretende cubrir.

### Mejoras entregadas

| Area | Mejora | Alcance practico |
| --- | --- | --- |
| Entrenamiento local | Calicatas Costa Rica (`Calicatas_01_02_21_Costa_Rica.csv`) como fuente de perfiles 0-30 cm | Perfiles nacionales; se filtran a buffer de 250 km y a los ~400 mas cercanos al poligono |
| Selector de perfiles | `Solo WoSIS/ISRIC`, `Solo calicatas Costa Rica` o `WoSIS + calicatas Costa Rica` | Permite comparar o combinar inventario global vs local CR |
| Modelo | 3 RF independientes + normalizacion a 100% | Evita un unico multi-output; predice fracciones coherentes para USDA |
| Espectro Sentinel-2 | Bandas red-edge `B05`/`B06`/`B07`/`B8A` e indices `NDRE`/`NDRE2` | Mejor sensibilidad a vegetacion/suelo fino frente a solo VNIR/SWIR |
| Radar Sentinel-1 | Coleccion RTC `VV_DB`, `VH_DB`, `VV_VH_DB` (mediana lineal → dB) | Covariable de rugosidad/humedad superficial independiente de nubes opticas |
| Relieve | DEM CR (Drive / CRTM05) o Copernicus GLO-30; features `ELEV`, `SLOPE_DEG`, `ASPECT_SIN`, `ASPECT_COS`, `CURV` | Sustituye `LON`/`LAT` para forzar aprendizaje espectro + topografia + radar |
| Bounds de entrenamiento | Union del envelope de perfiles + AOI al extraer DEM/S1 | Evita NaN en perfiles del buffer y corridas con 0 filas validas en AOIs pequenos |
| Suelo descubierto | Prioridad estacion seca dinamica (WorldClim/CHIRPS o selector manual) + umbrales NDVI/NDWI/BSI estrictos | Adapta meses secos al AOI en cualquier region; override manual disponible |
| Validacion | CV espacial por bloques: MAE ± std entre folds, R² y top features en la UI | Diagnostico exploratorio; no es certificacion de exactitud de campo |
| Operacion / smoke | Variables de entorno para limitar escenas, arboles RF y escenas S1 | Acelera pruebas E2E sin cambiar el codigo |

### Vector de features del RF experimental (50)

- Espectrales `_best` y `_mean` (20 + 20): `B02`–`B04`, `B05`–`B07`, `B08`, `B8A`, `B11`, `B12`, `NDVI`, `SAVI`, `MSAVI`, `BSI`, `CI`, `NDWI`, `GEOI`, `BI`, `NDRE`, `NDRE2`
- Suelo descubierto: `BARE_OBS`, `BARE_SCORE`
- DEM: `ELEV`, `SLOPE_DEG`, `ASPECT_SIN`, `ASPECT_COS`, `CURV`
- Sentinel-1 RTC: `VV_DB`, `VH_DB`, `VV_VH_DB`

### Alcances (que cubre)

- Diseno **preliminar** de puntos de muestreo a partir de textura USDA derivada de fracciones reales.
- Comparacion entre OpenLandMap, SoilGrids y el modelo experimental Sentinel + perfiles.
- Uso tipico en Costa Rica con calicatas locales + DEM nacional; tambien operable fuera de CR con WoSIS + Copernicus GLO-30.
- Prediccion experimental a **20 m** solo sobre pixeles clasificados como suelo descubierto dentro del poligono.
- Export GIS (raster, GeoJSON, CSV, metadata) y capas auxiliares de calidad/incertidumbre.

### Fuera de alcance (que no cubre)

- No sustituye muestreo de campo, laboratorio ni cartografia oficial de suelos.
- No garantiza representar todo el intervalo 0–30 cm (Sentinel observa la superficie).
- No es un producto calibrado/certificado: el CV espacial es interno y puede mostrar R² bajos o MAE altos en AOIs pequenos o con pocos perfiles.
- No incluye API keys ni autenticacion de Google Earth Engine en la configuracion actual.
- Corridas “completas” (sin caps de entorno) pueden superar ampliamente 20 minutos segun area, nubosidad y red.

### Validacion E2E realizada

Sobre un AOI diminuto cerca de Liberia, Costa Rica, con caps reducidos y entrenamiento `Solo calicatas Costa Rica`:

- Pipeline Python: fracciones arena/limo/arcilla finitas; modelo de 3 RF normalizados; 50 features; CV espacial 5 folds con MAE medio del orden de **~10 pp**.
- Streamlit UI: carga GeoJSON → fuente experimental → calicatas CR → procesar → panel de CV espacial y salidas en `salidas/`.

### Arquitectura de modulos

| Archivo | Rol |
| --- | --- |
| `app.py` | UI Streamlit, jobs en segundo plano, clasificacion USDA, vectorizacion y descargas |
| `experimental_sentinel_wosis.py` | Pipeline experimental: perfiles, S2, RF×3, prediccion, metricas CV |
| `climate_dry_season.py` | Meses secos por AOI via WorldClim/CHIRPS + modos auto/manual/default CA |
| `s1_covariates.py` | Busqueda/lectura Sentinel-1 RTC y features VV/VH |
| `dem_covariates.py` | DEM CR / GLO-30 y derivadas de relieve |
| `cr_dem_remote.py` | Acceso al MDE publico de Costa Rica (Google Drive / CRTM05) |
| `Calicatas_01_02_21_Costa_Rica.csv` | Perfiles de entrenamiento locales 0–30 cm |

## Metodologia Y Algoritmo

1. El usuario dibuja un poligono sobre el mapa satelital o sube un archivo GeoJSON.
2. Antes de procesar, el poligono activo se puede guardar o descargar como GeoJSON con nombre propio.
3. La app extrae arena, limo y arcilla desde la fuente seleccionada.
4. Las fracciones se normalizan a 100% y se clasifican en textura USDA.
5. Se crea un raster de clase textural, usando `0` solo como nodata.
6. El raster se vectoriza en poligonos continuos de clase textural.
7. Cada poligono continuo de clase textural se mantiene con el tamano resultante de la clasificacion.
8. Para cada poligono continuo de clase textural se genera un solo punto en su centroide.
9. Se crean archivos descargables y una carpeta local de salida por corrida.

En el modo experimental, el paso 3 se reemplaza por: (a) carga de perfiles, (b) extraccion de covariables Sentinel-2/1 + DEM en perfiles, (c) entrenamiento de 3 RF, (d) compuesto de suelo descubierto en el AOI, (e) prediccion + suavizado ligero de fracciones.

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

### Experimental Sentinel-2/1 + perfiles (WoSIS / calicatas CR)

Modo experimental para comparar contra las fuentes globales base. Entrena tres `RandomForestRegressor` independientes (arena, limo y arcilla) con observaciones 0-30 cm y covariables Sentinel-2 L2A + Sentinel-1 RTC + DEM; las predicciones se normalizan para sumar 100% antes de clasificar USDA.

- Entrenamiento: perfiles WoSIS y/o calicatas Costa Rica (`Calicatas_01_02_21_Costa_Rica.csv`) con `sand`, `silt` y `clay`, filtrados a profundidad 0-30 cm.
- Modelo: un RF por fraccion (no un unico multi-output); post-normalizacion `sum_to_100`; por defecto 300 arboles por RF (`SENTINEL_RF_TREES`).
- Selector en la app: `Solo WoSIS/ISRIC`, `Solo calicatas Costa Rica` o `WoSIS + calicatas Costa Rica` (predeterminado).
- Calicatas CR: ~7,434 perfiles usables a nivel nacional; se filtran al buffer de 250 km y se limitan a los ~400 mas cercanos al poligono para acotar Sentinel-2.
- La columna `Clase Textural` del CSV se conserva como referencia; los RF predicen fracciones y la app clasifica USDA despues.
- Descarga WoSIS: consulta el WFS por teselas con reintentos para reducir respuestas grandes o mal formadas.
- Area de entrenamiento: busca puntos conocidos dentro de un buffer de 250 km alrededor del poligono ingresado.
- Area Sentinel-2 de entrenamiento: usa el extent total de los perfiles encontrados para extraer covariables Sentinel-2 en esos puntos conocidos.
- Area DEM/S1 de entrenamiento: une el bounding box de los perfiles con el del AOI (mas un margen) para que los perfiles del buffer reciban elevacion y backscatter finitos.
- Area de prediccion: estima arena, limo y arcilla solo dentro del poligono original, a 20 m, y unicamente en pixeles Sentinel-2 clasificados como suelo descubierto.
- Imagenes: escenas Sentinel-2 L2A disponibles en Microsoft Planetary Computer.
- Seleccion Sentinel-2: prioriza escenas de **estacion seca** resuelta por AOI (WorldClim 2.1 precipitacion; fallback CHIRPS; override manual o calendario fijo Centroamerica), luego menor nubosidad; para entrenamiento tambien prioriza cobertura de perfiles. Limite por defecto: 120 escenas de entrenamiento y 90 de prediccion. Nubosidad de escena `< 60%`.
- Tiempo de ejecucion: la creacion del modelo Sentinel puede tardar bastante mas de 20 minutos porque descarga, lee y procesa muchas escenas y bandas para entrenamiento y prediccion.
- Acceso Sentinel-2: firma cada asset de Planetary Computer justo antes de leerlo y exige una vigencia minima para evitar tokens vencidos en corridas largas.
- Compuesto de suelo descubierto (estricto): SCL clase 5 como criterio principal con NDVI ≤ 0.25, NDWI < 0.05 y BSI > −0.10; respaldo restringido con SCL clase 7 (NDVI ≤ 0.18). Las observaciones de estacion seca reciben un bonus en el score de seleccion de la mejor observacion; se combinan mejor observacion y media multitemporal.
- Resolucion de salida: 20 m.
- Covariables: 50 features (ver seccion **Vector de features** arriba).
- Sentinel-1: coleccion `sentinel-1-rtc` de Planetary Computer; mediana temporal de VV/VH en potencia lineal, convertida a dB; `VV_VH_DB = VV_DB - VH_DB`.
- DEM en Costa Rica: MDE publico del proyecto [Runoff_CRC](https://github.com/asoto59g/Runoff_CRC) (Google Drive, CRTM05 / EPSG:5367). Si falla o el extent es demasiado grande para el mosaico remoto, usa Copernicus GLO-30.
- DEM fuera de Costa Rica: Copernicus DEM GLO-30 via Microsoft Planetary Computer.
- Las coordenadas `LON`/`LAT` ya no se usan como features, para forzar aprendizaje espectro + topografia + radar.
- Validacion: calcula metricas internas con validacion espacial por bloques (MAE y R² por fraccion, desviacion entre folds) cuando hay suficientes perfiles distribuidos; la app muestra un panel con MAE arena/limo/arcilla y las features mas importantes del RF.
- Postproceso: suaviza ligeramente las fracciones arena/limo/arcilla antes de clasificar USDA para reducir ruido salpicado de pixeles aislados.

## Datos De Entrada Y Formatos Soportados

- Poligono dibujado en el mapa de la app.
- Archivo `.geojson` o `.json` con geometria `Polygon` o `MultiPolygon`.
- Coordenadas de entrada esperadas en WGS84 / EPSG:4326.

No se requieren API keys, tokens, credenciales de Google Earth Engine ni archivos de autenticacion para la configuracion actual.

## Instalacion Y Ejecucion

```bash
pip install -r requirements.txt
streamlit run app.py
```

La app requiere conexion a internet para leer `s3.opengeohub.org`, `maps.isric.org` y, en el modo experimental, `planetarycomputer.microsoft.com` y (para DEM de Costa Rica) el MDE publico en Google Drive usado por [Runoff_CRC](https://github.com/asoto59g/Runoff_CRC).

Las corridas largas se ejecutan como procesos en segundo plano dentro del servidor Streamlit. Si el navegador se desconecta temporalmente, por ejemplo al apagar la pantalla, el proceso puede continuar mientras el equipo y el servidor Streamlit sigan activos.

### Variables de entorno (modo experimental / smoke)

Opcionales. Sirven para acotar corridas de prueba; los valores por defecto son los de uso exploratorio normal:

| Variable | Default | Uso |
| --- | --- | --- |
| `SENTINEL_MAX_TRAINING_ITEMS` | `120` | Max. escenas S2 para entrenamiento |
| `SENTINEL_MAX_PREDICTION_ITEMS` | `90` | Max. escenas S2 para prediccion/compuesto |
| `SENTINEL_MIN_TRAINING_ITEMS` | `25` | Minimo de escenas S2 de entrenamiento |
| `SENTINEL_MIN_PREDICTION_ITEMS` | `25` | Minimo de escenas S2 de prediccion |
| `SENTINEL_TRAINING_TARGET_VALID_SAMPLES` | `180` | Meta de perfiles validos con suelo descubierto |
| `SENTINEL_TARGET_BARE_PIXEL_PERCENT` | `85.0` | Meta de % AOI con suelo descubierto |
| `SENTINEL_TARGET_MEAN_BARE_OBSERVATIONS` | `3.0` | Meta de observaciones descubiertas/pixel (prediccion) |
| `SENTINEL_TRAINING_TARGET_MEAN_BARE_OBSERVATIONS` | `2.0` | Meta de observaciones/perfil (entrenamiento) |
| `SENTINEL_RF_TREES` | `300` | Arboles por cada RF (arena/limo/arcilla) |
| `S1_MAX_ITEMS` | `30` | Max. escenas Sentinel-1 RTC |

Ejemplo de smoke E2E (AOI pequeno):

```bash
export SENTINEL_MAX_TRAINING_ITEMS=12
export SENTINEL_MAX_PREDICTION_ITEMS=8
export SENTINEL_MIN_TRAINING_ITEMS=4
export SENTINEL_MIN_PREDICTION_ITEMS=3
export SENTINEL_TARGET_BARE_PIXEL_PERCENT=5
export SENTINEL_TARGET_MEAN_BARE_OBSERVATIONS=0.5
export SENTINEL_TRAINING_TARGET_VALID_SAMPLES=15
export SENTINEL_TRAINING_TARGET_MEAN_BARE_OBSERVATIONS=0.5
export SENTINEL_RF_TREES=40
export S1_MAX_ITEMS=6
streamlit run app.py
```

## Ejemplo Visual

![Mapa conceptual de zonas texturales y puntos de muestreo](docs/example-map.svg)

El mapa de la app permite cargar o dibujar el poligono, ejecutar la clasificacion y visualizar zonas texturales junto con los puntos propuestos.

## Archivos De Salida

Cada corrida crea una carpeta en `salidas/muestreo_YYYYMMDD_HHMMSS/` con:

- `raster_textura_120m.tif` para OpenLandMap-soildb
- `raster_textura_250m.tif` para SoilGrids250m
- `raster_textura_20m_sentinel_wosis.tif` para el modo experimental Sentinel-2/1 + perfiles
- `consistencia_intervalo_68_120m.tif` cuando se usa OpenLandMap-soildb
- `sentinel_suelo_descubierto_observaciones.tif` cuando se usa el modo experimental
- `sentinel_suelo_descubierto_score.tif` cuando se usa el modo experimental
- `incertidumbre_modelo_sentinel_wosis.tif` cuando se usa el modo experimental
- `zonas_texturales.geojson`
- `poligonos_muestreo.geojson`
- `puntos_muestreo.geojson`
- `puntos_muestreo.csv`
- `tabla_clases_textura.csv`
- `metadata_fuente.json`

La carpeta `salidas/` esta ignorada por Git para evitar subir archivos generados al repositorio. Los poligonos guardados antes de procesar se escriben en `salidas/poligonos/NOMBRE.geojson`.

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

## Regla De Muestreo

La app genera un punto por cada poligono continuo de clase textural. No subdivide poligonos mayores de 85 ha y no suma poligonos separados de la misma clase.

Cada punto se ubica en el centroide del poligono textural correspondiente. El CSV de puntos incluye `zona_id`, `texture_id`, `Textura`, `area_zona_ha`, `metodo_punto`, `Lat` y `Lon`.

## Limitaciones Tecnicas

- Las fuentes globales no sustituyen muestreo de campo, cartografia local ni validacion agronomica.
- Sentinel-2 observa principalmente la superficie; no garantiza representar todo el intervalo 0-30 cm.
- Humedad, rastrojo, sombra, residuos de cultivo, nubosidad y cobertura vegetal pueden sesgar la estimacion.
- El modo Sentinel experimental depende de que existan suficientes perfiles (WoSIS y/o calicatas CR) y pixeles Sentinel-2 de suelo descubierto.
- En AOIs muy pequenos o con pocos perfiles validos, el CV espacial puede mostrar MAE alto o R² negativo; eso es senal de incertidumbre, no un fallo silencioso.
- Si el mosaico DEM CR no cubre el extent de entrenamiento (demasiadas teselas), el pipeline puede caer a Copernicus GLO-30 automaticamente.
- Las capas remotas pueden cambiar, quedar temporalmente fuera de servicio o limitar respuestas.
- En ejecucion local, el equipo no debe entrar en suspension durante corridas Sentinel largas; apagar solo la pantalla no deberia detener el job si el servidor Streamlit sigue activo.
- Para uso operativo, conviene comparar OpenLandMap/SoilGrids/Sentinel experimental y revisar incertidumbre o consistencia exportada.

## Roadmap

Completado recientemente:

- [x] Calicatas Costa Rica como fuente de entrenamiento seleccionable.
- [x] DEM (CR Drive / GLO-30) en lugar de LON/LAT.
- [x] Tres RF independientes + normalizacion a 100%.
- [x] Red-edge Sentinel-2 (`B05`–`B07`, `B8A`, NDRE).
- [x] Sentinel-1 RTC VV/VH.
- [x] Compuesto bare-soil mas estricto con prioridad de estacion seca.
- [x] Panel UI de CV espacial (MAE/R²) y top features.
- [x] Correccion de bounds DEM/S1 para perfiles del buffer.
- [x] Caps por variables de entorno para smoke/E2E.
- [x] Estacion seca global por WorldClim/CHIRPS + selector manual en Streamlit.

Pendiente:

- Agregar pruebas automatizadas para clasificacion USDA, validacion de GeoJSON y escritura de salidas.
- Incorporar un ejemplo reproducible con datos sinteticos o anonimizados.
- Permitir configurar parametros del modo experimental desde la interfaz (hoy via env / constantes).
- Mejorar reportes de calidad del modelo (mapas de residuos, curvas de aprendizaje espacial).
- Agregar soporte opcional para otros criterios de distribucion de puntos.
- Evaluar caches locales de escenas STAC para acortar corridas repetidas sobre el mismo AOI.

## Seguridad Y Publicacion

- No se versionan credenciales, secretos, archivos `.env`, credenciales de Google Earth Engine ni archivos de autenticacion.
- No se versionan resultados generados, GeoTIFF, shapefiles, geopackages, ortofotos, DEM, PDFs de referencia ni GeoJSON con datos reales de clientes.
- Antes de publicar cambios con datos reales, revisar tambien el historial de Git. Borrar un archivo en un commit no lo elimina de commits anteriores.

## Licencia Y Citacion Sugerida

Este proyecto se distribuye bajo licencia MIT. Ver [LICENSE](LICENSE).

Citacion sugerida:

```text
Soto Barquero, A. (2026). Point Soil Sampler: Streamlit app for preliminary soil texture sampling design. GitHub repository: https://github.com/asoto59g/Point_Soil_Sampler
```
