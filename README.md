# Point Soil Sampler

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

## Metodologia

1. El usuario dibuja un poligono sobre el mapa satelital o sube un archivo GeoJSON.
2. La app extrae arena, limo y arcilla desde la fuente seleccionada.
3. Las fracciones se normalizan a 100% y se clasifican en textura USDA.
4. Se crea un raster de clase textural, usando `0` solo como nodata.
5. El raster se vectoriza y se disuelven celdas colindantes con la misma clase.
6. Cada zona textural se evalua por area. Si mide hasta 85 ha, genera un punto. Si supera 85 ha, se divide en unidades de muestreo de maximo 85 ha y el remanente genera una unidad adicional.
7. Para cada unidad de muestreo se genera un punto aleatorio dentro de la geometria.
8. Se crean archivos descargables y una carpeta local de salida por corrida.

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
- `consistencia_intervalo_68_120m.tif` cuando se usa OpenLandMap-soildb
- `zonas_texturales.geojson`
- `subzonas_muestreo.geojson`
- `puntos_muestreo.geojson`
- `puntos_muestreo.csv`
- `tabla_clases_textura.csv`
- `metadata_fuente.json`

La carpeta `salidas/` esta ignorada por Git para evitar subir archivos generados al repositorio.

## Regla De Densidad De Muestreo

La app genera un punto por cada 85 ha de una misma zona textural. El calculo se realiza por zona textural continua:

- 0 a 85 ha: 1 punto
- 85.01 a 170 ha: 2 puntos
- 170.01 a 255 ha: 3 puntos

Cuando una zona supera 85 ha, se crean subzonas internas y se genera un punto aleatorio dentro de cada subzona. Las zonas que no superan 85 ha tambien usan un punto aleatorio dentro de su geometria. El CSV de puntos incluye `zona_id`, `subzona_id`, `area_zona_ha`, `area_subzona_ha`, `puntos_zona` y `metodo_punto`.

Cada vez que se ejecuta nuevamente el procesamiento se generan nuevas ubicaciones aleatorias para mejorar la variabilidad de futuras muestras, manteniendo las mismas zonas texturales y la regla de densidad.

## Instalacion

```bash
pip install -r requirements.txt
```

## Ejecucion

```bash
streamlit run app.py
```

La app requiere conexion a internet para leer `s3.opengeohub.org` y `maps.isric.org`.
