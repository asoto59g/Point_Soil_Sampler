# Point_Soil_Sampler
https://pointsoilsampler-nzqz5m3sbjzuxxmwyappkrb.streamlit.app/

Aplicación Streamlit para establecer una metodología de puntos de muestreo de suelos en un polígono basándose en la textura del suelo USDA.

## Metodología Implementada
1) **Dataset de Origen**: Permite elegir entre OpenLandMap-soildb (arena, limo y arcilla; 30 m) y SoilGrids250m / ISRIC WCS (arena, limo y arcilla; 250 m).
2) **Entrada de Datos**: Permite al usuario dibujar un polígono de interés sobre un mapa satelital o subir un archivo GeoJSON personalizado.
3) **Raster de Textura**: Extrae variables reales de textura desde la fuente seleccionada y crea un raster donde cada píxel corresponde a una clase textural USDA (1-12). Si la fuente real no responde, la app detiene el proceso en lugar de inventar datos.
4) **Vectorización y Unión**: Vectoriza el raster, uniendo automáticamente las celdas colindantes que comparten el mismo valor de textura para formar polígonos o "zonas" contiguas, minimizando la cantidad total de geometrías.
5) **Generación de Puntos**: Calcula el centroide representativo de cada zona de textura identificada. Estos puntos aseguran caer siempre dentro del polígono respectivo y sirven como los puntos preliminares de muestreo en campo.

## Archivos de salida
Cada corrida crea una carpeta en `salidas/muestreo_YYYYMMDD_HHMMSS/` con:

- `raster_textura_30x30m.tif`
- `zonas_texturales.geojson`
- `puntos_muestreo.geojson`
- `puntos_muestreo.csv`
- `tabla_clases_textura.csv`
- `metadata_fuente.json`

La tabla de clases exportada contiene solo las 12 clases texturales. En el raster, el valor `0` queda reservado como nodata.

## Fuentes disponibles
- **OpenLandMap-soildb 30 m**: fuente predeterminada. Usa COGs globales de arena, limo y arcilla para 0-30 cm.
- **SoilGrids250m / ISRIC WCS**: fuente alternativa global. Usa las capas `sand`, `silt` y `clay` de SoilGrids en 0-5, 5-15 y 15-30 cm, con promedio ponderado para producir 0-30 cm. Es una fuente independiente, pero su resolución espacial es 250 m.

## Instalación
```bash
pip install -r requirements.txt
```

## Ejecución
```bash
streamlit run app.py
```
