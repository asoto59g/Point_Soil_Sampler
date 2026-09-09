# Point_Soil_Sampler

Aplicación Streamlit para establecer una metodología de puntos de muestreo de suelos en un polígono basándose en la textura del suelo USDA.

## Metodología Implementada
1) **Dataset de Origen**: Basado en variables de arena, limo y arcilla del dataset OpenLandMap SoilDB (30x30m de resolución).
2) **Entrada de Datos**: Permite al usuario dibujar un polígono de interés sobre un mapa satelital o subir un archivo GeoJSON personalizado.
3) **Raster de Textura (30x30m)**: Extrae o simula las variables de textura para el polígono y crea un raster donde cada píxel corresponde a una clase textural USDA (1-12).
4) **Vectorización y Unión**: Vectoriza el raster, uniendo automáticamente las celdas colindantes que comparten el mismo valor de textura para formar polígonos o "zonas" contiguas, minimizando la cantidad total de geometrías.
5) **Generación de Puntos**: Calcula el centroide representativo de cada zona de textura identificada. Estos puntos aseguran caer siempre dentro del polígono respectivo y sirven como los puntos preliminares de muestreo en campo.

## Instalación
```bash
pip install -r requirements.txt
```

## Ejecución
```bash
streamlit run app.py
```
