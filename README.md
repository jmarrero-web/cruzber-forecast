# CRUZBER — Demand Intelligence Platform

Dashboard Streamlit para el TFM del Máster de Data Analysis e IA (ISDI).

## Datos necesarios en `data/`

1. `cruzber_prevision_global_SOP.csv` — Modelo de previsión CatBoost (89K filas)
2. `Prediccion_OOS_H12_Provincia_vB.xlsx` — Modelo de riesgo OOS LightGBM (16K filas)

## Ejecución local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy en Streamlit Cloud

1. Subir este repo a GitHub
2. Ir a [share.streamlit.io](https://share.streamlit.io)
3. New app → seleccionar repo → Deploy

## Equipo Troncal Hedy Lamarr — ISDI MDA
