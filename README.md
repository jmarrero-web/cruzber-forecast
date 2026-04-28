# CRUZBER — Demand Intelligence Platform v10

Dashboard Streamlit — TFM ISDI (Equipo Hedy Lamarr)

## Datos en `data/`

1. `Prediccion_SnOP_NB39_S_.xlsx` — Validación S1-S27
2. `Forecast_S28_S39_2024_NB39.xlsx` — Predicción S28-S39
3. `pred_h12_por_provincia_sku_v3_ROI.csv` — Stock & ROI por provincia
4. `Prediccion_SnOP_NB29_v2_PROD.xlsx` — Descripciones de producto (opcional)

## Ejecución

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Secrets (Streamlit Cloud)

```toml
COHERE_API_KEY = "tu-key"
```
