# QuantumGrid AI — Electricity Demand Forecasting

This project uses the trained VQC parameters exported from Kaggle.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000

## Files
- `app.py` — Flask backend and VQC inference
- `templates/index.html` — forecast-only UI
- `vqc_model.json` — trained VQC parameters/scalers
- `pjm_demand_weather_hourly_v1.parquet` — source dataset

## Important
The model was trained for one-step-ahead demand prediction using:
- demand_mwh
- demand_24h_ago
- temp_pop_wt_c
- hour

The UI derives hour from the selected time. For timestamps covered by the dataset, the backend uses the dataset's demand/weather values. For timestamps beyond the dataset, it uses a recursive demo fallback and carries forward the last known temperature; this should not be presented as a real weather-based future forecast without a weather forecast source.
