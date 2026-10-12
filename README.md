# QuantumGrid AI - Level 3 Final Fixed

This version fixes the 7-day forecast behavior.

## Important fixes
- The VQC still uses the same 4 features: demand_mwh, demand_24h_ago, temp_pop_wt_c, hour.
- Recursive forecasting now uses the demand value from exactly 24 hours earlier for the second model input.
- The 7-day endpoint generates one continuous 168-hour forecast instead of restarting the same 24-hour forecast for every day.
- Date/time timestamps are normalized consistently to avoid timezone comparison errors.

## Run
1. Open this folder.
2. Run `run.bat`.
3. Open http://127.0.0.1:5000
