
from flask import Flask, render_template, request, jsonify
import json
import numpy as np
import pandas as pd
import pennylane as qml

app = Flask(__name__)

with open("vqc_model.json", "r") as f:
    MODEL = json.load(f)

FEATURES = MODEL["features"]
N_QUBITS = MODEL["n_qubits"]
N_LAYERS = MODEL["n_layers"]

WEIGHTS = np.asarray(MODEL["weights"], dtype=float)
READOUT = np.asarray(MODEL["readout_weights"], dtype=float)
BIAS = float(MODEL["readout_bias"])

FEATURE_MIN = np.asarray(MODEL["feature_scaler_min"], dtype=float)
FEATURE_SCALE = np.asarray(MODEL["feature_scaler_scale"], dtype=float)
TARGET_MIN = np.asarray(MODEL["target_scaler_min"], dtype=float)
TARGET_SCALE = np.asarray(MODEL["target_scaler_scale"], dtype=float)

# Load the same dataset used for the trained model.
df = pd.read_parquet("pjm_demand_weather_hourly_v1.parquet")
df["datetime_local"] = pd.to_datetime(df["datetime_local"], utc=True)
df["future_demand"] = df["demand_mwh"].shift(-1)
df["demand_24h_ago"] = df["demand_mwh"].shift(24)
df["temp_pop_wt_c"] = df["temp_pop_wt_c"].interpolate()

df = df.dropna(subset=FEATURES + ["future_demand"]).reset_index(drop=True)
df = df.sort_values("datetime_local").reset_index(drop=True)

@qml.qnode(qml.device("default.qubit", wires=N_QUBITS))
def quantum_circuit(x):
    # Angle encoding
    for q in range(N_QUBITS):
        qml.RY(x[q], wires=q)

    # Variational layers
    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            qml.RY(WEIGHTS[layer, q, 0], wires=q)
            qml.RZ(WEIGHTS[layer, q, 1], wires=q)

        for q in range(N_QUBITS - 1):
            qml.CNOT(wires=[q, q + 1])

    return [qml.expval(qml.PauliZ(q)) for q in range(N_QUBITS)]

def scale_features(x):
    return x * FEATURE_SCALE + FEATURE_MIN

def scale_target(y):
    return (y - TARGET_MIN) / TARGET_SCALE

def predict_one(row):
    raw = np.asarray([
        float(row["demand_mwh"]),
        float(row["demand_24h_ago"]),
        float(row["temp_pop_wt_c"]),
        float(row["hour"])
    ])
    scaled = raw * FEATURE_SCALE + FEATURE_MIN
    angles = scaled * np.pi
    qout = np.asarray(quantum_circuit(angles), dtype=float)
    y_scaled = float(np.dot(qout, READOUT) + BIAS)
    # inverse MinMaxScaler transform
    return float((y_scaled - TARGET_MIN[0]) / TARGET_SCALE[0])

def lookup_row(ts):
    # Exact hourly lookup after normalizing timezone.
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    hit = df[df["datetime_local"] == t]
    if len(hit):
        return hit.iloc[0]
    # nearest row within one hour
    idx = np.searchsorted(df["datetime_local"].values, t.to_datetime64())
    candidates = []
    for i in [idx-1, idx]:
        if 0 <= i < len(df):
            candidates.append((abs(df.iloc[i]["datetime_local"] - t), i))
    if candidates and min(x[0] for x in candidates) <= pd.Timedelta(hours=1):
        return df.iloc[min(candidates)[1]]
    return None

def make_forecast(start_ts, hours=168):
    start = pd.Timestamp(start_ts)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")

    # This implementation is for dates covered by the source dataset.
    # It uses the dataset's observed temperature and lagged demand.
    out = []
    work = df.set_index("datetime_local").copy()

    for h in range(hours):
        ts = start + pd.Timedelta(hours=h)

        # For a dataset-covered timestamp, use the source row.
        if ts in work.index:
            row = work.loc[ts]
            demand_now = float(row["demand_mwh"])
            lag_ts = ts - pd.Timedelta(hours=24)
            if lag_ts in work.index:
                lag = float(work.loc[lag_ts, "demand_mwh"])
            else:
                lag = demand_now
            temp = float(row["temp_pop_wt_c"])
        else:
            # Recursive fallback: last predicted demand and last known temperature.
            # This is a demo fallback for timestamps beyond the dataset.
            demand_now = float(out[-1]["predicted_mwh"]) if out else float(work["demand_mwh"].iloc[-1])
            lag_item = next((x for x in reversed(out) if x["timestamp"] == ts - pd.Timedelta(hours=24)), None)
            lag = float(lag_item["predicted_mwh"]) if lag_item else demand_now
            temp = float(work["temp_pop_wt_c"].iloc[-1])

        # The trained model expects hour in 0..23.
        hour = ts.hour
        fake_row = {
            "demand_mwh": demand_now,
            "demand_24h_ago": lag,
            "temp_pop_wt_c": temp,
            "hour": hour
        }
        pred = predict_one(fake_row)

        out.append({
            "timestamp": ts.isoformat(),
            "date": ts.strftime("%Y-%m-%d"),
            "time": ts.strftime("%H:%M"),
            "predicted_mwh": round(pred, 2)
        })
    return out

@app.route("/")
def home():
    min_date = df["datetime_local"].min().strftime("%Y-%m-%d")
    max_date = df["datetime_local"].max().strftime("%Y-%m-%d")
    return render_template("index.html", min_date=min_date, max_date=max_date)

@app.post("/api/predict")
def api_predict():
    data = request.get_json(force=True)
    date = data.get("date")
    time = data.get("time")
    if not date or not time:
        return jsonify({"error": "Select both date and time."}), 400

    ts = pd.Timestamp(f"{date} {time}", tz="UTC")
    row = lookup_row(ts)

    # For dates outside the dataset, use the recursive fallback.
    forecast = make_forecast(ts, 168)
    selected = forecast[0]

    return jsonify({
        "selected": selected,
        "forecast": forecast,
        "dataset_range": {
            "min": df["datetime_local"].min().isoformat(),
            "max": df["datetime_local"].max().isoformat()
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
