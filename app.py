
from flask import Flask, render_template, request, jsonify
import json
import os
import numpy as np
import pandas as pd
import pennylane as qml

BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True


@app.after_request
def no_cache(resp):
    resp.headers['Cache-Control'] = 'no-store'
    return resp


# --------------------------------------------------
# LOAD SAVED VQC MODEL
# --------------------------------------------------

with open(os.path.join(BASE, 'vqc_model.json'), 'r') as f:
    MODEL = json.load(f)


# Load only the columns required by the website.
# This reduces unnecessary memory usage.
DATA_PATH = os.path.join(
    BASE, 'pjm_demand_weather_hourly_v1.parquet'
)

DF = pd.read_parquet(
    DATA_PATH,
    columns=[
        'datetime_local',
        'demand_mwh',
        'temp_pop_wt_c'
    ]
)

DF['datetime_local'] = (
    pd.to_datetime(
        DF['datetime_local'],
        errors='coerce',
        utc=True
    )
    .dt.tz_convert('America/New_York')
    .dt.tz_localize(None)
)

DF = DF.dropna(
    subset=['datetime_local', 'demand_mwh']
)

DF = DF.sort_values(
    'datetime_local'
).reset_index(drop=True)

DF['temp_pop_wt_c'] = (
    DF['temp_pop_wt_c']
    .interpolate()
    .bfill()
    .ffill()
)

DF['demand_24h_ago'] = DF['demand_mwh'].shift(24)

DF = DF.dropna(
    subset=[
        'demand_mwh',
        'demand_24h_ago',
        'temp_pop_wt_c'
    ]
).reset_index(drop=True)

# Use datetime as an index for faster historical lookups.
# Keep the column too because other functions use it.
DF = DF.set_index(
    'datetime_local',
    drop=False
)


# --------------------------------------------------
# LOAD TRAINED MODEL PARAMETERS
# --------------------------------------------------

WEIGHTS = np.array(
    MODEL['weights'],
    dtype=float
)

READOUT_W = np.array(
    MODEL['readout_weights'],
    dtype=float
)

READOUT_B = float(
    MODEL['readout_bias']
)

S_MIN = np.array(
    MODEL['feature_scaler_min'],
    dtype=float
)

S_SCALE = np.array(
    MODEL['feature_scaler_scale'],
    dtype=float
)

T_MIN = np.array(
    MODEL['target_scaler_min'],
    dtype=float
)

T_SCALE = np.array(
    MODEL['target_scaler_scale'],
    dtype=float
)


# --------------------------------------------------
# QUANTUM CIRCUIT
# --------------------------------------------------

try:
    DEV = qml.device(
        'lightning.qubit',
        wires=4
    )
except Exception:
    DEV = qml.device(
        'default.qubit',
        wires=4
    )


@qml.qnode(DEV)
def quantum_circuit(inputs, weights):

    # Angle encoding
    for i in range(4):
        qml.RY(inputs[i], wires=i)

    # Two trainable variational layers
    for layer in range(2):

        for i in range(4):
            qml.RY(
                weights[layer, i, 0],
                wires=i
            )

            qml.RZ(
                weights[layer, i, 1],
                wires=i
            )

        # Entanglement between neighboring qubits
        for i in range(3):
            qml.CNOT(wires=[i, i + 1])

    return [
        qml.expval(qml.PauliZ(i))
        for i in range(4)
    ]


# --------------------------------------------------
# SCALING AND PREDICTION
# --------------------------------------------------

def scale_x(x):
    # Same MinMax scaling used during training.
    return (
        np.asarray(x, dtype=float) * S_SCALE
        + S_MIN
    )


def inverse_y(y):
    return (
        float(y) - T_MIN[0]
    ) / T_SCALE[0]


def predict_features(
    demand,
    demand24,
    temp,
    hour
):
    x_scaled = scale_x([
        demand,
        demand24,
        temp,
        hour
    ])

    angles = x_scaled * np.pi

    quantum_output = np.asarray(
        quantum_circuit(angles, WEIGHTS),
        dtype=float
    )

    output = float(
        np.dot(quantum_output, READOUT_W)
        + READOUT_B
    )

    pred_scaled = (
        np.tanh(output) + 1.0
    ) / 2.0

    return inverse_y(pred_scaled)


# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

DATA_START = DF['datetime_local'].min()
DATA_END = DF['datetime_local'].max()

LAST_TEMP = float(
    DF['temp_pop_wt_c'].iloc[-1]
)


# --------------------------------------------------
# HISTORICAL DATA LOOKUPS
# --------------------------------------------------

def get_temp(ts):

    if ts in DF.index:
        value = DF.at[
            ts,
            'temp_pop_wt_c'
        ]

        # Handle duplicate timestamps safely.
        if isinstance(value, (pd.Series, pd.DataFrame)):
            value = value.iloc[0]

        return float(value)

    # Future demo forecasts use the latest known temperature.
    return LAST_TEMP


def get_demand(ts):

    if ts in DF.index:
        value = DF.at[
            ts,
            'demand_mwh'
        ]

        # Handle duplicate timestamps safely.
        if isinstance(value, (pd.Series, pd.DataFrame)):
            value = value.iloc[0]

        return float(value)

    return None


def get_start_demand(ts):

    demand = get_demand(ts)

    if demand is not None:
        return demand, 'dataset'

    # Find the most recent observation before the requested time.
    candidates = DF[
        DF['datetime_local'] <= ts
    ]

    if not candidates.empty:
        return (
            float(candidates['demand_mwh'].iloc[-1]),
            'demo-fallback'
        )

    return (
        float(DF['demand_mwh'].iloc[0]),
        'demo-fallback'
    )


# --------------------------------------------------
# FORECAST GENERATION
# --------------------------------------------------

def forecast_from(start_ts, hours=24):

    current_demand, mode = get_start_demand(
        start_ts
    )

    history = {
        start_ts: float(current_demand)
    }

    rows = []

    for step in range(1, hours + 1):

        ts = start_ts + pd.Timedelta(
            hours=step
        )

        lag_ts = ts - pd.Timedelta(
            hours=24
        )

        # Demand from exactly 24 hours earlier.
        demand24 = history.get(lag_ts)

        if demand24 is None:
            demand24 = get_demand(lag_ts)

        if demand24 is None:
            demand24 = current_demand

        temp = get_temp(ts)
        hour = ts.hour

        pred = predict_features(
            current_demand,
            demand24,
            temp,
            hour
        )

        pred = float(pred)

        history[ts] = pred

        rows.append({
            'datetime': ts,
            'prediction': pred,
            'temperature': float(temp),
            'hour': hour
        })

        current_demand = pred

    return rows, mode


# --------------------------------------------------
# INPUT SNAPSHOT
# --------------------------------------------------

def input_snapshot(ts):

    demand, mode = get_start_demand(ts)

    demand24 = get_demand(
        ts - pd.Timedelta(hours=24)
    )

    if demand24 is None:
        demand24 = demand

    temp = get_temp(ts)

    return {
        'datetime': ts.strftime(
            '%Y-%m-%d %H:%M'
        ),
        'display_datetime': ts.strftime(
            '%d %b %Y, %I:%M %p'
        ),
        'demand': round(demand, 2),
        'demand24': round(demand24, 2),
        'temperature': round(temp, 2),
        'hour': ts.hour,
        'mode': mode
    }


# --------------------------------------------------
# MODEL PERFORMANCE METRICS
# --------------------------------------------------

def metrics_for_demo():

    # The predictions CSV is in the repository root.
    path = os.path.join(
        BASE,
        'vqc_predictions.csv'
    )

    if os.path.exists(path):

        data = pd.read_csv(path)

        actual = data[
            'Actual_Demand_MWh'
        ].to_numpy()

        pred = data[
            'VQC_Predicted_MWh'
        ].to_numpy()

        mae = float(
            np.mean(np.abs(actual - pred))
        )

        rmse = float(
            np.sqrt(
                np.mean((actual - pred) ** 2)
            )
        )

        ssr = float(
            np.sum((actual - pred) ** 2)
        )

        sst = float(
            np.sum(
                (actual - actual.mean()) ** 2
            )
        )

        r2 = (
            float(1 - ssr / sst)
            if sst
            else 0.0
        )

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

    return {
        'mae': None,
        'rmse': None,
        'r2': None
    }


# --------------------------------------------------
# WEBSITE PAGES
# --------------------------------------------------

@app.route('/')
def home():

    default = DATA_END.floor('h')

    return render_template(
        'home.html',
        default_date=default.strftime('%Y-%m-%d'),
        default_time=default.strftime('%H:%M')
    )


@app.route('/day')
def day_page():
    return render_template('day.html')


@app.route('/week')
def week_page():
    return render_template('week.html')


@app.route('/insights')
def insights_page():
    return render_template('insights.html')


# --------------------------------------------------
# 24-HOUR FORECAST API
# --------------------------------------------------

@app.get('/api/forecast')
def api_forecast():

    date = request.args.get('date')
    time = request.args.get('time', '00:00')

    ts = pd.to_datetime(
        f'{date} {time}',
        errors='raise'
    ).tz_localize(None)

    snapshot = input_snapshot(ts)

    forecasts, mode = forecast_from(
        ts,
        24
    )

    next_hour = forecasts[0]

    values = [
        row['prediction']
        for row in forecasts
    ]

    peak_index = int(
        np.argmax(values)
    )

    peak = forecasts[peak_index]

    change = (
        (
            snapshot['demand']
            - snapshot['demand24']
        )
        / snapshot['demand24']
        * 100
        if snapshot['demand24']
        else 0
    )

    demand_75 = float(
        DF['demand_mwh'].quantile(0.75)
    )

    demand_50 = float(
        DF['demand_mwh'].quantile(0.50)
    )

    if next_hour['prediction'] >= demand_75:
        status = 'HIGH'
    elif next_hour['prediction'] >= demand_50:
        status = 'MODERATE'
    else:
        status = 'NORMAL'

    return jsonify({
        'input': snapshot,
        'next_hour': next_hour,
        'forecast_24h': forecasts,
        'peak': peak,
        'demand_change_pct': change,
        'status': status,
        'data_start': DATA_START.strftime('%Y-%m-%d'),
        'data_end': DATA_END.strftime('%Y-%m-%d')
    })


# --------------------------------------------------
# 7-DAY FORECAST API
# --------------------------------------------------

@app.get('/api/week')
def api_week():

    date = request.args.get('date')
    time = request.args.get('time', '00:00')

    start = pd.to_datetime(
        f'{date} {time}',
        errors='raise'
    ).tz_localize(None)

    # One continuous 168-hour recursive forecast.
    rows, _ = forecast_from(
        start,
        168
    )

    days = []

    for day in range(7):

        day_rows = rows[
            day * 24:(day + 1) * 24
        ]

        values = np.array([
            row['prediction']
            for row in day_rows
        ])

        peak_index = int(
            np.argmax(values)
        )

        day_start = (
            start + pd.Timedelta(days=day)
        )

        days.append({
            'date': day_start.strftime(
                '%Y-%m-%d'
            ),
            'label': day_start.strftime(
                '%a %d %b'
            ),
            'average': float(values.mean()),
            'peak': float(values.max()),
            'peak_time': day_rows[
                peak_index
            ]['datetime'].strftime('%H:%M'),
            'hours': day_rows
        })

    return jsonify({
        'days': days
    })


# --------------------------------------------------
# INSIGHTS API
# --------------------------------------------------

@app.get('/api/insights')
def api_insights():

    date = request.args.get('date')
    time = request.args.get('time', '00:00')

    ts = pd.to_datetime(
        f'{date} {time}',
        errors='raise'
    ).tz_localize(None)

    forecasts, _ = forecast_from(
        ts,
        24
    )

    values = np.array([
        row['prediction']
        for row in forecasts
    ])

    snapshot = input_snapshot(ts)

    change = (
        (
            snapshot['demand']
            - snapshot['demand24']
        )
        / snapshot['demand24']
        * 100
        if snapshot['demand24']
        else 0
    )

    # Historical average demand for each hour.
    profile = (
        DF.groupby(
            DF['datetime_local'].dt.hour
        )['demand_mwh']
        .mean()
        .reindex(range(24))
        .fillna(DF['demand_mwh'].mean())
    )

    return jsonify({
        'input': snapshot,
        'change_pct': float(change),
        'pred_peak': float(values.max()),
        'pred_peak_time': forecasts[
            int(np.argmax(values))
        ]['datetime'].strftime('%I:%M %p'),
        'pred_min': float(values.min()),
        'hourly_profile': [
            [int(hour), float(value)]
            for hour, value in profile.items()
        ],
        'metrics': metrics_for_demo()
    })


# --------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False
    )
