from flask import Flask, render_template, request, jsonify
import json, os
import numpy as np
import pandas as pd
import pennylane as qml

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # never cache CSS/JS while developing
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.after_request
def no_cache(resp):
    resp.headers['Cache-Control'] = 'no-store'
    return resp

with open(os.path.join(BASE, 'vqc_model.json')) as f:
    MODEL = json.load(f)

DF = pd.read_parquet(os.path.join(BASE, 'pjm_demand_weather_hourly_v1.parquet'))
DF['datetime_local'] = pd.to_datetime(DF['datetime_local'], errors='coerce', utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None)
DF = DF.sort_values('datetime_local').reset_index(drop=True)
DF['temp_pop_wt_c'] = DF['temp_pop_wt_c'].interpolate().bfill().ffill()
DF['demand_24h_ago'] = DF['demand_mwh'].shift(24)
DF = DF.dropna(subset=['demand_mwh','demand_24h_ago','temp_pop_wt_c']).reset_index(drop=True)

WEIGHTS = np.array(MODEL['weights'], dtype=float)
READOUT_W = np.array(MODEL['readout_weights'], dtype=float)
READOUT_B = float(MODEL['readout_bias'])
S_MIN = np.array(MODEL['feature_scaler_min'], dtype=float)
S_SCALE = np.array(MODEL['feature_scaler_scale'], dtype=float)
T_MIN = np.array(MODEL['target_scaler_min'], dtype=float)
T_SCALE = np.array(MODEL['target_scaler_scale'], dtype=float)

try:
    DEV = qml.device('lightning.qubit', wires=4)
except Exception:
    DEV = qml.device('default.qubit', wires=4)

@qml.qnode(DEV)
def quantum_circuit(inputs, weights):
    for i in range(4):
        qml.RY(inputs[i], wires=i)
    for layer in range(2):
        for i in range(4):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RZ(weights[layer, i, 1], wires=i)
        for i in range(3):
            qml.CNOT(wires=[i, i+1])
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]

def scale_x(x):
    # sklearn MinMax: X_scaled = X * scale_ + min_
    return np.asarray(x, dtype=float) * S_SCALE + S_MIN

def inverse_y(y):
    return (float(y) - T_MIN[0]) / T_SCALE[0]

def predict_features(demand, demand24, temp, hour):
    x_scaled = scale_x([demand, demand24, temp, hour])
    angles = x_scaled * np.pi
    q = np.asarray(quantum_circuit(angles, WEIGHTS), dtype=float)
    out = float(np.dot(q, READOUT_W) + READOUT_B)
    pred_scaled = (np.tanh(out) + 1.0) / 2.0
    return inverse_y(pred_scaled)

DATA_START = DF['datetime_local'].min()
DATA_END = DF['datetime_local'].max()
LAST_TEMP = float(DF['temp_pop_wt_c'].iloc[-1])

# Fast lookup for exact historical rows
ROW_BY_TIME = {ts: r for ts, r in DF.set_index('datetime_local').iterrows()}

def get_temp(ts):
    if ts in ROW_BY_TIME:
        return float(ROW_BY_TIME[ts]['temp_pop_wt_c'])
    # For future demo forecasts, use the latest available temperature.
    return LAST_TEMP

def get_demand(ts):
    if ts in ROW_BY_TIME:
        return float(ROW_BY_TIME[ts]['demand_mwh'])
    return None

def get_start_demand(ts):
    # For a historical timestamp use actual demand. For future timestamps,
    # anchor to the closest earlier available demand at the same hour if possible,
    # otherwise the latest observed demand.
    d = get_demand(ts)
    if d is not None:
        return d, 'dataset'
    candidates = DF[DF['datetime_local'] <= ts]
    if not candidates.empty:
        return float(candidates.iloc[-1]['demand_mwh']), 'demo-fallback'
    return float(DF['demand_mwh'].iloc[0]), 'demo-fallback'

def forecast_from(start_ts, hours=24):
    # Recursive one-hour-ahead forecast. The second demand input is ALWAYS
    # the demand from exactly 24 hours earlier, matching the trained feature.
    current_demand, mode = get_start_demand(start_ts)
    history = {start_ts: float(current_demand)}
    rows=[]
    for step in range(1, hours+1):
        ts = start_ts + pd.Timedelta(hours=step)
        lag_ts = ts - pd.Timedelta(hours=24)

        # For the first 24 forecast steps, use the real historical 24h lag
        # whenever it exists. After that, use the recursively predicted value.
        demand24 = history.get(lag_ts)
        if demand24 is None:
            demand24 = get_demand(lag_ts)
        if demand24 is None:
            demand24 = current_demand

        temp = get_temp(ts)
        hour = ts.hour
        pred = predict_features(current_demand, demand24, temp, hour)
        pred = float(pred)
        history[ts] = pred
        rows.append({'datetime': ts, 'prediction': pred, 'temperature': float(temp), 'hour': hour})
        current_demand = pred
    return rows, mode

def input_snapshot(ts):
    demand, mode = get_start_demand(ts)
    d24 = get_demand(ts - pd.Timedelta(hours=24))
    if d24 is None: d24 = demand
    temp = get_temp(ts)
    return {
        'datetime': ts.strftime('%Y-%m-%d %H:%M'),
        'display_datetime': ts.strftime('%d %b %Y, %I:%M %p'),
        'demand': round(demand,2),
        'demand24': round(d24,2),
        'temperature': round(temp,2),
        'hour': ts.hour,
        'mode': mode,
    }

def metrics_for_demo():
    # Use the exported VQC predictions when available for honest performance display.
    p = os.path.join(BASE, '..', 'vqc_predictions.csv')
    if os.path.exists(p):
        x = pd.read_csv(p)
        actual=x['Actual_Demand_MWh'].to_numpy(); pred=x['VQC_Predicted_MWh'].to_numpy()
        mae=float(np.mean(np.abs(actual-pred)))
        rmse=float(np.sqrt(np.mean((actual-pred)**2)))
        ssr=float(np.sum((actual-pred)**2)); sst=float(np.sum((actual-actual.mean())**2))
        r2=float(1-ssr/sst) if sst else 0
        return {'mae':mae,'rmse':rmse,'r2':r2}
    return {'mae':None,'rmse':None,'r2':None}

@app.route('/')
def home():
    default = DATA_END.floor('h')
    return render_template('home.html', default_date=default.strftime('%Y-%m-%d'), default_time=default.strftime('%H:%M'))

@app.route('/day')
def day_page():
    return render_template('day.html')

@app.route('/week')
def week_page():
    return render_template('week.html')

@app.route('/insights')
def insights_page():
    return render_template('insights.html')

@app.get('/api/forecast')
def api_forecast():
    date=request.args.get('date'); time=request.args.get('time','00:00')
    ts=pd.to_datetime(f'{date} {time}', errors='raise').tz_localize(None)
    snap=input_snapshot(ts)
    forecasts, mode=forecast_from(ts, 24)
    next_hour=forecasts[0]
    vals=[x['prediction'] for x in forecasts]
    peak_i=int(np.argmax(vals)); peak=forecasts[peak_i]
    change=((snap['demand']-snap['demand24'])/snap['demand24']*100) if snap['demand24'] else 0
    status='HIGH' if next_hour['prediction'] >= float(DF['demand_mwh'].quantile(.75)) else ('MODERATE' if next_hour['prediction'] >= float(DF['demand_mwh'].quantile(.5)) else 'NORMAL')
    return jsonify({'input':snap,'next_hour':next_hour,'forecast_24h':forecasts,'peak':peak,'demand_change_pct':change,'status':status,'data_start':DATA_START.strftime('%Y-%m-%d'),'data_end':DATA_END.strftime('%Y-%m-%d')})

@app.get('/api/week')
def api_week():
    date=request.args.get('date'); time=request.args.get('time','00:00')
    start=pd.to_datetime(f'{date} {time}', errors='raise').tz_localize(None)
    # Generate one continuous 168-hour recursive forecast so each day
    # starts from the previous day's predicted state instead of restarting
    # from the same value. This prevents identical 7-day cards.
    rows,_=forecast_from(start,168)
    days=[]
    for day in range(7):
        day_rows=rows[day*24:(day+1)*24]
        arr=np.array([r['prediction'] for r in day_rows])
        days.append({'date':(start + pd.Timedelta(days=day)).strftime('%Y-%m-%d'),
                     'label':(start + pd.Timedelta(days=day)).strftime('%a %d %b'),
                     'average':float(arr.mean()),
                     'peak':float(arr.max()),
                     'peak_time':day_rows[int(arr.argmax())]['datetime'].strftime('%H:%M'),
                     'hours':day_rows})
    return jsonify({'days':days})

@app.get('/api/insights')
def api_insights():
    date=request.args.get('date'); time=request.args.get('time','00:00')
    ts=pd.to_datetime(f'{date} {time}', errors='raise').tz_localize(None)
    forecasts,_=forecast_from(ts,24)
    arr=np.array([x['prediction'] for x in forecasts])
    snap=input_snapshot(ts)
    change=((snap['demand']-snap['demand24'])/snap['demand24']*100) if snap['demand24'] else 0
    # Historical hourly profile from dataset
    profile=DF.groupby(DF['datetime_local'].dt.hour)['demand_mwh'].mean().reindex(range(24)).fillna(DF['demand_mwh'].mean())
    return jsonify({'input':snap,'change_pct':float(change),'pred_peak':float(arr.max()),'pred_peak_time':forecasts[int(arr.argmax())]['datetime'].strftime('%I:%M %p'),'pred_min':float(arr.min()),'hourly_profile':[[int(h),float(v)] for h,v in profile.items()],'metrics':metrics_for_demo()})

if __name__=='__main__':
    app.run(host='0.0.0.0',port=5000,debug=False)
