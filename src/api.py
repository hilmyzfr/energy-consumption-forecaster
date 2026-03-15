import sys
import os
import holidays
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append('src')
from preprocess import fetch_temperature

# load saved models from train.py
MODELS_DIR = 'models'

if not os.path.exists(os.path.join(MODELS_DIR, 'knn.joblib')):
    print("ERROR: No saved models found. Run 'python3 src/train.py' first.")
    sys.exit(1)

print("Loading saved models...")
knn = joblib.load(os.path.join(MODELS_DIR, 'knn.joblib'))
mlp = joblib.load(os.path.join(MODELS_DIR, 'mlp.joblib'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
full_data = pd.read_parquet(os.path.join(MODELS_DIR, 'full_data.parquet'))
de_holidays = holidays.Germany()
print("Models loaded.")


def dow_average_baseline(date, full_data, n_weeks=5, decay=0.85):
    dow = date.dayofweek
    month = date.month
    same_dow_month = full_data[
        (full_data.index.dayofweek == dow) &
        (full_data.index.month == month) &
        (full_data.index < date)
    ]
    last_n = same_dow_month.iloc[-n_weeks:]
    weights = np.array([decay ** i for i in range(len(last_n)-1, -1, -1)])
    weights = weights / weights.sum()
    return float(np.average(last_n['Consumption'].values, weights=weights))


def check_plausibility(date, prediction, full_data, special_event,
                       request_lag_1, threshold=0.20, n_weeks=8):
    dow = date.dayofweek
    is_weekend = dow >= 5
    is_holiday = date in de_holidays

    historical_mean = full_data[
        (full_data.index.dayofweek == dow) &
        (full_data.index.month == date.month)
    ]['Consumption'].mean()

    lag_deviation = abs(request_lag_1 - historical_mean) / historical_mean * 100
    if lag_deviation > 50:
        return {
            "is_plausible": bool(False),
            "warning": f"data issue - lag_1 deviates {round(lag_deviation, 1)}% from historical mean. check input data pipeline",
            "expected_range": [round(historical_mean * 0.5, 2), round(historical_mean * 1.5, 2)],
            "deviation_pct": round(lag_deviation, 1),
            "special_event_mode": bool(special_event),
            "data_issue": bool(True)
        }

    same_dow = full_data[
        (full_data.index.dayofweek == dow) &
        (full_data.index.month == date.month) &
        (full_data.index < date)
    ]
    last_n = same_dow['Consumption'].iloc[-n_weeks:]

    if len(last_n) == 0:
        return {
            "is_plausible": bool(True),
            "warning": "not enough historical data to check plausibility",
            "expected_range": None,
            "deviation_pct": None,
            "special_event_mode": bool(special_event),
            "data_issue": bool(False)
        }

    mean_val = last_n.mean()
    lower = round(mean_val * (1 - threshold), 2)
    upper = round(mean_val * (1 + threshold), 2)
    deviation_pct = round(abs(prediction - mean_val) / mean_val * 100, 1)
    is_plausible = bool(lower <= prediction <= upper)

    if special_event:
        warning = "special event flagged - prediction plausibility check suppressed"
        is_plausible = bool(True)
    elif not is_plausible:
        day_type = "holiday" if is_holiday else "weekend" if is_weekend else "weekday"
        warning = (f"prediction deviates {deviation_pct}% from expected range "
                   f"for {day_type} in this month")
    else:
        warning = None

    return {
        "is_plausible": is_plausible,
        "warning": warning,
        "expected_range": [lower, upper],
        "deviation_pct": deviation_pct,
        "special_event_mode": bool(special_event),
        "data_issue": bool(False)
    }


app = FastAPI()


class PredictionRequest(BaseModel):
    date: str
    lag_1: float
    lag_7: float
    special_event: bool = False
    model: str = "knn"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    date = pd.Timestamp(request.date)

    temp_df = fetch_temperature(request.date, request.date)
    temp = float(temp_df['temperature'].values[0])

    is_holiday = int(date in de_holidays)
    is_weekend = int(date.dayofweek >= 5)
    rolling_7 = (request.lag_1 + request.lag_7) / 2

    # interaction features
    holiday_temp = is_holiday * temp
    weekend_temp = is_weekend * temp

    X = np.array([[
        date.dayofweek,
        date.month,
        is_weekend,
        is_holiday,
        temp,
        request.lag_1,
        request.lag_7,
        rolling_7,
        holiday_temp,
        weekend_temp
    ]])
    X_scaled = scaler.transform(X)

    predictions = {}
    if request.model in ("knn", "all"):
        predictions["knn"] = round(float(knn.predict(X_scaled)[0]), 2)
    if request.model in ("mlp", "all"):
        predictions["mlp"] = round(float(mlp.predict(X_scaled)[0]), 2)
    if request.model in ("baseline", "all"):
        predictions["baseline"] = round(dow_average_baseline(date, full_data), 2)

    reference_pred = predictions.get(request.model) or predictions.get("knn")

    plausibility = check_plausibility(
        date, reference_pred, full_data, request.special_event, request.lag_1
    )

    return {
        "date": request.date,
        "model": request.model,
        "predictions_gwh": predictions,
        "temperature_c": round(temp, 1),
        "is_holiday": is_holiday,
        "plausibility": plausibility
    }