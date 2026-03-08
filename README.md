# Energy Consumption Forecaster

Forecasting daily electricity consumption for Germany using Open Power System Data.
Built to practice time series forecasting methods I used during my time at enercity.

## What it does

Takes a date and recent consumption values, fetches live temperature from Open-Meteo,
checks German public holidays, and returns a predicted electricity consumption in GWh.

## Dataset

Open Power System Data — daily electricity consumption for Germany, 2012-2017.
Source: https://open-power-system-data.org

## Models

Three models compared on 2017 holdout data (365 days):

1. **Day-of-week baseline** — weighted average of last 5 same weekday + same month 
   observations. Recent weeks weighted higher.

2. **KNN** — k-nearest neighbours on time, lag and weather features.

3. **MLP** — simple two-layer neural network. Same features as KNN.

## Results

| Model | MAE | RMSE | Train time | Inference |
|-------|-----|------|------------|-----------|
| Day-of-week baseline | 48 GWh | 78 GWh | — | 0.11s |
| KNN | 25 GWh | 35 GWh | 0.02s | 0.006s |
| MLP | 20 GWh | 32 GWh | 1.85s | ~0s |

KNN is the best tradeoff for production — fast to train, near instant inference,
only slightly behind MLP on accuracy.

## Features

- Day of week, month, is_weekend, is_holiday (German public holidays)
- Temperature (fetched live from Open-Meteo historical/forecast API)
- lag_1 — yesterday's consumption
- lag_7 — same day last week
- rolling_7 — 7-day rolling average

## How to run

**Train models:**
```bash