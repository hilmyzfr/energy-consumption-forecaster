"""
SMARD API Integration for Energy Consumption Forecaster
========================================================
Fetches real electricity consumption data from Germany's official
Bundesnetzagentur SMARD platform. Free, no API key needed.

API docs: https://github.com/bundesAPI/smard-api

Usage:
    from smard_client import fetch_consumption, get_latest_consumption

    # Get daily consumption for a date range
    df = fetch_consumption('2025-12-01', '2025-12-31')

    # Get the most recent available consumption values (for lag features)
    latest = get_latest_consumption()
    print(latest['lag_1'], latest['lag_7'])
"""

import requests
import pandas as pd
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# SMARD API Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.smard.de/app/chart_data"
FILTER_CONSUMPTION = "410"  # Stromverbrauch: Gesamt (Netzlast)
REGION = "DE"


# ---------------------------------------------------------------------------
# Core API Functions
# ---------------------------------------------------------------------------

def _get_available_timestamps(resolution="day"):
    """Get all available timestamps from the SMARD API."""
    url = f"{BASE_URL}/{FILTER_CONSUMPTION}/{REGION}/index_{resolution}.json"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("timestamps", [])
    except Exception as e:
        print(f"Error fetching SMARD timestamps: {e}")
        return []


def _get_timeseries(timestamp, resolution="day"):
    """Fetch timeseries data starting from a given timestamp."""
    url = (
        f"{BASE_URL}/{FILTER_CONSUMPTION}/{REGION}/"
        f"{FILTER_CONSUMPTION}_{REGION}_{resolution}_{timestamp}.json"
    )
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("series", [])
    except Exception as e:
        print(f"Error fetching SMARD timeseries: {e}")
        return []


def _find_timestamps_for_range(start_date, end_date, timestamps):
    """Find which SMARD timestamp chunks cover the requested date range."""
    start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    # Add one day buffer to end to make sure we capture the full range
    end_ms += 86400000

    relevant = []
    sorted_ts = sorted(timestamps)

    for i, ts in enumerate(sorted_ts):
        # Each timestamp chunk covers data until the next chunk starts
        next_ts = sorted_ts[i + 1] if i + 1 < len(sorted_ts) else ts + 365 * 86400000
        # Include this chunk if it overlaps with our range
        if ts <= end_ms and next_ts >= start_ms:
            relevant.append(ts)

    return relevant


# ---------------------------------------------------------------------------
# Public Functions
# ---------------------------------------------------------------------------

def fetch_consumption(start_date, end_date, resolution="day"):
    """
    Fetch daily electricity consumption (grid load) for Germany.

    Args:
        start_date: Start date as 'YYYY-MM-DD'
        end_date: End date as 'YYYY-MM-DD'
        resolution: 'day', 'hour', or 'quarterhour'

    Returns:
        DataFrame with columns: date, Consumption (in GWh)
    """
    print(f"Fetching SMARD consumption data: {start_date} to {end_date}...")

    timestamps = _get_available_timestamps(resolution)
    if not timestamps:
        print("Warning: Could not get SMARD timestamps")
        return pd.DataFrame(columns=["date", "Consumption"])

    relevant_ts = _find_timestamps_for_range(start_date, end_date, timestamps)
    if not relevant_ts:
        # Fallback: use the last few timestamps
        relevant_ts = sorted(timestamps)[-3:]

    all_data = []
    for ts in relevant_ts:
        series = _get_timeseries(ts, resolution)
        for point in series:
            if point[0] is not None and point[1] is not None:
                dt = datetime.utcfromtimestamp(point[0] / 1000)
                # SMARD returns MWh, convert to GWh for consistency with your dataset
                consumption_gwh = point[1] / 1000.0
                all_data.append({"date": dt, "Consumption": consumption_gwh})

    if not all_data:
        print("Warning: No data returned from SMARD")
        return pd.DataFrame(columns=["date", "Consumption"])

    df = pd.DataFrame(all_data)
    df["date"] = pd.to_datetime(df["date"])

    # Filter to requested range
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    df = df[mask].copy()

    # Remove duplicates and sort
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    print(f"  Retrieved {len(df)} data points")
    return df


def get_latest_consumption(lookback_days=14):
    """
    Get the most recent consumption values, useful for computing lag features.

    Args:
        lookback_days: How many days back to fetch (default 14)

    Returns:
        dict with keys: lag_1, lag_7, rolling_7, latest_date, latest_value
        Returns None if data can't be fetched.
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    df = fetch_consumption(start_date, end_date)

    if df.empty or len(df) < 8:
        print(f"Warning: Only got {len(df)} days of data, need at least 8")
        return None

    # Sort by date descending to get most recent
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    latest = df.iloc[0]
    yesterday = df.iloc[1] if len(df) > 1 else latest
    last_week = df.iloc[7] if len(df) > 7 else df.iloc[-1]

    # Rolling 7-day average (excluding today)
    rolling_7 = df.iloc[1:8]["Consumption"].mean() if len(df) > 7 else df["Consumption"].mean()

    return {
        "latest_date": latest["date"].strftime("%Y-%m-%d"),
        "latest_value": round(latest["Consumption"], 2),
        "lag_1": round(yesterday["Consumption"], 2),
        "lag_7": round(last_week["Consumption"], 2),
        "rolling_7": round(rolling_7, 2),
    }


def fetch_and_save_dataset(start_date, end_date, output_path="data/raw/smard_consumption.csv"):
    """
    Fetch consumption data and save as CSV in the same format as your
    existing opsd_germany_daily.csv dataset.

    Args:
        start_date: Start date 'YYYY-MM-DD' (e.g. '2018-01-01')
        end_date: End date 'YYYY-MM-DD' (e.g. '2025-12-31')
        output_path: Where to save the CSV
    """
    df = fetch_consumption(start_date, end_date)

    if df.empty:
        print("No data to save")
        return None

    # Format to match your existing dataset structure
    df = df.set_index("date")
    df.index.name = "Date"

    df.to_csv(output_path)
    print(f"Saved {len(df)} rows to {output_path}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        start = sys.argv[1]
        end = sys.argv[2]
        print(f"\nFetching consumption: {start} to {end}")
        df = fetch_consumption(start, end)
        if not df.empty:
            print(f"\n{df.to_string(index=False)}")
    else:
        # Default: show latest values and Christmas 2025
        print("=" * 50)
        print("  SMARD Data Fetcher")
        print("=" * 50)

        print("\n--- Latest consumption values ---")
        latest = get_latest_consumption()
        if latest:
            for k, v in latest.items():
                print(f"  {k}: {v}")

        print("\n--- Christmas 2025 ---")
        xmas = fetch_consumption("2025-12-23", "2025-12-27")
        if not xmas.empty:
            for _, row in xmas.iterrows():
                print(f"  {row['date'].strftime('%Y-%m-%d')} ({row['date'].strftime('%A')}): {row['Consumption']:.1f} GWh")

        print("\n--- Usage ---")
        print("  python3 smard_client.py 2025-01-01 2025-12-31")
        print("  python3 smard_client.py 2018-01-01 2025-12-31  # extend your dataset")
