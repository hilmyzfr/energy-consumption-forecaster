import os
import httpx
from datetime import date
from src.agent.parser import parse_email
from src.agent.store import init_db, save_event, get_recent_events, check_conflict

API_BASE_URL = os.getenv("ENERGY_API_URL", "http://localhost:8000")

def run(customer_id: str, raw_email: str, lag_1: float, lag_7: float) -> dict:
    # Step 1: parse the email
    event = parse_email(raw_email)

    # Step 2: check context store for conflicts
    init_db()
    conflict = check_conflict(customer_id, event)
    save_event(customer_id, event)
    recent = get_recent_events(customer_id)

    # Step 3: call forecast API
    forecast = None
    api_error = None
    if event.special_event and event.start_date:
        try:
            response = httpx.post(f"{API_BASE_URL}/predict", json={
                "date": event.start_date,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "special_event": True,
                "model": "knn"
            })
            forecast = response.json()
        except Exception as e:
            api_error = str(e)

    return {
        "customer_id": customer_id,
        "parsed_event": event.model_dump(),
        "conflict_warning": conflict,
        "recent_history": recent,
        "forecast": forecast,
        "api_error": api_error
    }
