"""
LangChain Conversational Agent for Energy Consumption Forecaster
================================================================
Wraps the existing FastAPI /predict endpoint as a LangChain tool,
allowing natural language queries like:
  - "What's the predicted consumption for next Friday?"
  - "Run all models for March 15 with yesterday's consumption at 1350 GWh"
  - "Is 1500 GWh plausible for a Monday in January?"

Prerequisites:
  pip install langchain langchain-anthropic python-dateutil requests

Usage:
  1. Start your energy forecaster API:  uvicorn src.api:app --reload
  2. Set your Anthropic API key:         export ANTHROPIC_API_KEY=sk-ant-...
  3. Run this agent:                     python agent.py
"""

import json
import os
import sys
from datetime import date, timedelta
from typing import Optional

import requests
from dateutil import parser as dateparser

# Add parent src/ directory to path so we can import smard_client
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

from langgraph.prebuilt import create_react_agent

# Try to import the SMARD client for live consumption data
try:
    from smard_client import get_latest_consumption
    SMARD_AVAILABLE = True
except ImportError:
    SMARD_AVAILABLE = False
    print("Note: smard_client not found in src/. Using default lag values.")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("ENERGY_API_URL", "http://localhost:8000")
LLM_MODEL = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")


# ---------------------------------------------------------------------------
# Helper: resolve natural language dates
# ---------------------------------------------------------------------------

def resolve_date(date_str: str) -> str:
    """Convert natural language date references to YYYY-MM-DD format."""
    today = date.today()
    lower = date_str.lower().strip()

    # Handle relative references
    if lower in ("today", "now"):
        return today.isoformat()
    if lower in ("tomorrow",):
        return (today + timedelta(days=1)).isoformat()
    if lower in ("yesterday",):
        return (today - timedelta(days=1)).isoformat()

    # Handle "next Monday", "this Friday", etc.
    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    for day_name, day_num in weekdays.items():
        if day_name in lower:
            days_ahead = (day_num - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7  # default to next week if "next Monday" and today is Monday
            return (today + timedelta(days=days_ahead)).isoformat()

    # Fallback: let dateutil try to parse it
    try:
        parsed = dateparser.parse(date_str, fuzzy=True)
        return parsed.date().isoformat()
    except (ValueError, TypeError):
        return today.isoformat()


# ---------------------------------------------------------------------------
# LangChain Tools
# ---------------------------------------------------------------------------

@tool
def get_energy_forecast(
    target_date: str,
    lag_1: Optional[float] = None,
    lag_7: Optional[float] = None,
    model: str = "knn",
    special_event: bool = False,
) -> str:
    """Get an energy consumption forecast for Germany on a given date.

    Args:
        target_date: The date to forecast, e.g. "2024-03-08", "next Friday",
                     "tomorrow", or any natural language date.
        lag_1: Yesterday's actual consumption in GWh. If not provided,
               fetches live data from SMARD (Germany's official grid data).
        lag_7: Consumption from the same day last week in GWh. If not provided,
               fetches live data from SMARD.
        model: Which model to use. Options: "knn", "mlp", "baseline", "all".
               Default is "knn".
        special_event: Set to True if there's a known special event (e.g. plant
                       shutdown, holiday closure) that would affect consumption.

    Returns:
        A JSON string with the prediction, temperature, holiday status, and
        plausibility assessment.
    """
    resolved_date = resolve_date(target_date)

    # If lag values not provided, try fetching live data from SMARD
    data_source = "default"
    if lag_1 is None or lag_7 is None:
        if SMARD_AVAILABLE:
            try:
                latest = get_latest_consumption()
                if latest:
                    lag_1 = lag_1 or latest["lag_1"]
                    lag_7 = lag_7 or latest["lag_7"]
                    data_source = "smard_live"
            except Exception:
                pass

    # Final fallback to reasonable defaults
    if lag_1 is None:
        lag_1 = 1350.0
    if lag_7 is None:
        lag_7 = lag_1

    payload = {
        "date": resolved_date,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "special_event": special_event,
        "model": model,
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()
        result["data_source"] = data_source
        result["input_lag_1"] = lag_1
        result["input_lag_7"] = lag_7
        return json.dumps(result, indent=2)
    except requests.ConnectionError:
        return json.dumps({
            "error": "Could not connect to the Energy Forecaster API. "
                     f"Is it running at {API_BASE_URL}? "
                     "Start it with: uvicorn src.api:app --reload"
        })
    except requests.HTTPError as e:
        return json.dumps({"error": f"API returned an error: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Unexpected error: {str(e)}"})


@tool
def check_api_health() -> str:
    """Check if the Energy Forecaster API is running and healthy.

    Returns:
        A status message indicating whether the API is reachable.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return json.dumps({
            "status": "healthy",
            "message": "Energy Forecaster API is running.",
            "details": response.json(),
        })
    except requests.ConnectionError:
        return json.dumps({
            "status": "unreachable",
            "message": f"Cannot reach the API at {API_BASE_URL}. "
                       "Start it with: uvicorn src.api:app --reload",
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e),
        })


@tool
def compare_models(
    target_date: str,
    lag_1: float = 1350.0,
    lag_7: Optional[float] = None,
    special_event: bool = False,
) -> str:
    """Compare predictions from all three models (KNN, MLP, Baseline) for a date.

    This is useful when the user wants to see how different models compare or
    wants the most robust estimate.

    Args:
        target_date: The date to forecast.
        lag_1: Yesterday's consumption in GWh.
        lag_7: Same day last week consumption in GWh.
        special_event: Whether a special event is occurring.

    Returns:
        Predictions from all three models with comparison.
    """
    resolved_date = resolve_date(target_date)

    payload = {
        "date": resolved_date,
        "lag_1": lag_1,
        "lag_7": lag_7 if lag_7 is not None else lag_1,
        "special_event": special_event,
        "model": "all",
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        # Add a helpful comparison summary
        preds = data.get("predictions_gwh", {})
        if preds:
            values = list(preds.values())
            data["comparison"] = {
                "spread_gwh": round(max(values) - min(values), 2),
                "average_gwh": round(sum(values) / len(values), 2),
                "best_model_note": "KNN is the production model (best speed/accuracy tradeoff)",
            }

        return json.dumps(data, indent=2)
    except requests.ConnectionError:
        return json.dumps({
            "error": f"Could not connect to the API at {API_BASE_URL}."
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Agent Setup
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an energy consumption forecasting assistant for Germany.
You help users get electricity consumption predictions by talking to them naturally
and using the Energy Forecaster API behind the scenes.

Key context:
- Predictions are in GWh (gigawatt-hours) of daily electricity consumption for Germany.
- The API needs a date and recent consumption values (lag_1 = yesterday, lag_7 = same day last week).
- When the user doesn't provide lag values, the system automatically fetches LIVE
  consumption data from SMARD (Germany's official Bundesnetzagentur grid data platform).
  This means predictions use real, up-to-date input values.
- Three models are available: KNN (production default), MLP, and a day-of-week Baseline.
- The API automatically fetches weather data and checks German holidays.
- A plausibility check flags suspicious predictions or data pipeline issues.

When presenting results:
- Always mention the predicted consumption clearly.
- If the data_source is "smard_live", mention that real grid data was used.
- Explain the plausibility check result in plain language.
- If temperature or holiday status is interesting, mention it.
- If the user asks to compare models, use the compare_models tool.
- If something seems off, suggest they check their lag values.

Be concise but helpful. You're talking to energy professionals who know the domain."""


def create_energy_agent():
    """Create and return the LangChain energy forecasting agent."""

    # Initialize the LLM
    llm = ChatAnthropic(model=LLM_MODEL, temperature=0)

    # Create the agent with tools
    tools = [get_energy_forecast, check_api_health, compare_models]
    agent = create_react_agent(llm, tools)

    return agent


# ---------------------------------------------------------------------------
# Interactive Chat Loop
# ---------------------------------------------------------------------------

def chat():
    """Run an interactive chat session with the energy forecasting agent."""
    agent = create_energy_agent()

    print("=" * 60)
    print("  Energy Consumption Forecaster - Chat Interface")
    print("=" * 60)
    print()
    print("Ask me about energy consumption forecasts for Germany!")
    print("Examples:")
    print('  - "What\'s the forecast for tomorrow?"')
    print('  - "Predict consumption for next Monday, yesterday was 1380 GWh"')
    print('  - "Compare all models for March 20th"')
    print('  - "Is the API running?"')
    print()
    print("Type 'quit' or 'exit' to stop.")
    print("-" * 60)

    # Maintain conversation history
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        messages.append(HumanMessage(content=user_input))

        # Stream the agent response
        print("\nAssistant: ", end="", flush=True)

        response = agent.invoke({"messages": messages})

        # Get the last assistant message
        assistant_msg = response["messages"][-1]
        print(assistant_msg.content)

        # Add to history for multi-turn conversation
        messages.append(assistant_msg)


if __name__ == "__main__":
    chat()
