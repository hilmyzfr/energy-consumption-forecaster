"""
Streamlit Chat UI for the Energy Forecaster Agent
==================================================
A web-based chat interface for the LangChain energy forecasting agent.

Run with:
  streamlit run streamlit_app.py

Make sure:
  1. Your Energy Forecaster API is running (uvicorn src.api:app --reload)
  2. Your OPENAI_API_KEY is set in the environment
"""

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from agent import create_energy_agent, SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Energy Forecaster Chat",
    page_icon="⚡",
    layout="centered",
)

st.title("⚡ Energy Consumption Forecaster")
st.caption("Ask me about electricity consumption forecasts for Germany")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = create_energy_agent()

if "langchain_messages" not in st.session_state:
    st.session_state.langchain_messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ]

# ---------------------------------------------------------------------------
# Example prompts
# ---------------------------------------------------------------------------

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    cols = st.columns(2)
    examples = [
        "What's the forecast for tomorrow?",
        "Compare all models for next Monday",
        "Predict consumption if yesterday was 1400 GWh",
        "Is the API healthy?",
    ]
    for i, example in enumerate(examples):
        if cols[i % 2].button(example, key=f"example_{i}"):
            st.session_state.messages.append({"role": "user", "content": example})
            st.session_state.langchain_messages.append(
                HumanMessage(content=example)
            )
            st.rerun()

# ---------------------------------------------------------------------------
# Display chat history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------------------------
# Handle new user input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask about energy forecasts..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add to LangChain history
    st.session_state.langchain_messages.append(HumanMessage(content=prompt))

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent.invoke(
                    {"messages": st.session_state.langchain_messages}
                )
                assistant_msg = response["messages"][-1]
                reply = assistant_msg.content

                st.markdown(reply)

                # Update histories
                st.session_state.messages.append(
                    {"role": "assistant", "content": reply}
                )
                st.session_state.langchain_messages.append(
                    AIMessage(content=reply)
                )

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

# ---------------------------------------------------------------------------
# Sidebar info
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### How it works")
    st.markdown(
        """
        This chat interface uses **LangChain** to wrap your
        Energy Forecaster API. The agent:

        1. Parses your natural language query
        2. Extracts date, consumption values, and model choice
        3. Calls the `/predict` endpoint
        4. Explains the results in plain language

        **Available tools:**
        - `get_energy_forecast` — single model prediction
        - `compare_models` — run all 3 models
        - `check_api_health` — verify API status
        """
    )

    st.markdown("---")
    st.markdown("### API Status")
    import requests
    import os

    api_url = os.getenv("ENERGY_API_URL", "http://localhost:8000")
    try:
        r = requests.get(f"{api_url}/health", timeout=3)
        if r.status_code == 200:
            st.success(f"API is running at {api_url}")
        else:
            st.warning(f"API returned status {r.status_code}")
    except requests.ConnectionError:
        st.error(f"API not reachable at {api_url}")

    st.markdown("---")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.langchain_messages = [
            SystemMessage(content=SYSTEM_PROMPT)
        ]
        st.rerun()
