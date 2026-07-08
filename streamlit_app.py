"""
Streamlit UI for the travel planning assistant.

Wires together:
  - Chat interface (talks to the orchestrator, which talks to the LLM)
  - Map exploration widget (pydeck) for the stay_location flow, kept
    entirely client-side per the assistant/frontend split -- pin
    interaction never round-trips through the LLM, only the final
    selection + explicit lock-in confirmation do.
  - Recommendation display once the Recommendation Engine has run.

Run with: streamlit run streamlit_app.py

Uses a MockLLMClient by default so the app is testable/demoable without an
API key. Set ANTHROPIC_API_KEY and flip USE_REAL_LLM to True to go live.
"""

import os

import pandas as pd
import pydeck as pdk
import streamlit as st

from llm_client import MockLLMClient, ResilientLLMClient, SafeFallbackClient
from orchestrator import (
    ConversationState,
    apply_map_selection,
    lock_map_selection,
    process_turn,
)

# A small deterministic demo script so the whole app is click-through-able
# without a live model, and used as the visible chat script in mock mode.
DEMO_SCRIPT = [
    {
        "reply": "A Japan trip sounds wonderful! How many of you are traveling, and for how long?",
        "profile_updates": {
            "trip": {"destination": {"confirmed": ["Japan"]}},
            "trip_objective": {"intent": "Vacation", "confidence": "medium", "inferred": True},
        },
        "trigger_recommendation": False,
        "show_map": None,
    },
    {
        "reply": "Great -- 2 travelers, 5 days. Take a look at the map and pick roughly where you'd like to stay.",
        "profile_updates": {
            "trip": {"duration_days": 5},
            "traveller_composition": {"adults": 2, "relationship": "couple"},
        },
        "trigger_recommendation": False,
        "show_map": {"destination": "Japan"},
    },
    {
        "reply": "Perfect, locked in. Building your recommendations now.",
        "profile_updates": {
            "interests": {"food": 9, "nature": 6},
            "budget": {"overall": 250000, "currency": "INR"},
        },
        "trigger_recommendation": True,
        "show_map": None,
    },
]


def get_api_key() -> str | None:
    """Check Streamlit secrets first (how the deployed app is configured),
    then fall back to environment variable (local/dev runs)."""
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass  # no secrets.toml present locally -- not an error
    return os.environ.get("ANTHROPIC_API_KEY")


def get_llm_client():
    api_key = get_api_key()
    if not api_key:
        return MockLLMClient(DEMO_SCRIPT)

    from llm_client import AnthropicLLMClient

    primary = AnthropicLLMClient(api_key=api_key)
    fallback = SafeFallbackClient()

    def on_fallback(exc):
        st.session_state.setdefault("fallback_warnings", []).append(str(exc))

    return ResilientLLMClient(primary, fallback, on_fallback=on_fallback)


def init_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = ConversationState()
    if "llm_client" not in st.session_state:
        st.session_state.llm_client = get_llm_client()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # [(role, text)] for rendering
    if "map_destination" not in st.session_state:
        st.session_state.map_destination = None
    if "map_pin" not in st.session_state:
        st.session_state.map_pin = None  # {"area": str, "lat": float, "lng": float}


def render_map(destination: str):
    """
    Client-side map exploration widget. Panning/pin placement here never
    calls the LLM -- only the explicit 'Lock this in' button below reports
    a decision back through the orchestrator.
    """
    st.subheader(f"Explore {destination}")

    # Seed points -- in production, pull real POI coordinates for the
    # destination from a places API. Kept static here for the demo.
    seed_pois = pd.DataFrame([
        {"name": "City Center", "lat": 35.6762, "lon": 139.6503},
        {"name": "Shibuya", "lat": 35.6595, "lon": 139.7005},
        {"name": "Asakusa", "lat": 35.7148, "lon": 139.7967},
    ])

    selected_area = st.selectbox(
        "Pick roughly where you'd like to stay:",
        options=seed_pois["name"].tolist(),
    )
    row = seed_pois[seed_pois["name"] == selected_area].iloc[0]

    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=row["lat"], longitude=row["lon"], zoom=12, pitch=0,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=seed_pois,
                get_position="[lon, lat]",
                get_radius=300,
                get_fill_color=[230, 100, 60],
                pickable=True,
            ),
        ],
        tooltip={"text": "{name}"},
    ))

    st.session_state.map_pin = {
        "area": selected_area,
        "lat": float(row["lat"]),
        "lng": float(row["lon"]),
        "nearby": seed_pois["name"].tolist(),
    }

    if st.button("Lock this in as my base for the trip"):
        pin = st.session_state.map_pin
        apply_map_selection(
            st.session_state.conversation,
            pin["area"], pin["lat"], pin["lng"], pin["nearby"],
        )
        lock_map_selection(st.session_state.conversation)
        st.session_state.map_destination = None
        st.success(f"Locked in {pin['area']} as your base.")
        st.rerun()


def render_recommendation(rec: dict):
    st.subheader("Your trip plan")

    with st.expander("Itinerary", expanded=True):
        for day in rec["itinerary"]:
            st.markdown(f"**Day {day['day']}**")
            if day["notes"]:
                st.caption(" / ".join(day["notes"]))

    col1, col2 = st.columns(2)
    with col1:
        with st.expander("Hotel options"):
            for h in rec["hotel_ranking"]:
                st.write(h["rationale"])
        with st.expander("Restaurants"):
            for r in rec["restaurants"]:
                st.write(r["rationale"])
    with col2:
        with st.expander("Flights"):
            for f in rec["flight_ranking"]:
                st.write(f["rationale"])
        with st.expander("Activities"):
            for a in rec["activities"]:
                st.write(a["rationale"])

    with st.expander("Packing list"):
        if rec["packing_list"]:
            for item in rec["packing_list"]:
                st.write(f"- **{item['item']}** ({item['category']}) — {item['reason']}")
        else:
            st.caption("Nothing flagged yet.")

    with st.expander("Budget summary"):
        st.json(rec["budget_summary"])

    with st.expander("Risk analysis"):
        if rec["risk_analysis"]["flags"]:
            for flag in rec["risk_analysis"]["flags"]:
                st.warning(f"[{flag['severity']}] {flag['detail']}")
        else:
            st.caption(rec["risk_analysis"]["summary"])

    st.caption(
        "⚠️ Live data sources not yet connected for this demo build — "
        "flight, hotel, and restaurant results above are placeholders "
        "pending real API integration (see recommendation_engine.py TODOs)."
    )


def main():
    st.set_page_config(page_title="Travel Planning Assistant", page_icon="🧭", layout="wide")
    init_state()

    st.title("🧭 Travel Planning Assistant")
    st.caption("Chat naturally — destination, dates, and preferences are picked up automatically.")

    mode_label = "🟢 LIVE (Claude)" if get_api_key() else "🟡 MOCK (scripted demo)"
    st.caption(f"Mode: {mode_label}")

    if st.session_state.get("fallback_warnings"):
        with st.expander(f"⚠️ {len(st.session_state['fallback_warnings'])} live call(s) fell back to safe mode", expanded=False):
            for w in st.session_state["fallback_warnings"]:
                st.caption(w)

    chat_col, side_col = st.columns([2, 1])

    with chat_col:
        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(text)

        user_input = st.chat_input("Tell me about your trip...")
        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            result = process_turn(
                st.session_state.conversation, user_input, st.session_state.llm_client,
            )
            st.session_state.chat_history.append(("assistant", result["reply"]))
            if result["show_map"]:
                st.session_state.map_destination = result["show_map"]["destination"]
            st.rerun()

    with side_col:
        if st.session_state.map_destination:
            render_map(st.session_state.map_destination)

        rec = st.session_state.conversation.recommendation
        if rec:
            render_recommendation(rec)

        with st.expander("Debug: traveller_profile"):
            st.json(st.session_state.conversation.profile)


if __name__ == "__main__":
    main()
