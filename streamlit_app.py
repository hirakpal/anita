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

# Static opening greeting -- shown immediately on load, no LLM call needed.
# The model's first real turn only happens once the user replies, so this
# costs nothing and can't fail even if the LLM/network is having trouble.
OPENING_GREETING = (
    "👋 Hi there! I'm ANITA, your travel planning assistant. "
    "What's your name, and where are you dreaming of traveling?"
)

# A small deterministic demo script so the whole app is click-through-able
# without a live model, and used as the visible chat script in mock mode.
DEMO_SCRIPT = [
    {
        "reply": "Great to meet you! A Japan trip sounds wonderful -- how many of you are traveling, and for how long?",
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
        st.session_state.chat_history = [("assistant", OPENING_GREETING)]
    if "map_destination" not in st.session_state:
        st.session_state.map_destination = None
    if "map_pin" not in st.session_state:
        st.session_state.map_pin = None  # {"area": str, "lat": float, "lng": float}


def get_places_api_key() -> str | None:
    """Same pattern as get_api_key(): st.secrets first, then env var."""
    try:
        if "GOOGLE_PLACES_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_PLACES_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GOOGLE_PLACES_API_KEY")


def geocode_destination_live(destination: str, api_key: str) -> tuple[float, float] | None:
    """
    Text Search (New): resolve a destination name to coordinates.
    Returns None on any failure (bad key, network error, no results) so
    callers can fall back to the static centroid lookup -- never raises.
    """
    try:
        import requests
        response = requests.post(
            "https://places.googleapis.com/v1/places:searchText",
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": "places.location,places.displayName",
            },
            json={"textQuery": destination, "maxResultCount": 1},
            timeout=5,
        )
        response.raise_for_status()
        places = response.json().get("places", [])
        if not places:
            return None
        loc = places[0]["location"]
        return (loc["latitude"], loc["longitude"])
    except Exception:
        return None


def nearby_places_live(lat: float, lng: float, api_key: str, radius_m: int = 3000) -> list[dict]:
    """
    Nearby Search (New): real POIs around a point. Returns [] on any
    failure -- never raises. Field mask kept minimal (name + location) to
    control cost.
    """
    try:
        import requests
        response = requests.post(
            "https://places.googleapis.com/v1/places:searchNearby",
            headers={
                "Content-Type": "application/json",
                "X-Goog-Api-Key": api_key,
                "X-Goog-FieldMask": "places.displayName,places.location",
            },
            json={
                "maxResultCount": 8,
                "locationRestriction": {
                    "circle": {"center": {"latitude": lat, "longitude": lng}, "radius": radius_m}
                },
            },
            timeout=5,
        )
        response.raise_for_status()
        places = response.json().get("places", [])
        return [
            {
                "name": p.get("displayName", {}).get("text", "Unnamed"),
                "lat": p["location"]["latitude"],
                "lon": p["location"]["longitude"],
            }
            for p in places if "location" in p
        ]
    except Exception:
        return []


# Rough centroids for common demo destinations -- used ONLY when the Places
# API key isn't set or a live call fails. This is the safety-net path, not
# the primary path once GOOGLE_PLACES_API_KEY is configured.
DESTINATION_CENTROIDS = {
    "japan": (35.6762, 139.6503),
    "tokyo": (35.6762, 139.6503),
    "agra": (27.1767, 78.0081),
    "paris": (48.8566, 2.3522),
    "france": (48.8566, 2.3522),
    "bali": (-8.3405, 115.0920),
    "indonesia": (-8.3405, 115.0920),
    "new york": (40.7128, -74.0060),
    "usa": (40.7128, -74.0060),
    "london": (51.5074, -0.1278),
    "uk": (51.5074, -0.1278),
    "dubai": (25.2048, 55.2708),
    "singapore": (1.3521, 103.8198),
    "bangkok": (13.7563, 100.5018),
    "thailand": (13.7563, 100.5018),
    "goa": (15.2993, 74.1240),
    "kerala": (10.8505, 76.2711),
    "rome": (41.9028, 12.4964),
    "italy": (41.9028, 12.4964),
}
DEFAULT_CENTROID = (20.5937, 78.9629)  # geographic center of India, neutral fallback


def get_destination_centroid(destination: str) -> tuple[float, float]:
    key = (destination or "").strip().lower()
    return DESTINATION_CENTROIDS.get(key, DEFAULT_CENTROID)


def render_map(destination: str):
    """
    Client-side map exploration widget. Panning/pin placement here never
    calls the LLM -- only the explicit 'Lock this in' button below reports
    a decision back through the orchestrator.

    Tries live Google Places data first (if GOOGLE_PLACES_API_KEY is set);
    falls back to the static centroid lookup + generic area labels on any
    failure, so this never breaks the demo regardless of API/key state.
    """
    st.subheader(f"Explore {destination}")

    places_key = get_places_api_key()
    live_pois = []
    center_lat, center_lon = None, None

    if places_key:
        coords = geocode_destination_live(destination, places_key)
        if coords:
            center_lat, center_lon = coords
            live_pois = nearby_places_live(center_lat, center_lon, places_key)

    using_live = bool(places_key and center_lat is not None and live_pois)
    st.caption("📍 Live Google Places data" if using_live else "📍 Approximate (fallback data)")

    if center_lat is None:
        center_lat, center_lon = get_destination_centroid(destination)

    if live_pois:
        seed_pois = pd.DataFrame(live_pois)
    else:
        # Generic, destination-agnostic area labels centered on the actual
        # destination -- avoids showing wrong-city place names for any
        # destination not in DESTINATION_CENTROIDS.
        seed_pois = pd.DataFrame([
            {"name": "City Center", "lat": center_lat, "lon": center_lon},
            {"name": "Area A", "lat": center_lat + 0.02, "lon": center_lon - 0.015},
            {"name": "Area B", "lat": center_lat - 0.015, "lon": center_lon + 0.02},
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

    # Confirmation step: show what will be locked in before it's final,
    # rather than locking immediately on button click. Matches the map
    # exploration flow's "explicitly confirm lock-in" requirement.
    pin = st.session_state.map_pin
    with st.container(border=True):
        st.markdown(f"**Review your selection**")
        st.write(f"📍 **Area:** {pin['area']}")
        st.caption(f"Coordinates: {pin['lat']:.4f}, {pin['lng']:.4f}")
        if pin["nearby"]:
            other_spots = [n for n in pin["nearby"] if n != pin["area"]]
            if other_spots:
                st.caption(f"Nearby: {', '.join(other_spots)}")
        st.caption("This will be used as the base for your hotel search and itinerary. Change the dropdown above to pick a different area, or confirm below.")

        confirm_col, change_col = st.columns([1, 1])
        with confirm_col:
            if st.button("✅ Confirm & lock this in", type="primary", use_container_width=True):
                apply_map_selection(
                    st.session_state.conversation,
                    pin["area"], pin["lat"], pin["lng"], pin["nearby"],
                )
                lock_map_selection(st.session_state.conversation)
                st.session_state.map_destination = None
                st.success(f"Locked in {pin['area']} as your base.")
                st.rerun()
        with change_col:
            st.caption("👆 Or pick a different area above, then confirm.")


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
