"""Quick end-to-end test of the orchestrator using a scripted MockLLMClient."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_client import MockLLMClient
from orchestrator import ConversationState, process_turn, apply_map_selection, lock_map_selection

scripted_turns = [
    {
        "reply": "A Japan trip sounds wonderful! How many of you are traveling, and roughly how many days?",
        "profile_updates": {
            "trip": {"destination": {"confirmed": ["Japan"]}},
            "trip_objective": {"intent": "Vacation", "confidence": "medium", "inferred": True},
        },
        "trigger_recommendation": False,
        "show_map": None,
    },
    {
        "reply": "Great, 2 travelers for 5 days. Let me show you the map so you can pick where to stay.",
        "profile_updates": {
            "trip": {"duration_days": 5},
            "traveller_composition": {"adults": 2, "relationship": "couple"},
        },
        "trigger_recommendation": False,
        "show_map": {"destination": "Japan"},
    },
    {
        "reply": "Locked in Shibuya as your base. Building your recommendations now.",
        "profile_updates": {
            "interests": {"food": 9, "nature": 6},
            "budget": {"overall": 250000, "currency": "INR"},
        },
        "trigger_recommendation": True,
        "show_map": None,
    },
]

client = MockLLMClient(scripted_turns)
state = ConversationState()

print("--- Turn 1 ---")
result = process_turn(state, "I want to plan a trip to Japan", client)
print("Reply:", result["reply"])
print("Show map:", result["show_map"])
print("Recommendation triggered:", result["recommendation"] is not None)
assert state.profile["trip"]["destination"]["confirmed"] == ["Japan"]

print("\n--- Turn 2 ---")
result = process_turn(state, "2 of us, 5 days", client)
print("Reply:", result["reply"])
print("Show map:", result["show_map"])
assert result["show_map"] == {"destination": "Japan"}
assert state.pending_map_destination == "Japan"

print("\n--- Simulated client-side map interaction (not routed through LLM) ---")
apply_map_selection(state, "Shibuya", 35.6595, 139.7005, ["Shibuya Crossing", "Meiji Shrine"])
assert state.profile["trip"]["stay_location"]["locked"] is False
print("Selected area (pre-lock):", state.profile["trip"]["stay_location"]["selected_area"])

lock_map_selection(state)
assert state.profile["trip"]["stay_location"]["locked"] is True
assert state.pending_map_destination is None
print("Locked:", state.profile["trip"]["stay_location"]["locked"])

print("\n--- Turn 3 (should trigger Recommendation Engine) ---")
result = process_turn(state, "Locking that in, let's see the plan", client)
print("Reply:", result["reply"])
print("Recommendation triggered:", result["recommendation"] is not None)
assert result["recommendation"] is not None

rec = result["recommendation"]
print("\nRecommendation top-level keys:", list(rec.keys()))
print("Itinerary days:", len(rec["itinerary"]))
print("Hotel search note:", rec["hotel_ranking"][0]["rationale"])
print("\nFinal profile snapshot (selected fields):")
print(" destination:", state.profile["trip"]["destination"])
print(" duration_days:", state.profile["trip"]["duration_days"])
print(" traveller_composition:", state.profile["traveller_composition"])
print(" stay_location:", state.profile["trip"]["stay_location"])
print(" interests:", state.profile["interests"])
print(" budget:", state.profile["budget"])

print("\nAll assertions passed. Orchestration flow works end to end.")
