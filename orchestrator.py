"""
Orchestrator

Ties the Chat Assistant (LLM-driven profile building) to the Recommendation
Engine (deterministic, grounded ranking). Owns conversation state, merges
incremental profile updates from the LLM, decides when to trigger the
Recommendation Engine, and handles the map-exploration handoff described in
chat_assistant_role.md.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

from chat_assistant_prompt import CHAT_ASSISTANT_SYSTEM_PROMPT, new_traveller_profile
from llm_client import LLMClient
from recommendation_engine import run_recommendation_engine


@dataclass
class ConversationState:
    profile: dict = field(default_factory=new_traveller_profile)
    messages: list[dict[str, str]] = field(default_factory=list)
    recommendation: Optional[dict] = None
    pending_map_destination: Optional[str] = None
    map_locked: bool = False


def _deep_merge(base: dict, updates: dict) -> dict:
    """
    Merge `updates` into `base` in place, recursively. Lists overwrite
    (the LLM sends full lists for fields like interests/candidates, not
    diffs). None/empty values in `updates` never erase existing data --
    only explicit non-empty values overwrite.
    """
    for key, value in updates.items():
        if value is None or value == {} or value == []:
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def apply_profile_updates(state: ConversationState, updates: dict) -> None:
    _deep_merge(state.profile, updates)


def profile_is_sufficient(profile: dict) -> bool:
    """
    Minimal viability check before the Recommendation Engine can run
    meaningfully: needs a destination (fixed or candidates), some notion of
    dates/duration, and traveller composition. Mirrors the "direct-ask
    tier" from chat_assistant_role.md.
    """
    trip = profile.get("trip", {})
    destination = trip.get("destination", {})
    has_destination = bool(destination.get("confirmed") or destination.get("candidates"))
    has_duration = bool(trip.get("duration_days"))
    composition = profile.get("traveller_composition", {})
    has_travelers = bool(composition.get("adults"))
    return has_destination and has_duration and has_travelers


def process_turn(state: ConversationState, user_message: str, llm_client: LLMClient) -> dict:
    """
    Process one user message: call the LLM for a structured turn, merge
    profile updates, optionally trigger the map flow or the Recommendation
    Engine, and return everything the UI layer needs to render.

    Returns:
      {
        "reply": str,
        "show_map": {"destination": str} | None,
        "recommendation": dict | None,   # only set if triggered this turn
      }
    """
    state.messages.append({"role": "user", "content": user_message})

    turn = llm_client.chat_structured(
        system=CHAT_ASSISTANT_SYSTEM_PROMPT,
        messages=state.messages,
    )

    apply_profile_updates(state, turn.get("profile_updates", {}))
    state.messages.append({"role": "assistant", "content": turn.get("reply", "")})

    show_map = turn.get("show_map")
    if show_map:
        state.pending_map_destination = show_map.get("destination")

    recommendation = None
    should_trigger = turn.get("trigger_recommendation", False) and profile_is_sufficient(state.profile)
    if should_trigger:
        # Same llm_client instance is reused here -- it implements both
        # chat_structured() (used above, fixed turn schema) and
        # complete_json() (used internally by the Recommendation Engine's
        # judgment-layer sub-engines, generic per-call schema). See
        # llm_client.py's LLMClient protocol for the split.
        recommendation = run_recommendation_engine(state.profile, llm_client)
        state.recommendation = recommendation

    return {
        "reply": turn.get("reply", ""),
        "show_map": show_map,
        "recommendation": recommendation,
    }


def apply_map_selection(
    state: ConversationState,
    area_name: str,
    lat: float,
    lng: float,
    nearby_pois: list[str],
) -> None:
    """
    Called by the UI layer when the user adjusts the map pin (client-side
    interaction, not routed through the LLM per the assistant/frontend
    split in chat_assistant_role.md). Updates the in-progress selection but
    does NOT lock it -- locking is a separate, explicit confirmation step.
    """
    stay_location = state.profile["trip"]["stay_location"]
    stay_location["selected_area"] = area_name
    stay_location["coordinates"] = {"lat": lat, "lng": lng}
    stay_location["nearby_points_of_interest"] = nearby_pois
    stay_location["locked"] = False
    state.map_locked = False


def lock_map_selection(state: ConversationState) -> None:
    """Explicit lock-in confirmation, per the map exploration flow's step 5."""
    state.profile["trip"]["stay_location"]["locked"] = True
    state.map_locked = True
    state.pending_map_destination = None
