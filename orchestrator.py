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
    family_form_offered: bool = False


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


def _normalize_composition(profile: dict) -> None:
    """
    Re-derive senior_citizen status from age for every member, and recompute
    the senior_citizens count, after ANY profile update -- not just ones
    that went through apply_family_members(). This matters because the LLM
    can also populate traveller_composition.members directly from
    conversation text (profile_updates), bypassing the form entirely, and
    that path has no built-in age->senior_citizen derivation of its own.
    Running this after every merge makes the derivation correct regardless
    of which path populated the data.
    """
    composition = profile.get("traveller_composition", {})
    members = composition.get("members")
    if not isinstance(members, list):
        return
    for m in members:
        if isinstance(m, dict):
            age = m.get("age")
            if isinstance(age, (int, float)) and age >= 60:
                m["senior_citizen"] = True
    senior_count = sum(1 for m in members if isinstance(m, dict) and m.get("senior_citizen"))
    if senior_count:
        composition["senior_citizens"] = senior_count


def apply_profile_updates(state: ConversationState, updates: dict) -> None:
    _deep_merge(state.profile, updates)
    _normalize_composition(state.profile)


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

    show_family_form = bool(turn.get("show_family_form", False)) and not state.family_form_offered
    if show_family_form:
        state.family_form_offered = True

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
        "show_family_form": show_family_form,
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


def apply_family_members(state: ConversationState, members: list[dict]) -> None:
    """
    Called by the UI layer when the user submits the family/group member
    form (client-side interaction, same pattern as apply_map_selection --
    not routed through the LLM). Each member dict: {"name", "age",
    "relation", "senior_citizen"}. Recomputes traveller_composition's
    senior_citizens count from flagged members rather than trusting a
    stale value.

    senior_citizen is derived from age (>=60) OR'd with any manually-set
    flag -- not read from the manual flag alone. Streamlit checkbox
    widgets only apply their `value=` default the first time they're
    rendered; if age is entered after the checkbox already exists, the
    checkbox keeps its stale old value instead of re-deriving from the
    new age. Computing it here at save time, from the final age, sidesteps
    that widget-staleness class of bug entirely.
    """
    composition = state.profile["traveller_composition"]
    composition["members"] = [
        {
            "name": m.get("name", "").strip(),
            "age": (age := m.get("age") or None),
            "relation": m.get("relation", "").strip() or None,
            "senior_citizen": bool(m.get("senior_citizen", False)) or (age is not None and age >= 60),
        }
        for m in members if m.get("name", "").strip()
    ]
    senior_count = sum(1 for m in composition["members"] if m["senior_citizen"])
    if senior_count:
        composition["senior_citizens"] = senior_count
