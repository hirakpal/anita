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
import json
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
    """
    Merge LLM-authored profile_updates into state.profile, with two
    deliberate exceptions -- fields that are UI-owned ground truth and
    must only ever be written by the real user-driven flow, never by the
    LLM claiming something happened in its own profile_updates:

    - traveller_composition.members: only apply_family_members() (the
      form's Save button) may write it.
    - trip.stay_location: only apply_map_selection() / lock_map_selection()
      (the map UI's pin selection and explicit lock-in button) may write
      it.

    Why stay_location needed this too: confirmed live -- the model would
    narrate "let me show you a map" and then, in a LATER turn, claim a
    specific area was "confirmed" and locked, without the user ever
    having interacted with the actual map widget at all. Since nothing
    was stripping trip.stay_location from LLM-authored updates, that
    self-reported claim silently became the real profile state -- the
    exact same fabrication-without-real-interaction pattern as the
    members incident, just for location instead of names.
    """
    sanitized = copy.deepcopy(updates)
    composition_update = sanitized.get("traveller_composition")
    if isinstance(composition_update, dict):
        composition_update.pop("members", None)
    trip_update = sanitized.get("trip")
    if isinstance(trip_update, dict):
        trip_update.pop("stay_location", None)
    _deep_merge(state.profile, sanitized)
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


def _build_dynamic_profile_context(profile: dict) -> str:
    """
    Build the DYNAMIC (uncached) per-turn context block: the current
    accumulated profile state as grounded JSON. Kept separate from
    CHAT_ASSISTANT_SYSTEM_PROMPT (static, same every turn) specifically so
    the static part can be cached via cache_control -- if this dynamic
    block were concatenated into the same string as the static prompt,
    every turn would produce a different prefix and never hit the cache,
    since profile state changes turn to turn.

    Why this content exists at all: without it, the model only ever sees
    the plain reply TEXT of its own past turns in conversation history --
    never the structured profile_updates it emitted via tool calls. That
    leaves it unable to tell what's already known vs. still missing,
    which produces two symptoms at once: re-asking for information
    already captured, and (more seriously) fabricating plausible-sounding
    values to fill out the schema since it has no grounded state to check
    itself against. Explicit instruction here closes both: check this
    block before asking anything, and never invent a value not present
    in it.
    """
    try:
        profile_json = json.dumps(profile, indent=2)
    except (TypeError, ValueError):
        # Defensive: if anything non-serializable ever slips into the
        # profile, degrade to no state-injection for this turn rather
        # than crashing the whole Streamlit process (which would wipe
        # session_state for every connected user, not just this session).
        profile_json = "{}"

    return (
        "## Current known traveller_profile state (ground truth)\n"
        + "This is everything actually captured so far -- from this "
        + "conversation, the map exploration flow, or the family/group "
        + "member form. Check this before asking the user anything: do "
        + "not re-ask for fields that already have a non-null/non-empty "
        + "value here. Do not invent or assume values for anything shown "
        + "as null/empty -- only set a field in profile_updates when the "
        + "user has actually just told you that information.\n\n```json\n"
        + profile_json
        + "\n```\n"
    )


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
        dynamic_context=_build_dynamic_profile_context(state.profile),
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
