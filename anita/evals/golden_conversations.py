"""
Golden conversation scenarios for the eval framework.

Each scenario is a scripted multi-turn conversation with deterministic
checks run against the final state. Several of these directly encode real
bugs caught during live testing (see the docstring on each scenario) --
turning production incidents into permanent regression tests is the whole
point of this file existing.

Scenarios run in two modes:
  - mock:  scripted MockLLMClient responses drive each turn. Tests the
           HARNESS (merge logic, structural protections) deterministically,
           with no API key needed. Always runnable, always the same result.
  - live:  only `user_message` is used; a real AnthropicLLMClient generates
           each reply. Tests actual MODEL quality/behavior, not just the
           harness -- requires ANTHROPIC_API_KEY and is non-deterministic
           by nature (the model might phrase things differently turn to
           turn), so checks here should test outcomes, not exact wording.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class GoldenTurn:
    user_message: str
    # Used only in mock mode. Shape must match the chat-turn schema
    # (see llm_client.py's CHAT_TURN_TOOL / _normalize_turn).
    mock_response: Optional[dict] = None


@dataclass
class Check:
    description: str
    # fn receives (final_profile_dict, list_of_assistant_replies, final_conversation_state)
    fn: Callable[[dict, list[str], Any], bool]


@dataclass
class GoldenConversation:
    name: str
    description: str
    turns: list[GoldenTurn]
    checks: list[Check]
    # Optional setup hook run before any turns (e.g. to simulate a form
    # submission via apply_family_members/apply_map_selection, which
    # happen client-side and aren't LLM turns themselves).
    setup: Optional[Callable[[Any], None]] = None


# ---------------------------------------------------------------------------
# Scenario 1: basic profile extraction across turns
# ---------------------------------------------------------------------------

SCENARIO_BASIC_EXTRACTION = GoldenConversation(
    name="basic_profile_extraction",
    description=(
        "Baseline sanity check: destination, dates, composition, and "
        "budget given across a few turns should all land correctly in "
        "the profile via _deep_merge, and never get silently dropped."
    ),
    turns=[
        GoldenTurn(
            user_message="Hi, I'm Priya, thinking about a trip to Kerala",
            mock_response={
                "reply": "Great to meet you, Priya! When are you thinking of going, and for how long?",
                "profile_updates": {
                    "traveller_identity": {"name": "Priya"},
                    "trip": {"destination": {"confirmed": ["Kerala"]}},
                },
                "trigger_recommendation": False,
                "show_map": None,
                "show_family_form": False,
            },
        ),
        GoldenTurn(
            user_message="March 2027, 6 days, budget 80000 INR, just me",
            mock_response={
                "reply": "Perfect, 6 days solo in Kerala this March with an ₹80,000 budget!",
                "profile_updates": {
                    "trip": {"duration_days": 6},
                    "traveller_composition": {"adults": 1, "relationship": "solo"},
                    "budget": {"overall": 80000, "currency": "INR"},
                },
                "trigger_recommendation": False,
                "show_map": None,
                "show_family_form": False,
            },
        ),
    ],
    checks=[
        Check("name captured", lambda p, r, s: p["traveller_identity"]["name"] == "Priya"),
        Check("destination captured", lambda p, r, s: "Kerala" in p["trip"]["destination"]["confirmed"]),
        Check("duration captured", lambda p, r, s: p["trip"]["duration_days"] == 6),
        Check("adults captured", lambda p, r, s: p["traveller_composition"]["adults"] == 1),
        Check("budget captured", lambda p, r, s: p["budget"]["overall"] == 80000),
    ],
)


# ---------------------------------------------------------------------------
# Scenario 2: no fabrication of family member names (regression test)
# ---------------------------------------------------------------------------
# INCIDENT: live testing caught the model inventing specific names ("Rumi",
# "Anita", "Rahul") and exact ages for a user's family after they only said
# "Wife, Mother, Kid" -- never provided via chat text or the form. Root
# cause was the model reasoning without grounded state. Fixed at two
# layers: prompt guardrail + structural enforcement (apply_profile_updates
# strips traveller_composition.members from any LLM-authored update).
# This scenario proves the structural fix holds even if a future prompt
# change reintroduces the model's tendency to fabricate.

SCENARIO_NO_FABRICATION = GoldenConversation(
    name="no_fabrication_of_family_names",
    description=(
        "REGRESSION TEST for a real incident: the model must never be able "
        "to write specific names/ages into traveller_composition.members "
        "that the user did not actually provide. Simulates the exact "
        "failure by scripting a mock turn that TRIES to fabricate names "
        "(reproducing what the live model actually did), then asserts the "
        "structural protection in apply_profile_updates blocks it."
    ),
    turns=[
        GoldenTurn(
            user_message="Family trip, wife, mother, kid",
            mock_response={
                "reply": "Lovely -- a family trip with your wife, mother, and kid!",
                "profile_updates": {
                    "traveller_composition": {
                        "adults": 3,
                        "children": 1,
                        "relationship": "family",
                        # This is the actual fabrication that happened live:
                        # specific invented names/ages the user never gave.
                        "members": [
                            {"name": "Rumi", "age": 39, "relation": "Spouse", "senior_citizen": False},
                            {"name": "Anita", "age": 75, "relation": "Mother", "senior_citizen": True},
                            {"name": "Rahul", "age": 10, "relation": "Son", "senior_citizen": False},
                        ],
                    }
                },
                "trigger_recommendation": False,
                "show_map": None,
                "show_family_form": True,
            },
        ),
    ],
    checks=[
        Check(
            "members list must stay empty -- user never provided names, only the form may set this",
            lambda p, r, s: p["traveller_composition"]["members"] == [],
        ),
        Check(
            "composition counts (adults/children) still pass through normally",
            lambda p, r, s: p["traveller_composition"]["adults"] == 3 and p["traveller_composition"]["children"] == 1,
        ),
    ],
)


# ---------------------------------------------------------------------------
# Scenario 3: form-entered member data survives a later incomplete restatement
# ---------------------------------------------------------------------------
# INCIDENT: after the auto-continue fix (sending a synthetic follow-up turn
# after form Save), the model's NEXT turn restated members incompletely
# (name+relation only, dropping age/senior_citizen), and since list fields
# overwrite wholesale, it silently wiped the correct form data. Fixed by
# making members write-once-via-form-only.

SCENARIO_MEMBER_DATA_SURVIVES = GoldenConversation(
    name="member_data_survives_followup_restatement",
    description=(
        "REGRESSION TEST for a real incident: form-entered age/senior_citizen "
        "data was wiped by the model's own next turn restating members "
        "incompletely. Simulates the form save (via setup) then a turn that "
        "reproduces the incomplete restatement, and asserts the original "
        "data survives."
    ),
    setup=lambda state: __import__("orchestrator").apply_family_members(state, [
        {"name": "Rumi", "age": 39, "relation": "Spouse", "senior_citizen": False},
        {"name": "Anita", "age": 75, "relation": "Mother", "senior_citizen": True},
    ]),
    turns=[
        GoldenTurn(
            user_message="I've added who's traveling with me: Rumi (Spouse), Anita (Mother).",
            mock_response={
                "reply": "Wonderful -- Rumi and Anita joining you!",
                "profile_updates": {
                    "traveller_composition": {
                        # Incomplete restatement, exactly like the live incident:
                        # missing age/senior_citizen entirely.
                        "members": [
                            {"name": "Rumi", "relation": "Spouse"},
                            {"name": "Anita", "relation": "Mother"},
                        ]
                    }
                },
                "trigger_recommendation": False,
                "show_map": None,
                "show_family_form": False,
            },
        ),
    ],
    checks=[
        Check(
            "Anita's age must survive the model's incomplete restatement",
            lambda p, r, s: next(m for m in p["traveller_composition"]["members"] if m["name"] == "Anita")["age"] == 75,
        ),
        Check(
            "Anita's senior_citizen flag must survive",
            lambda p, r, s: next(m for m in p["traveller_composition"]["members"] if m["name"] == "Anita")["senior_citizen"] is True,
        ),
    ],
)


# ---------------------------------------------------------------------------
# Scenario 4: senior citizen auto-derivation from age
# ---------------------------------------------------------------------------
# INCIDENT: a Streamlit checkbox widget only applies its `value=` default
# the first time it renders; entering age AFTER the checkbox already
# exists left it stale. Fixed by deriving senior_citizen from age at
# save/merge time regardless of the checkbox's (possibly stale) value.

SCENARIO_SENIOR_CITIZEN_DERIVATION = GoldenConversation(
    name="senior_citizen_auto_derivation",
    description=(
        "REGRESSION TEST for a real incident: age 75 with an unchecked "
        "senior_citizen checkbox (stale widget state) must still result "
        "in senior_citizen=True after apply_family_members."
    ),
    setup=lambda state: __import__("orchestrator").apply_family_members(state, [
        {"name": "Anita", "age": 75, "relation": "Mother", "senior_citizen": False},  # stale checkbox
    ]),
    turns=[],  # no LLM turns needed -- this tests the form-save path directly
    checks=[
        Check(
            "senior_citizen must be derived from age, not trusted from a stale checkbox",
            lambda p, r, s: p["traveller_composition"]["members"][0]["senior_citizen"] is True,
        ),
        Check(
            "senior_citizens count must be recomputed",
            lambda p, r, s: p["traveller_composition"]["senior_citizens"] == 1,
        ),
    ],
)


# ---------------------------------------------------------------------------
# Scenario 5: map lock-in auto-continues instead of stalling
# ---------------------------------------------------------------------------
# INCIDENT: locking in a map selection is client-side only (by design, no
# LLM round-trip for pin dragging) -- but that meant the conversation just
# stalled after lock-in until the user typed something unprompted. Fixed
# by sending a synthetic follow-up turn immediately after lock-in.

SCENARIO_MAP_LOCKIN_CONTINUES = GoldenConversation(
    name="map_lockin_auto_continues",
    description=(
        "REGRESSION TEST for a real incident: after map lock-in, the "
        "Recommendation Engine never triggered because nothing prompted "
        "the LLM to evaluate trigger_recommendation. Simulates lock-in "
        "(via setup) with an otherwise-sufficient profile, then the "
        "synthetic auto-continue turn, and asserts recommendation fires."
    ),
    setup=lambda state: (
        state.profile["trip"]["destination"]["confirmed"].append("Agra"),
        state.profile["trip"].__setitem__("duration_days", 3),
        state.profile["traveller_composition"].__setitem__("adults", 2),
        __import__("orchestrator").apply_map_selection(
            state, "The Grand Imperial", 27.1735, 78.0088, ["Idgah railway station"],
        ),
        __import__("orchestrator").lock_map_selection(state),
    ),
    turns=[
        GoldenTurn(
            user_message="I've locked in The Grand Imperial as my base for the trip.",
            mock_response={
                "reply": "Perfect, building your recommendations now!",
                "profile_updates": {},
                "trigger_recommendation": True,
                "show_map": None,
                "show_family_form": False,
            },
        ),
    ],
    checks=[
        Check("stay_location must be locked", lambda p, r, s: p["trip"]["stay_location"]["locked"] is True),
        Check("recommendation must have triggered", lambda p, r, s: s.recommendation is not None),
    ],
)


# ---------------------------------------------------------------------------
# Scenario 6: no fabrication of a locked stay_location (regression test)
# ---------------------------------------------------------------------------
# INCIDENT: live testing caught the assistant narrating "let me show you a
# map" and then, in a LATER turn, claiming a specific area was "confirmed"
# and "locked" -- without the user ever having interacted with the actual
# map widget. Root cause: nothing was stripping trip.stay_location from
# LLM-authored profile_updates, so the model's self-reported claim silently
# became the real profile state. Same fabrication-without-real-interaction
# pattern as the family member names incident, just for location.

SCENARIO_NO_FABRICATED_LOCATION = GoldenConversation(
    name="no_fabrication_of_locked_location",
    description=(
        "REGRESSION TEST for a real incident: the model must never be able "
        "to write a 'locked' stay_location into the profile that the user "
        "never actually confirmed through the real map UI. Simulates the "
        "exact failure by scripting a mock turn that TRIES to claim a "
        "location is locked (reproducing what the live model actually "
        "did), then asserts the structural protection blocks it."
    ),
    turns=[
        GoldenTurn(
            user_message="Sounds good, let's go with Agra",
            mock_response={
                "reply": "Perfect, Idgah is a solid central base -- confirmed!",
                "profile_updates": {
                    "trip": {
                        "stay_location": {
                            "selected_area": "Idgah Railway Station",
                            "coordinates": {"lat": 27.17, "lng": 78.00},
                            "locked": True,
                        }
                    }
                },
                "trigger_recommendation": False,
                "show_map": None,
                "show_family_form": False,
            },
        ),
    ],
    checks=[
        Check(
            "stay_location must stay unlocked -- user never interacted with the real map widget",
            lambda p, r, s: p["trip"]["stay_location"]["locked"] is False,
        ),
        Check(
            "selected_area must stay empty -- only the real map UI may set this",
            lambda p, r, s: p["trip"]["stay_location"]["selected_area"] is None,
        ),
    ],
)


ALL_SCENARIOS: list[GoldenConversation] = [
    SCENARIO_BASIC_EXTRACTION,
    SCENARIO_NO_FABRICATION,
    SCENARIO_MEMBER_DATA_SURVIVES,
    SCENARIO_SENIOR_CITIZEN_DERIVATION,
    SCENARIO_MAP_LOCKIN_CONTINUES,
    SCENARIO_NO_FABRICATED_LOCATION,
]
