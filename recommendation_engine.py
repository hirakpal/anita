"""
Recommendation Engine

Deterministic, grounded layer downstream of the Chat Assistant. Consumes a
traveller_profile (see chat_assistant_prompt.py) and produces structured
recommendations across nine sub-engines. Does not converse with the user.

Hard rule: no hallucinated facts. Every concrete claim (price, name,
address, availability) must trace back to a retrieved/live data source.
Where a live integration isn't wired up yet, sub-engines return clearly
marked mock data via the `grounded=False` flag rather than inventing
plausible-looking results.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Optional, TYPE_CHECKING

from recommendation_engine_prompt import RECOMMENDATION_ENGINE_SYSTEM_PROMPT

if TYPE_CHECKING:
    from llm_client import LLMClient

# Sub-engine classification, per recommendation_engine_role.md:
#
# LLM-assisted (judgment layer -- uses RECOMMENDATION_ENGINE_SYSTEM_PROMPT
# via llm_client.complete_json(), NOT chat_structured() -- that method is
# reserved for the Chat Assistant's fixed turn schema):
#   rank_destinations, plan_activities, build_itinerary,
#   generate_packing_list, and rationale text on the deterministic engines.
#
# Deterministic-only (no LLM -- retrieved data + formula/filter logic):
#   rank_flights, rank_hotels (ranking math over live search results),
#   recommend_restaurants (hard filter first), optimize_budget (arithmetic),
#   analyze_risk (flag rules against structured advisory/health data).
#
# All sub-engines accept an optional `llm_client` (see llm_client.py's
# LLMClient protocol). When None, they fall back to placeholder output
# rather than silently degrading into fabricated content -- consistent
# with the no-hallucination guardrail.


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

TRIP_RECOMMENDATION_SCHEMA = {
    "destination_ranking": [],   # [{"destination": str, "score": float, "rationale": str, "grounded": bool}]
    "flight_ranking": [],        # [{"flight": {...}, "score": float, "rationale": str, "grounded": bool}]
    "hotel_ranking": [],         # [{"hotel": {...}, "score": float, "rationale": str, "grounded": bool}]
    "itinerary": [],             # [{"day": int, "items": [...], "notes": [...]}]
    "activities": [],            # [{"activity": {...}, "score": float, "rationale": str}]
    "restaurants": [],           # [{"restaurant": {...}, "score": float, "rationale": str}]
    "packing_list": [],          # [{"item": str, "category": str, "reason": str}]
    "budget_summary": {
        "planned_total": None,
        "budget_ceiling": None,
        "over_budget": False,
        "overage_amount": None,
        "breakdown": {},         # category -> planned amount
        "tradeoff_suggestions": [],
    },
    "risk_analysis": {
        "flags": [],              # [{"type": str, "severity": "low|medium|high", "detail": str, "source": str}]
        "summary": None,
    },
    "data_sources": {},           # sub_engine_name -> {"grounded": bool, "source": str|None}
}


def new_trip_recommendation() -> dict:
    """Return a fresh, deep-copied trip_recommendation output shape."""
    return copy.deepcopy(TRIP_RECOMMENDATION_SCHEMA)


# ---------------------------------------------------------------------------
# Sub-engines
#
# Each takes the traveller_profile (and, where relevant, upstream engine
# outputs) and returns its slice of TRIP_RECOMMENDATION_SCHEMA. These are
# stubs: wire in real search/data providers where marked TODO. Every stub
# marks its output as grounded=False until a real source is connected, per
# the no-hallucination guardrail.
# ---------------------------------------------------------------------------


def rank_destinations(profile: dict, llm_client: Optional['LLMClient'] = None) -> list[dict]:
    """Score candidate destinations against interests, climate, safety, budget.

    LLM-assisted: uses RECOMMENDATION_ENGINE_SYSTEM_PROMPT via llm_client to
    weigh interests/hidden scores against candidates and write rationale.
    Falls back to an unscored placeholder list if no llm_client is passed.
    """
    trip = profile.get("trip", {})
    destination = trip.get("destination", {})
    if not destination.get("flexible") and not destination.get("candidates"):
        return []  # destination already fixed, nothing to rank

    candidates = destination.get("candidates", [])
    interests = profile.get("interests", {})
    hidden = profile.get("hidden_preferences", {})

    if llm_client is None:
        return [{
            "destination": dest,
            "score": None,
            "rationale": (
                f"No judgment-layer LLM connected -- scoring not yet run "
                f"against interests={list(interests.keys())} or hidden "
                f"preference signals."
            ),
            "grounded": False,
        } for dest in candidates]

    # TODO: call llm_client.complete_json(system=RECOMMENDATION_ENGINE_SYSTEM_PROMPT,
    # messages=[{"role": "user", "content": json.dumps({"candidates": candidates,
    # "interests": interests, "hidden_preferences": hidden, ...})}]) and parse
    # a scored+rationale list from the structured response.
    return [{
        "destination": dest,
        "score": None,
        "rationale": "LLM judgment call not yet implemented -- placeholder.",
        "grounded": False,
    } for dest in candidates]


def rank_flights(profile: dict) -> list[dict]:
    """Rank flight options by fit (preferences + budget + comfort), not price alone."""
    prefs = profile.get("flight_preferences", {})
    budget = profile.get("budget", {}).get("flight_budget")

    # TODO: call a live flight search API (e.g. via SerpAPI/GDS) here.
    # Never fabricate flight numbers, times, or airlines in the meantime.
    return [{
        "flight": None,
        "score": None,
        "rationale": (
            f"No live flight search connected yet. Preferences captured: "
            f"{prefs}, budget ceiling: {budget}."
        ),
        "grounded": False,
    }]


def rank_hotels(profile: dict) -> list[dict]:
    """Rank hotels using accommodation prefs, budget, and locked stay_location as search center."""
    accommodation = profile.get("accommodation", {})
    budget = profile.get("budget", {}).get("hotel_budget")
    stay_location = profile.get("trip", {}).get("stay_location", {})
    locked = stay_location.get("locked", False)

    note = (
        "Search centered on locked stay_location."
        if locked else
        "stay_location not yet locked — results are provisional / wide-radius. "
        "Prompt the map exploration flow before finalizing."
    )

    # TODO: call a live hotel search API, centered on stay_location coordinates
    # when locked, with radius derived from pace/walking_tolerance.
    return [{
        "hotel": None,
        "score": None,
        "rationale": f"{note} Preferences: {accommodation}, budget ceiling: {budget}.",
        "grounded": False,
    }]


def build_itinerary(profile: dict, activities: list[dict], llm_client: Optional['LLMClient'] = None) -> list[dict]:
    """Assemble a day-by-day schedule from pace, duration, and planned activities.

    LLM-assisted: sequencing/pacing benefits from judgment (what goes
    together, what order makes sense) but never invents activities not
    present in the `activities` list passed in.
    """
    duration = profile.get("trip", {}).get("duration_days")
    pace = profile.get("pace", {})
    density = profile.get("hidden_preferences", {}).get("schedule_density")

    if not duration:
        return []

    attractions_per_day = pace.get("attractions_per_day")
    days = []
    for day_num in range(1, duration + 1):
        note = (
            f"attractions_per_day={attractions_per_day}, "
            f"schedule_density={density}, rest_time_needed="
            f"{pace.get('rest_time_needed')}"
        )
        if llm_client is None:
            note += " -- no judgment-layer LLM connected, sequencing not yet run."
        # TODO (with llm_client): call llm_client.complete_json(system=
        # RECOMMENDATION_ENGINE_SYSTEM_PROMPT, ...) to distribute `activities`
        # across days respecting pace/density and health-based walking limits.
        days.append({
            "day": day_num,
            "items": [],
            "notes": [note],
        })
    return days


def plan_activities(profile: dict, llm_client: Optional['LLMClient'] = None) -> list[dict]:
    """Select activities weighted by interests + hidden scores, filtered by health/safety.

    LLM-assisted: hard filters (accessibility/safety) are applied here in
    code before anything reaches the LLM -- the LLM only judges among
    already-eligible candidates, per the "hard constraints are
    non-negotiable" rule in RECOMMENDATION_ENGINE_SYSTEM_PROMPT.
    """
    interests = profile.get("interests", {})
    hidden = profile.get("hidden_preferences", {})
    health = profile.get("health", {})
    stay_location = profile.get("trip", {}).get("stay_location", {})
    poi_pool = stay_location.get("nearby_points_of_interest", [])

    # Hard filters first (accessibility/safety), before any scoring or LLM call.
    excluded_reasons = []
    if health.get("wheelchair") or health.get("walking_difficulty"):
        excluded_reasons.append("high-exertion / non-accessible activities excluded")

    if llm_client is None:
        return [{
            "activity": None,
            "score": None,
            "rationale": (
                f"No judgment-layer LLM connected. POI pool from map "
                f"exploration: {poi_pool}. Interest weights: {interests}. "
                f"Exclusions applied: {excluded_reasons or 'none'}."
            ),
        }]

    # TODO: call llm_client.complete_json(system=RECOMMENDATION_ENGINE_SYSTEM_PROMPT,
    # messages=[...]) with the (already-filtered) poi_pool/candidate activities,
    # interests, and hidden scores; parse a scored+rationale list back.
    return [{
        "activity": None,
        "score": None,
        "rationale": f"LLM judgment call not yet implemented. Exclusions applied: {excluded_reasons or 'none'}.",
    }]


def recommend_restaurants(profile: dict) -> list[dict]:
    """Filter by diet/allergies (hard), then rank by food_explorer_score and budget."""
    food = profile.get("food_profile", {})
    allergies = food.get("allergies", [])
    diet = food.get("diet", [])
    explorer_score = profile.get("hidden_preferences", {}).get("food_explorer_score")
    budget = profile.get("budget", {}).get("food_budget")

    # TODO: call a live restaurant search API; apply allergies/diet as hard
    # excludes before any ranking, never as a soft weight.
    return [{
        "restaurant": None,
        "score": None,
        "rationale": (
            f"No live restaurant search connected yet. Hard filters: "
            f"diet={diet}, allergies={allergies}. Ranking signal: "
            f"food_explorer_score={explorer_score}, budget ceiling={budget}."
        ),
    }]


def generate_packing_list(profile: dict, activities: list[dict], llm_client: Optional['LLMClient'] = None) -> list[dict]:
    """Derive packing list from climate, season, trip type, activities, and health needs.

    LLM-assisted for the generative expansion (climate/activity-driven
    items); health/family essentials below are deterministic and always
    included regardless of whether an llm_client is available.
    """
    trip = profile.get("trip", {})
    climate = profile.get("climate_preference", [])
    composition = profile.get("traveller_composition", {})
    health = profile.get("health", {})

    items = []
    if health.get("medicine_requirement"):
        items.append({
            "item": health["medicine_requirement"],
            "category": "health",
            "reason": "Stated medical requirement — always included regardless of other filters.",
        })
    if composition.get("infants"):
        items.append({
            "item": "infant care essentials",
            "category": "family",
            "reason": f"{composition.get('infants')} infant(s) traveling.",
        })

    if llm_client is None:
        return items

    # TODO: call llm_client.complete_json(system=RECOMMENDATION_ENGINE_SYSTEM_PROMPT,
    # ...) to expand `items` using destination climate/season data and the
    # assembled `activities` list (e.g. hiking gear if adventure activities
    # are planned). Merge results into `items` rather than replacing the
    # deterministic health/family entries above.
    return items


def optimize_budget(profile: dict, recommendation: dict) -> dict:
    """Reconcile planned costs against budget.overall; flag overages, never absorb silently."""
    budget = profile.get("budget", {})
    ceiling = budget.get("overall")
    sensitivity = profile.get("hidden_preferences", {}).get("budget_sensitivity")
    flexibility = budget.get("luxury_flexibility")

    # TODO: sum actual costs once flight/hotel/activity/restaurant results are
    # grounded. Until then, planned_total stays None rather than a fabricated
    # number.
    return {
        "planned_total": None,
        "budget_ceiling": ceiling,
        "over_budget": False,
        "overage_amount": None,
        "breakdown": {
            "flights": budget.get("flight_budget"),
            "hotel": budget.get("hotel_budget"),
            "food": budget.get("food_budget"),
            "activities": budget.get("activity_budget"),
            "shopping": budget.get("shopping_budget"),
        },
        "tradeoff_suggestions": [
            f"Budget optimization pending grounded pricing data. "
            f"budget_sensitivity={sensitivity}, luxury_flexibility={flexibility}."
        ],
    }


def analyze_risk(profile: dict, recommendation: dict) -> dict:
    """Cross-check destination/plan against safety preferences and health conditions. Informs, never decides."""
    health = profile.get("health", {})
    safety_prefs = profile.get("safety_preferences", [])
    destination = profile.get("trip", {}).get("destination", {}).get("confirmed", [])

    flags = []
    if health.get("altitude_issues"):
        flags.append({
            "type": "health",
            "severity": "medium",
            "detail": "Traveller has noted altitude issues — verify destination elevation before confirming.",
            "source": "user-stated health field",
        })
    if health.get("pregnant"):
        flags.append({
            "type": "health",
            "severity": "medium",
            "detail": "Pregnancy noted — verify airline/activity policies and destination medical access.",
            "source": "user-stated health field",
        })

    # TODO: cross-check `destination` against a live travel-advisory source
    # and append grounded flags (not fabricated ones) for safety_prefs like
    # "avoid_night_travel", "low_crime", etc.

    return {
        "flags": flags,
        "summary": (
            f"{len(flags)} flag(s) raised from stated profile data. "
            f"Live destination advisory check not yet connected."
        ),
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_recommendation_engine(profile: dict, llm_client: Optional['LLMClient'] = None) -> dict:
    """
    Run all sub-engines in dependency order and return an aggregated
    trip_recommendation. Order matters:
      Destination -> (Flight, Hotel) -> Activities -> Itinerary ->
      Restaurants -> Packing List -> Budget Optimizer -> Risk Analysis (last,
      since it evaluates the assembled plan, not just the raw profile).

    `llm_client` (see llm_client.py's LLMClient protocol) is passed only to
    the LLM-assisted judgment-layer sub-engines: rank_destinations,
    plan_activities, build_itinerary, generate_packing_list. Deterministic
    sub-engines (rank_flights, rank_hotels, recommend_restaurants,
    optimize_budget, analyze_risk) never receive it -- they stay
    formula/rule-based over retrieved data. Pass None to run in fully
    deterministic/placeholder mode (e.g. for testing).
    """
    result = new_trip_recommendation()

    result["destination_ranking"] = rank_destinations(profile, llm_client)
    result["flight_ranking"] = rank_flights(profile)
    result["hotel_ranking"] = rank_hotels(profile)

    result["activities"] = plan_activities(profile, llm_client)
    result["itinerary"] = build_itinerary(profile, result["activities"], llm_client)

    result["restaurants"] = recommend_restaurants(profile)
    result["packing_list"] = generate_packing_list(profile, result["activities"], llm_client)

    result["budget_summary"] = optimize_budget(profile, result)
    result["risk_analysis"] = analyze_risk(profile, result)

    result["data_sources"] = {
        "destination_ranking": {"grounded": False, "source": None},
        "flight_ranking": {"grounded": False, "source": None},
        "hotel_ranking": {"grounded": False, "source": None},
        "activities": {"grounded": False, "source": None},
        "restaurants": {"grounded": False, "source": None},
        "risk_analysis": {"grounded": False, "source": None},
    }

    return result
