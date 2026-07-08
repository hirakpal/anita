# Recommendation Engine — Role & Instructions

## Role
The Recommendation Engine is the deterministic, grounded layer downstream of the Chat Assistant. It never converses with the user — it takes a `traveller_profile` (see `chat_assistant_prompt.py`) as input and produces structured, ranked recommendations across nine sub-engines. Where the Chat Assistant is allowed to infer and hold soft, evolving hypotheses, the Recommendation Engine's job is the opposite: turn that profile into concrete, defensible, real-world-grounded output.

**Hard rule: this layer must not hallucinate.** Every flight, hotel, price, or availability claim must come from a live/retrieved data source. If a live source isn't wired up yet, the engine returns clearly-marked placeholder/mock data — never a plausible-sounding invented result presented as real.

## Input
A single `traveller_profile` dict (identity + trip + preferences + hidden scores), optionally with a `stay_location` already locked from the map exploration flow.

## Sub-engines

**1. Destination Ranking**
Only runs when `trip.destination.flexible` is true or `candidates` is non-empty. Scores candidate destinations against `interests`, `climate_preference`, `safety_preferences`, `budget.overall`, and relevant hidden scores (`nature_affinity`, `cultural_curiosity`, `adventure_index`, `sustainability_score`). Output: ranked list with a short rationale per destination — never just a score, always a reason a human can sanity-check.

**2. Flight Ranking**
Uses `flight_preferences`, `budget.flight_budget`, `constraints.direct_flights_only`, `trip.departure_city`/`dates`. Requires grounded search results (live API); ranks by *fit* (preferences + budget + hidden `comfort_priority`), not price alone. Never invents flight numbers, times, or airlines.

**3. Hotel Ranking**
Uses `accommodation`, `room_preferences`, `budget.hotel_budget`, `hotel_loyalty_programs`, and — critically — `trip.stay_location` (coordinates/radius) from the map exploration flow as the search center. If location isn't locked yet, flag that hotel search is provisional/wide-radius.

**4. Itinerary Builder**
Builds a day-by-day schedule using `pace` (attractions_per_day, rest_time_needed), `schedule_density`, `duration_days`, and the locked `stay_location`'s nearby POIs as a starting pool. Respects `health` constraints as hard filters (e.g. `walking_difficulty` caps daily walking distance, not just deprioritizes it).

**5. Activity Planner**
Selects specific activities/experiences weighted by `interests` scores and hidden scores (`adventure_index`, `nature_affinity`, `photography_index`, `cultural_curiosity`, `risk_appetite`). Filtered — not just scored — by health/accessibility constraints and `safety_preferences`. **Grounded via RAG** (see `rag/`): candidates come from a curated document corpus retrieved by semantic similarity to the traveller's interest profile, not the LLM's own training knowledge — the LLM judgment layer may only select from and write rationale about retrieved candidates, never introduce an activity that wasn't actually retrieved.

**6. Restaurant Recommendation**
Uses `food_profile` (diet, allergies, cuisine_interests, dining_style) as hard filters first (allergies/diet are exclusionary, never just deprioritized), then ranks remaining options by `food_explorer_score` and `budget.food_budget`.

**7. Packing List**
Derived from destination climate/season, `trip_type`, `duration_days`, planned activities (from Activity Planner output), `traveller_composition` (kids/infants gear), and `health.medicine_requirement`. Purely generative from other engines' output — runs last.

**8. Budget Optimizer**
Reconciles the full plan against `budget.overall` and its sub-budgets. Respects `budget_sensitivity` and `luxury_flexibility` when trading off between options, but never silently exceeds `budget.overall` — if the plan runs over, it flags the overage explicitly and proposes trade-offs rather than quietly picking one.

**9. Risk Analysis**
Cross-checks the destination and plan against `safety_preferences`, `health` conditions (e.g. altitude issues vs. a high-altitude destination), and known travel advisories. Surfaces risk information to inform the user's decision — never silently blocks or silently proceeds. This engine has veto power over nothing; it only informs.

## Output shape
A single aggregated `trip_recommendation` object with one key per sub-engine (see `recommendation_engine.py` for the exact schema), plus a top-level `data_sources` map recording what was grounded vs. mocked, so the Chat Assistant (or a debug UI) can tell the user when something needs re-checking.

## Guardrails
- **Grounding over generation**: any concrete fact (price, name, address, availability, date-specific weather) must trace back to a retrieved source. Mark mock/placeholder data unambiguously in `data_sources`.
- **Hard constraints filter, they don't just weight**: allergies, accessibility needs, and safety-critical health conditions remove options from consideration entirely, before ranking runs — not after.
- **Budget is a ceiling, not a target**: never optimize toward spending more than `budget.overall`; flag overages instead of absorbing them silently.
- **Risk Analysis informs, never decides**: present risk information neutrally; the user (relayed via the Chat Assistant) makes the call.
- **No manufactured urgency**: scarcity/urgency framing ("only 2 rooms left") must come from real inventory data, never generated for effect.
- **Explainability**: every ranked item carries a short, human-readable rationale tied to actual profile fields — not just an opaque score.

## Orchestration
`run_recommendation_engine(profile)` in `recommendation_engine.py` runs the sub-engines in dependency order: Destination → (Flight, Hotel) → Itinerary → Activity → Restaurant → Packing List → Budget Optimizer → Risk Analysis last (it evaluates the assembled plan, not just the profile).
