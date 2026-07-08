# ANITA
**A**daptive **N**avigation & **I**tinerary **T**ravel **A**ssistant

A conversational AI travel planning assistant that builds a rich, 21-category
traveller profile through natural conversation (not a form), infers hidden
preference signals rather than asking for them directly, and hands off to a
grounded Recommendation Engine for destination, flight, hotel, itinerary,
activity, restaurant, packing, budget, and risk output.

## Architecture

```
User
  │
  ▼
Chat Assistant  ──────────────────────────────────────────
  │  Understand Intent → Ask Intelligent Questions →
  │  Build Traveller Profile → Detect Preferences →
  │  Infer Hidden Interests → Build Structured JSON
  ▼
Recommendation Engine  ────────────────────────────────────
  │  Destination Ranking · Flight Ranking · Hotel Ranking
  │  Itinerary Builder · Activity Planner · Restaurant Rec.
  │  Packing List · Budget Optimizer · Risk Analysis
  ▼
Streamlit UI
```

Two LLM-facing prompts, deliberately separate:
- **Chat Assistant** (`chat_assistant_prompt.py`) — conversational, elicits
  and infers the traveller profile, never converses about internals.
- **Recommendation Engine judgment layer** (`recommendation_engine_prompt.py`)
  — non-conversational, scores/sequences/explains over data it's given,
  never invents concrete facts (prices, names, availability).

See `docs/chat_assistant_role.md` and `docs/recommendation_engine_role.md`
for the full design rationale, guardrails, and field-by-field spec behind
each.

## Project structure

```
anita/
├── chat_assistant_prompt.py       # Chat Assistant system prompt + traveller_profile schema
├── recommendation_engine_prompt.py # Recommendation Engine judgment-layer system prompt
├── recommendation_engine.py       # 9 sub-engines + orchestration
├── llm_client.py                  # LLMClient protocol: Anthropic + Mock implementations
├── orchestrator.py                # Ties Chat Assistant ↔ Recommendation Engine together
├── streamlit_app.py                # UI: chat, map exploration, recommendation display
├── tests/
│   └── test_orchestrator.py       # End-to-end orchestration test (MockLLMClient)
└── docs/
    ├── chat_assistant_role.md
    ├── recommendation_engine_role.md
    └── ui_mockup.html             # Visual design mockup (boarding-pass/departures-board direction)
```

## Design language

The UI follows a **boarding pass / departures board** visual direction —
charcoal/graphite base, JetBrains Mono for status chips and data (split-flap
board feel), IBM Plex Sans for chat text, perforated ticket-stub dividers as
the signature element. See `docs/ui_mockup.html`.

## Running it

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Defaults to a scripted `MockLLMClient` demo conversation — no API key
needed to try it. To go live, set `ANTHROPIC_API_KEY` and flip
`USE_REAL_LLM = True` in `streamlit_app.py`.

## Testing

```bash
python tests/test_orchestrator.py       # end-to-end orchestration test
python tests/test_prompt_caching.py     # cache placement + stats tracking
```

## Evals

Golden conversation scenarios, several encoding real bugs caught during
live testing as permanent regression tests. See `evals/README.md`.

```bash
python evals/run_evals.py               # mock mode, no API key needed
python evals/run_evals.py --client live # real model, needs ANTHROPIC_API_KEY
```

## Status

Early scaffold. Chat Assistant profile-building and orchestration logic are
tested end-to-end against a scripted conversation. Prompt caching is
implemented and verified (static system prompt cached, dynamic profile
state sent fresh each turn). Recommendation Engine sub-engines are
structured with the correct guardrails (hard-constraint filtering, no
fabricated facts, budget-as-ceiling) but most are stubs pending live data
source integration — see `TODO`s in `recommendation_engine.py`.
