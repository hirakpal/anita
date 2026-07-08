# ANITA
**A**daptive **N**avigation & **I**tinerary **T**ravel **A**ssistant

A conversational AI travel planning assistant that builds a rich, 21-category traveller profile through natural conversation — not a form — infers hidden preference signals rather than asking for them directly, and hands off to a grounded Recommendation Engine for destination, flight, hotel, itinerary, activity, restaurant, packing, budget, and risk output.

Built as a Gen AI capstone with two goals: ship a working, demoable product under real deadline pressure, and use the time after that to deliberately close the gap between "works for a demo" and genuine Gen AI engineering depth — prompt caching, an eval framework, retrieval-augmented generation, and observability, each with its own test suite rather than a slide-only concept.

**→ Full documentation, including problem statement, competitive research, Gen AI concept coverage map, and architecture diagrams: [`docs/PROJECT_DOCUMENTATION.md`](docs/PROJECT_DOCUMENTATION.md)**

---

## The problem

Existing AI travel tools — general-purpose assistants and dedicated planners alike — reliably produce itineraries fast, but travelers can't tell what to trust in them, and the tools don't get better at helping them personally over time. Three recurring failures: recommendations stated as flat fact with no reliability signal (leading to hallucinated details and stale pricing), shallow and short-lived preference memory, and tools that break under real-world complexity (multi-traveler groups, accessibility needs, shared budgets).

*(Full competitive landscape research and honest gap analysis in the documentation linked above.)*

## What's actually built vs. designed-but-deferred

| | Status |
|---|---|
| Grounded activity recommendations (RAG + live location data) | ✅ Built |
| Multi-traveler / accessibility / shared-budget handling | ✅ Built, tested live with a real multi-generational family trip |
| Durable cross-session memory | 🔲 Schema designed for it; persistence explicitly deferred ([decision record](docs/decisions/0001-traveller-profile-persistence.md)) |
| Live flight/hotel/restaurant data | 🔲 Stubbed with correct guardrail structure, pending provider integration |

This project tries to be honest about that split rather than presenting everything as finished — see the documentation's "Known Gaps" section for the full list.

## Architecture

```
User
  │
  ▼
Chat Assistant  ──────────────────────────────────────────
  │  Forced tool-calling (record_turn) for structured output
  │  Real schema embedded in prompt + current profile state
  │  injected every turn — the fix for two real fabrication
  │  incidents caught during live testing (see docs)
  ▼
Orchestrator  ─────────────────────────────────────────────
  │  Merges profile updates · write-protects UI-owned fields
  │  (family members, locked location) from the LLM itself
  ▼
Recommendation Engine  ────────────────────────────────────
  │  Destination Ranking · Flight Ranking · Hotel Ranking
  │  Itinerary Builder · Activity Planner (RAG-grounded) ·
  │  Restaurant Rec. · Packing List · Budget Optimizer ·
  │  Risk Analysis
  ▼
Streamlit UI
```

Two LLM-facing prompts, deliberately separate:
- **Chat Assistant** (`chat_assistant_prompt.py`) — conversational, elicits and infers the traveller profile, never converses about internals.
- **Recommendation Engine judgment layer** (`recommendation_engine_prompt.py`) — non-conversational, scores/sequences/explains over data it's given, never invents concrete facts (prices, names, availability).

No agent framework (LangGraph/CrewAI/LangChain) — orchestration is built directly against the Anthropic SDK for full control over provider-specific mechanics (prompt caching, forced tool-calling) and debuggability during incident response. Reasoning documented in [decision record 0002](docs/decisions/0002-no-agent-framework.md).

See `docs/chat_assistant_role.md` and `docs/recommendation_engine_role.md` for full design rationale, guardrails, and field-by-field spec.

## Gen AI concepts covered

Structured output via forced tool-calling · context grounding / anti-hallucination via state injection · prompt caching · retrieval-augmented generation (curated corpus + live Google Places data) · observability/tracing · graceful degradation on API failure · deliberate LLM-vs-deterministic sub-engine boundaries · an eval framework built directly from real incidents caught in live testing.

Full coverage map (what's built, what's not, and why) in the [documentation](docs/PROJECT_DOCUMENTATION.md#4-gen-ai-concepts--coverage-map).

## Project structure

```
anita/
├── chat_assistant_prompt.py          # Chat Assistant system prompt + traveller_profile schema
├── recommendation_engine_prompt.py   # Recommendation Engine judgment-layer system prompt
├── recommendation_engine.py          # 9 sub-engines + orchestration — Activity Planner is RAG-grounded
├── llm_client.py                     # LLMClient protocol: Anthropic + Mock implementations,
│                                      # forced tool-calling, prompt caching, cache stats
├── tracing.py                        # Observability: TraceStore records every LLM call
│                                      # (request/response/latency/tokens/errors)
├── orchestrator.py                   # Ties Chat Assistant ↔ Recommendation Engine together;
│                                      # write-protects UI-owned profile fields from the LLM
├── streamlit_app.py                  # UI: chat, map exploration, family form, recommendation
│                                      # display, cache stats + trace log panels
├── rag/                               # RAG retrieval: curated corpus + TF-IDF embeddings + vector search
│   ├── documents.py                   # Curated activity/destination knowledge base
│   ├── embeddings.py                  # TfidfEmbeddingBackend (default, local) + Voyage AI stub
│   ├── vector_store.py                # In-memory vector store, cosine similarity search
│   └── retriever.py                   # retrieve_activities() — curated + live Places combined
├── tests/
│   ├── test_orchestrator.py          # End-to-end orchestration test (MockLLMClient)
│   ├── test_prompt_caching.py        # Cache placement, stats tracking
│   ├── test_rag.py                    # Retrieval quality, hard filtering, fabrication protection
│   └── test_tracing.py                # Trace log recording, bounded size, wiring
├── evals/
│   ├── golden_conversations.py        # Scenarios, several encoding real bugs as regression tests
│   ├── run_evals.py                    # CLI runner (mock or live mode)
│   └── README.md
└── docs/
    ├── PROJECT_DOCUMENTATION.md       # Problem statement, competitive research, phases,
    │                                  # Gen AI coverage map, architecture, Mermaid diagrams
    ├── chat_assistant_role.md
    ├── recommendation_engine_role.md
    ├── ui_mockup.html                 # Visual design mockup (boarding-pass/departures-board direction)
    └── decisions/
        ├── 0001-traveller-profile-persistence.md
        └── 0002-no-agent-framework.md
```

## Design language

The UI follows a **boarding pass / departures board** visual direction — charcoal/graphite base, JetBrains Mono for status chips and data (split-flap board feel), IBM Plex Sans for chat text, perforated ticket-stub dividers as the signature element. See `docs/ui_mockup.html`.

## Running it

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Auto-detects mode from configured secrets — no manual flag needed:
- **No `ANTHROPIC_API_KEY` set** → runs a scripted `MockLLMClient` demo conversation, fully click-through-able, no key required.
- **`ANTHROPIC_API_KEY` set** (via `st.secrets` on Streamlit Cloud or an environment variable locally) → goes live, wrapped in a `ResilientLLMClient` that gracefully degrades to a safe response on any API failure instead of crashing.
- **`GOOGLE_PLACES_API_KEY`** (optional) → enables live destination geocoding and real POI data in the map exploration flow; falls back to a static centroid list without it.

## Testing

```bash
python tests/test_orchestrator.py       # end-to-end orchestration test
python tests/test_prompt_caching.py     # cache placement + stats tracking
python tests/test_rag.py                # retrieval quality + hard filtering + fabrication protection
python tests/test_tracing.py            # trace log recording, bounded size, wiring
```

## Evals

Golden conversation scenarios, several encoding real bugs caught during live testing as permanent regression tests — see `evals/README.md` for the full incident-to-scenario mapping.

```bash
python evals/run_evals.py               # mock mode, no API key needed
python evals/run_evals.py --client live # real model, needs ANTHROPIC_API_KEY
```

## Status

MVP built and demoed live. Post-MVP work added four Gen AI capabilities with dedicated test coverage: prompt caching, an eval framework, RAG-grounded activity recommendations, and observability/tracing. Two real fabrication bugs were caught during live testing and closed with structural fixes (not just prompt tweaks), each now a permanent regression test — see the [incident log](docs/PROJECT_DOCUMENTATION.md#9-incident-log--bugs-caught-live-and-their-permanent-fixes) for the full account. Flight, hotel, and restaurant sub-engines remain stubs pending live data source integration (see `TODO`s in `recommendation_engine.py`); persistent cross-session memory is designed for but explicitly deferred.
