# Decision: no agent framework (LangGraph / CrewAI / LangChain)

**Date:** 2026-07-08
**Status:** Decided, in effect for the current build; reversible

## Context

ANITA's orchestration (`orchestrator.py`) is, conceptually, a hand-rolled
state machine: the Chat Assistant elicits and infers a traveller profile
turn by turn, hands off to the Recommendation Engine once sufficient, and
two client-side UI actions (map lock-in, family form save) trigger
synthetic follow-up turns to keep the conversation moving. This is
exactly the shape of problem LangGraph exists to formalize as a graph of
nodes and conditional edges.

None of LangGraph, CrewAI, or LangChain are used anywhere in this
codebase. All API calls go directly through the Anthropic Python SDK,
wrapped in a small custom `LLMClient` protocol (`llm_client.py`).

This mirrors a decision made on an earlier related project (Horizon
Travel AI), where LangGraph and CrewAI were also explicitly declined in
favor of a custom Root Orchestrator pattern, documented there as
reversible. Same call made again here, for the same underlying reasons,
independently arrived at for this codebase's specific needs.

## What each framework actually offers

- **LangChain** — general-purpose LLM app toolkit: prompt chaining,
  memory abstractions, document loaders, retriever/tool integrations,
  provider-agnostic model wrappers.
- **LangGraph** — stateful multi-step agent workflows modeled as a graph
  (nodes = steps/agents, edges = transitions), with support for cycles
  (retry, loop back, wait for human input, continue). Built by the
  LangChain team but usable independently.
- **CrewAI** — higher-level, opinionated "team of agents" abstraction:
  agents defined by role/goal/backstory, assigned tasks, coordinated
  sequentially or hierarchically toward a shared objective.

All three are legitimate, widely-used tools. None were ruled out for
being bad — the decision below is about this project's specific needs
right now, not a general judgment against agent frameworks.

## Decision

**Build orchestration directly against the Anthropic SDK, no framework.**

## Why

- **Full control over provider-specific features.** `cache_control`
  prompt caching and forced `tool_choice` tool-calling are Anthropic
  API mechanics this project depends on directly (see
  `docs/PROJECT_DOCUMENTATION.md` §5.2.1 and §4.6). Framework
  abstraction layers often generalize these away or lag behind
  provider-specific features — going direct avoided any risk of losing
  access to exactly the caching/tool-calling behavior the build relies on.
- **Debuggability during incident response.** Both fabrication incidents
  caught live tonight (family member names, a "locked" location the user
  never selected) were root-caused by reading exact API request/response
  content and tracing state mutation through `orchestrator.py` directly.
  That's meaningfully harder through a framework's abstraction layers
  than through a ~150-line hand-written orchestrator.
- **Project scale doesn't justify it yet.** LangGraph/CrewAI earn their
  weight when many agents need to coordinate dynamically with complex
  branching. ANITA currently has two agents (Chat Assistant,
  Recommendation Engine) with a simple linear handoff, and the
  Recommendation Engine's 9 sub-engines don't currently coordinate with
  each other — they're independent functions called in sequence by
  `run_recommendation_engine()`.
- **Precedent.** The same call was made on the predecessor project
  (Horizon) for materially the same reasons.

## When this should be revisited

This is a **reversible** decision, not a permanent architectural stance.
Concrete triggers that would justify reopening it:

1. The Recommendation Engine's sub-engines start needing to coordinate
   dynamically (e.g. Itinerary Builder needing to renegotiate with
   Budget Optimizer mid-generation, rather than running once in a fixed
   sequence) — that's a real graph, and LangGraph would likely be less
   code than hand-rolling conditional branching for it.
2. A second LLM provider gets added (e.g. the Gemini exploration noted
   as a Phase 3 item) — at that point a provider-agnostic abstraction
   layer starts pulling its weight, since maintaining two hand-written
   provider clients has real duplication cost.
3. The number of distinct agents grows past what a human can hold in
   their head as "just read the orchestrator file" — CrewAI's
   role/goal/task structure becomes a real documentation and
   maintainability aid past a certain agent count, not just ceremony.

None of these are true today. If/when one becomes true, the migration
path is straightforward: `LLMClient` is already a clean protocol
boundary, and `orchestrator.py`'s state transitions are already
explicit enough to translate into graph nodes/edges without a redesign
of the underlying data model.
