# Decision: traveller_profile persistence (deferred to next phase)

**Date:** 2026-07-08 (night before MVP demo)
**Status:** Deferred, not implemented in MVP

## Context

The traveller_profile schema was designed with a two-tier split from the
start (see `docs/chat_assistant_role.md`):
- **Tier 1 — `traveller_identity`**: name, age, nationality, home airport,
  loyalty programs, traveller-type flags. Intended to persist *across*
  trips for a returning user.
- **Tier 2 — everything else**: trip objective, dates, composition,
  budget, stay_location, etc. Intended to be fresh *per trip*.

Currently, none of this actually persists. Every session starts from a
fresh `ConversationState()` in Streamlit's in-memory `session_state`. The
two-tier design exists in the schema shape but delivers no real benefit
yet — a returning user re-answers their name, home airport, and
everything else from scratch every session.

## Decision

**Not implementing persistence for the MVP demo.** Ship with in-memory
`st.session_state` as-is.

## Why

- No user identity/auth exists yet -- nothing to key DB records by.
- Every fix validated tonight (schema-state injection, auto-continue
  after client-side actions, anti-fabrication guardrail) was tested
  against the in-memory path. Swapping storage now would mean
  re-validating all of it against a new unknown, hours before a demo.
- Zero demo-visible upside: the demo is a single sitting in one browser
  tab. Cross-session persistence isn't observable in that context.
- New failure surface (connection setup, schema creation, migration
  bugs) with no time budget to debug it.

## Path for next phase

The two-tier schema split was deliberately designed to map cleanly onto
two tables, so this should be a low-risk migration later rather than a
redesign:

- `traveller_identity` table, keyed by `user_id`, persists indefinitely.
- `trip_profile` table, keyed by `trip_id`, one-to-many per user, holds
  everything else (trip objective, composition, budget, stay_location,
  etc.).

Recommended sequence when this is picked up:
1. Add minimal auth/user identity (even a simple persistent user_id
   cookie would unlock most of the value without full auth).
2. SQLite to start (zero-ops, fine for a single-deployment MVP);
   Postgres later if this grows beyond one deployment.
3. `new_traveller_profile()` in `chat_assistant_prompt.py` becomes "load
   existing identity + fresh trip tier" instead of "always fresh."
4. Re-run the full test suite (`tests/test_orchestrator.py` and any new
   persistence tests) before considering it done -- the state-injection
   fix in `orchestrator.py` assumes `state.profile` is the single source
   of truth in memory; a DB-backed version needs to preserve that
   invariant (load once per turn, don't re-query mid-turn).
