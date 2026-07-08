# Eval Framework

Golden conversation scenarios that test actual behavior, not just code
correctness. Several scenarios encode real bugs caught during live
testing — turning production incidents into permanent regression tests.

## Running

```bash
# Mock mode: deterministic harness test, no API key needed, always the
# same result. Tests the merge logic and structural protections.
python evals/run_evals.py

# Live mode: real model behavior, needs ANTHROPIC_API_KEY. Non-deterministic
# by nature (the model phrases things differently turn to turn) -- checks
# test outcomes (did the right field get set), not exact wording.
python evals/run_evals.py --client live

# Run a single scenario
python evals/run_evals.py --scenario no_fabrication_of_family_names
```

## What's covered

| Scenario | What it protects against |
|---|---|
| `basic_profile_extraction` | Baseline: destination/dates/composition/budget merge correctly across turns |
| `no_fabrication_of_family_names` | **Real incident.** Model invented specific names/ages ("Rumi", "Anita", "Rahul") for a user's family that were never provided. Structural fix (members is write-once-via-form-only) is what this test protects. |
| `member_data_survives_followup_restatement` | **Real incident.** Form-entered age/senior_citizen data was silently wiped when the model's next turn restated members incompletely. |
| `senior_citizen_auto_derivation` | **Real incident.** Stale Streamlit checkbox widget state meant age 60+ didn't always auto-flag senior_citizen. |
| `map_lockin_auto_continues` | **Real incident.** Recommendation Engine never triggered after map lock-in because nothing prompted the LLM to evaluate `trigger_recommendation`. |

## Adding a new scenario

Add a `GoldenConversation` to `golden_conversations.py`:

```python
SCENARIO_MY_NEW_CASE = GoldenConversation(
    name="my_new_case",
    description="What this protects against and why it matters.",
    turns=[
        GoldenTurn(
            user_message="what the user actually typed",
            mock_response={...},  # matches the chat-turn schema
        ),
    ],
    checks=[
        Check("description of what should be true", lambda p, r, s: ...),
    ],
)
```

Add it to `ALL_SCENARIOS` at the bottom of the file.

**When you catch a new bug during testing:** write the eval scenario
*before* fixing the code, confirm it fails (proving it actually detects
the bug), then fix the code and confirm it passes. This is exactly how
the five scenarios above were built.

## Design notes

- `setup` hooks simulate client-side UI actions (form submissions, map
  lock-in) that don't go through the LLM at all — matching the real
  architecture where these are deliberately not LLM round-trips.
- Checks receive `(final_profile, list_of_replies, final_conversation_state)`
  so they can assert on profile fields, reply text, or state flags like
  `state.recommendation`.
- Mock mode scenarios can deliberately script a *buggy* response (see
  `no_fabrication_of_family_names`, which scripts the model fabricating
  names) to test that the harness-level protection catches it — the
  point isn't "does the model behave," it's "does the system hold even
  if the model doesn't."
