"""
LLM Client abstraction.

Keeps the orchestrator decoupled from any specific LLM provider. The real
client talks to the Anthropic API and asks the model to return structured
JSON (reply + profile_updates + actions). The mock client drives the same
interface deterministically, so the orchestration logic can be tested
without live API calls or credentials.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol


class LLMClient(Protocol):
    def chat_structured(
        self,
        system: str,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        """
        Chat-Assistant-specific. Send a conversation to the model and
        return a structured turn:
          {
            "reply": str,                # natural-language text for the user
            "profile_updates": dict,     # partial traveller_profile to merge
            "trigger_recommendation": bool,
            "show_map": {"destination": str} | None,
          }
        This shape is fixed -- only the Chat Assistant orchestrator should
        call this. The Recommendation Engine's judgment-layer sub-engines
        use complete_json() instead, since their output shapes differ per
        sub-engine (scored lists, sequenced days, etc.), not the chat-turn
        schema above.
        """
        ...

    def complete_json(
        self,
        system: str,
        user_content: str,
    ) -> dict[str, Any]:
        """
        Generic structured-JSON call, no fixed response shape. Used by the
        Recommendation Engine's judgment-layer sub-engines (rank_destinations,
        plan_activities, build_itinerary, generate_packing_list), each of
        which asks for and parses its own JSON shape. `system` should be
        RECOMMENDATION_ENGINE_SYSTEM_PROMPT (see recommendation_engine_prompt.py)
        plus that sub-engine's specific output-shape instructions.
        """
        ...


TURN_RESPONSE_INSTRUCTIONS = """\
Respond with ONLY a single JSON object, no markdown fences, no preamble, \
matching this exact shape:

{
  "reply": "<natural language reply to show the user>",
  "profile_updates": { <partial traveller_profile fields to merge, only \
what changed this turn> },
  "trigger_recommendation": <true if the profile is now sufficient to run \
the Recommendation Engine, else false>,
  "show_map": { "destination": "<destination name>" } or null
}

profile_updates should use the same nested shape as TRAVELLER_PROFILE_SCHEMA. \
Only include fields you are setting or changing this turn -- omit anything \
unchanged. Never include commentary outside the JSON object.
"""


class AnthropicLLMClient:
    """Real client. Requires ANTHROPIC_API_KEY in the environment."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        import anthropic  # local import so the mock path has no hard dependency

        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def chat_structured(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        full_system = system + "\n\n" + TURN_RESPONSE_INSTRUCTIONS
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=full_system,
            messages=messages,
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return _safe_parse_turn(text)

    def complete_json(self, system: str, user_content: str) -> dict[str, Any]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return _safe_parse_json(text)


class MockLLMClient:
    """
    Deterministic test double. Takes a pre-scripted list of chat-turn
    responses (for chat_structured) and, separately, a scripted list of
    generic JSON responses (for complete_json). Ignores actual message
    content -- lets orchestration/wiring logic be tested without a real
    model.
    """

    def __init__(
        self,
        scripted_turns: list[dict[str, Any]],
        scripted_json_calls: list[dict[str, Any]] | None = None,
    ):
        self._turns = list(scripted_turns)
        self._turn_index = 0
        self._json_calls = list(scripted_json_calls or [])
        self._json_index = 0

    def chat_structured(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        if self._turn_index >= len(self._turns):
            raise IndexError("MockLLMClient: no more scripted chat turns available")
        turn = self._turns[self._turn_index]
        self._turn_index += 1
        return turn

    def complete_json(self, system: str, user_content: str) -> dict[str, Any]:
        if self._json_index >= len(self._json_calls):
            raise IndexError("MockLLMClient: no more scripted complete_json calls available")
        result = self._json_calls[self._json_index]
        self._json_index += 1
        return result


def _safe_parse_json(text: str) -> dict[str, Any]:
    """Parse arbitrary model JSON output, stripping stray markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}\nRaw text: {text!r}")


def _safe_parse_turn(text: str) -> dict[str, Any]:
    """Parse model output into the chat-turn shape, stripping stray fences."""
    parsed = _safe_parse_json(text)
    parsed.setdefault("reply", "")
    parsed.setdefault("profile_updates", {})
    parsed.setdefault("trigger_recommendation", False)
    parsed.setdefault("show_map", None)
    return parsed
