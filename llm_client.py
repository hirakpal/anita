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


CHAT_TURN_TOOL = {
    "name": "record_turn",
    "description": (
        "Record this conversational turn: your reply to the user, any "
        "traveller_profile fields you learned or inferred this turn, "
        "whether the profile is now sufficient to run the Recommendation "
        "Engine, and whether to trigger the map exploration flow."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reply": {
                "type": "string",
                "description": "Natural-language reply to show the user. Never mention this tool, JSON, or internal fields.",
            },
            "profile_updates": {
                "type": "object",
                "description": (
                    "Partial traveller_profile update, using the same nested "
                    "shape as TRAVELLER_PROFILE_SCHEMA. Only include fields "
                    "learned or changed this turn -- omit anything unchanged. "
                    "Empty object if nothing new this turn."
                ),
            },
            "trigger_recommendation": {
                "type": "boolean",
                "description": "True only if the profile is now sufficient to run the Recommendation Engine.",
            },
            "show_map_destination": {
                "type": "string",
                "description": (
                    "Destination name to show the map exploration flow for, "
                    "if this turn should trigger it. Empty string if not."
                ),
            },
            "show_family_form": {
                "type": "boolean",
                "description": (
                    "True if this turn should offer the user a short inline "
                    "form to add names/ages/relations of family or group "
                    "members traveling with them. Only offer once, and only "
                    "when traveller_composition indicates more than one "
                    "traveler and members haven't been captured yet."
                ),
            },
        },
        "required": [
            "reply", "profile_updates", "trigger_recommendation",
            "show_map_destination", "show_family_form",
        ],
    },
}


class AnthropicLLMClient:
    """Real client. Requires ANTHROPIC_API_KEY in the environment."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        import anthropic  # local import so the mock path has no hard dependency

        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def chat_structured(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        # Forced tool-use, not prompt-only JSON instructions: the model
        # MUST call record_turn and its input arrives already parsed by
        # the SDK -- this is materially more reliable than asking a model
        # to format free text as JSON, which it can (and does, sometimes)
        # ignore in favor of a natural conversational reply.
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system,
            messages=messages,
            tools=[CHAT_TURN_TOOL],
            tool_choice={"type": "tool", "name": "record_turn"},
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "record_turn":
                return _normalize_turn(block.input)
        # Should be unreachable with forced tool_choice, but never trust
        # that fully -- fall through to a safe empty turn rather than
        # raising, so ResilientLLMClient's fallback path (which expects
        # exceptions, not None) still isn't the only safety net.
        raise ValueError("Model did not return a record_turn tool call despite forced tool_choice.")

    def complete_json(self, system: str, user_content: str) -> dict[str, Any]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return _safe_parse_json(text)


class ResilientLLMClient:
    """
    Wraps a primary client (e.g. AnthropicLLMClient) with a fallback client
    (e.g. MockLLMClient) used if the primary raises for any reason --
    network error, rate limit, bad key, malformed JSON response. Exists
    specifically for live demos: a transient API failure degrades to a
    scripted response instead of crashing the app mid-conversation.

    `on_fallback`, if provided, is called with the exception whenever a
    fallback happens -- wire this to a UI warning (e.g. st.warning) so the
    failure is visible rather than silent.
    """

    def __init__(self, primary: "LLMClient", fallback: "LLMClient", on_fallback=None):
        self.primary = primary
        self.fallback = fallback
        self.on_fallback = on_fallback

    def chat_structured(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        try:
            return self.primary.chat_structured(system, messages)
        except Exception as e:
            if self.on_fallback:
                self.on_fallback(e)
            return self.fallback.chat_structured(system, messages)

    def complete_json(self, system: str, user_content: str) -> dict[str, Any]:
        try:
            return self.primary.complete_json(system, user_content)
        except Exception as e:
            if self.on_fallback:
                self.on_fallback(e)
            return self.fallback.complete_json(system, user_content)


class SafeFallbackClient:
    """
    Purpose-built fallback for ResilientLLMClient -- NOT the scripted demo
    mock. Always returns a graceful, turn-agnostic response rather than a
    scripted one, since a live failure could happen on any turn and a
    scripted response tied to turn-index would desync and return the wrong
    content. No profile_updates, no crash -- just asks the user to retry.
    """

    def chat_structured(self, system: str, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "reply": "Sorry, I had trouble processing that just now — could you try rephrasing or sending that again?",
            "profile_updates": {},
            "trigger_recommendation": False,
            "show_map": None,
            "show_family_form": False,
        }

    def complete_json(self, system: str, user_content: str) -> dict[str, Any]:
        return {}


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


def _normalize_turn(tool_input: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a record_turn tool call's input (which uses the flat
    show_map_destination string, since nullable object schemas are less
    reliable across tool-use implementations) into the standard turn shape
    the rest of the app expects (show_map: {"destination": str} | None).
    """
    destination = tool_input.get("show_map_destination", "") or ""
    return {
        "reply": tool_input.get("reply", ""),
        "profile_updates": tool_input.get("profile_updates", {}) or {},
        "trigger_recommendation": bool(tool_input.get("trigger_recommendation", False)),
        "show_map": {"destination": destination} if destination.strip() else None,
        "show_family_form": bool(tool_input.get("show_family_form", False)),
    }


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
