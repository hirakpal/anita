"""
Observability / tracing for LLM calls.

Records structured detail on every call made through AnthropicLLMClient:
timestamp, latency, token usage (incl. cache hit/miss), the actual
request content sent, the actual response received, and success/failure.

This is the piece that would have made tonight's actual debugging faster
-- several incidents (the fabricated family member names, the schema
that never reached the model, the incomplete restatement wiping form
data) required reconstructing what the model actually saw and returned
from screenshots of the profile debug panel. A trace log makes that
directly inspectable instead of reverse-engineered after the fact.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TraceEvent:
    id: str
    timestamp: float                    # unix epoch seconds
    call_type: str                      # "chat_structured" | "complete_json"
    latency_ms: float
    success: bool

    # Request detail. system/dynamic_context are capped to keep the trace
    # store from growing unbounded over a long session -- long enough to
    # actually debug with, not so long a session balloons in memory.
    system_prompt_preview: str
    dynamic_context_preview: str
    message_count: int
    last_user_message: str

    # Response detail -- kept in full, since a parsed turn/tool-call dict
    # is small and this is exactly the content most worth inspecting.
    response: Optional[dict] = None
    error: Optional[str] = None

    # Token usage
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "call_type": self.call_type,
            "latency_ms": round(self.latency_ms, 1),
            "success": self.success,
            "system_prompt_preview": self.system_prompt_preview,
            "dynamic_context_preview": self.dynamic_context_preview,
            "message_count": self.message_count,
            "last_user_message": self.last_user_message,
            "response": self.response,
            "error": self.error,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


_PREVIEW_CAP = 2000  # chars -- generous enough to actually debug with


class TraceStore:
    """
    In-memory trace log for a single client's calls. One store per
    client instance (see AnthropicLLMClient.traces), same pattern as
    CacheStats -- exposed through ResilientLLMClient via a property so
    the wrapping doesn't hide it from the UI layer.
    """

    def __init__(self, max_events: int = 200):
        self._events: list[TraceEvent] = []
        self.max_events = max_events

    def start(self, call_type: str, system: str, dynamic_context: str, messages: list[dict]) -> dict:
        """Call before making the API request; returns a context dict to
        pass to finish()/fail()."""
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                last_user = content if isinstance(content, str) else str(content)
                break
        return {
            "id": str(uuid.uuid4())[:8],
            "call_type": call_type,
            "start_time": time.time(),
            "system_prompt_preview": system[:_PREVIEW_CAP],
            "dynamic_context_preview": dynamic_context[:_PREVIEW_CAP] if dynamic_context else "",
            "message_count": len(messages),
            "last_user_message": last_user[:_PREVIEW_CAP],
        }

    def finish(self, ctx: dict, response: dict, usage: Any) -> None:
        elapsed_ms = (time.time() - ctx["start_time"]) * 1000
        event = TraceEvent(
            id=ctx["id"],
            timestamp=ctx["start_time"],
            call_type=ctx["call_type"],
            latency_ms=elapsed_ms,
            success=True,
            system_prompt_preview=ctx["system_prompt_preview"],
            dynamic_context_preview=ctx["dynamic_context_preview"],
            message_count=ctx["message_count"],
            last_user_message=ctx["last_user_message"],
            response=response,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )
        self._add(event)

    def fail(self, ctx: dict, error: Exception) -> None:
        elapsed_ms = (time.time() - ctx["start_time"]) * 1000
        event = TraceEvent(
            id=ctx["id"],
            timestamp=ctx["start_time"],
            call_type=ctx["call_type"],
            latency_ms=elapsed_ms,
            success=False,
            system_prompt_preview=ctx["system_prompt_preview"],
            dynamic_context_preview=ctx["dynamic_context_preview"],
            message_count=ctx["message_count"],
            last_user_message=ctx["last_user_message"],
            error=str(error),
        )
        self._add(event)

    def _add(self, event: TraceEvent) -> None:
        self._events.append(event)
        if len(self._events) > self.max_events:
            self._events.pop(0)  # drop oldest, keep the store bounded

    def recent(self, n: int = 20) -> list[TraceEvent]:
        return list(reversed(self._events[-n:]))

    def all(self) -> list[TraceEvent]:
        return list(self._events)

    def clear(self) -> None:
        self._events.clear()

    def summary(self) -> dict:
        if not self._events:
            return {"call_count": 0}
        latencies = [e.latency_ms for e in self._events]
        errors = [e for e in self._events if not e.success]
        return {
            "call_count": len(self._events),
            "error_count": len(errors),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "max_latency_ms": round(max(latencies), 1),
            "total_output_tokens": sum(e.output_tokens for e in self._events),
        }

    def to_json(self) -> str:
        """Export the full trace log as JSON, e.g. for a 'download traces' button."""
        return json.dumps([e.to_dict() for e in self._events], indent=2)
