"""
Tests for tracing.py (observability) and its integration into
AnthropicLLMClient. Run with: python tests/test_tracing.py
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_tracestore_records_success():
    from tracing import TraceStore

    store = TraceStore()
    ctx = store.start("chat_structured", "system prompt text", "dynamic context", [
        {"role": "user", "content": "hello"},
    ])
    usage = MagicMock(cache_read_input_tokens=100, cache_creation_input_tokens=0, input_tokens=20, output_tokens=50)
    store.finish(ctx, response={"reply": "hi"}, usage=usage)

    traces = store.recent()
    assert len(traces) == 1
    assert traces[0].success is True
    assert traces[0].last_user_message == "hello"
    assert traces[0].response == {"reply": "hi"}
    assert traces[0].cache_read_tokens == 100
    print("PASS: TraceStore records a successful call correctly")


def test_tracestore_records_failure():
    from tracing import TraceStore

    store = TraceStore()
    ctx = store.start("chat_structured", "system", "", [{"role": "user", "content": "hi"}])
    store.fail(ctx, ValueError("simulated failure"))

    traces = store.recent()
    assert len(traces) == 1
    assert traces[0].success is False
    assert "simulated failure" in traces[0].error
    print("PASS: TraceStore records a failed call correctly")


def test_tracestore_bounded_size():
    """The store must not grow unbounded over a long session."""
    from tracing import TraceStore

    store = TraceStore(max_events=5)
    for i in range(10):
        ctx = store.start("chat_structured", "system", "", [{"role": "user", "content": f"msg {i}"}])
        usage = MagicMock(cache_read_input_tokens=0, cache_creation_input_tokens=0, input_tokens=1, output_tokens=1)
        store.finish(ctx, response={"reply": f"reply {i}"}, usage=usage)

    assert len(store.all()) == 5, "Store should be capped at max_events, dropping oldest"
    assert store.all()[0].last_user_message == "msg 5", "Oldest events (0-4) should have been dropped"
    print("PASS: TraceStore correctly bounds its size, dropping oldest events")


def test_summary_stats():
    from tracing import TraceStore

    store = TraceStore()
    for success in [True, True, False]:
        ctx = store.start("chat_structured", "system", "", [{"role": "user", "content": "x"}])
        if success:
            usage = MagicMock(cache_read_input_tokens=0, cache_creation_input_tokens=0, input_tokens=1, output_tokens=10)
            store.finish(ctx, response={}, usage=usage)
        else:
            store.fail(ctx, RuntimeError("fail"))

    summary = store.summary()
    assert summary["call_count"] == 3
    assert summary["error_count"] == 1
    assert summary["total_output_tokens"] == 20
    print("PASS: summary stats aggregate correctly across mixed success/failure calls")


def test_traces_wired_into_anthropic_client():
    """End-to-end: AnthropicLLMClient.chat_structured must populate
    .traces automatically, with no extra wiring needed by the caller."""
    mock_anthropic_module = MagicMock()
    mock_client_instance = MagicMock()
    mock_anthropic_module.Anthropic.return_value = mock_client_instance

    mock_usage = MagicMock(cache_read_input_tokens=500, cache_creation_input_tokens=0, input_tokens=10, output_tokens=30)
    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "record_turn"
    mock_tool_block.input = {"reply": "ok", "profile_updates": {}, "trigger_recommendation": False, "show_map_destination": ""}
    mock_response = MagicMock(content=[mock_tool_block], usage=mock_usage)
    mock_client_instance.messages.create.return_value = mock_response
    sys.modules["anthropic"] = mock_anthropic_module

    from llm_client import AnthropicLLMClient
    client = AnthropicLLMClient(api_key="test-key")
    client.chat_structured(system="SYS", messages=[{"role": "user", "content": "test message"}])

    assert client.traces.summary()["call_count"] == 1
    assert client.traces.recent()[0].cache_read_tokens == 500
    print("PASS: tracing is automatically populated by AnthropicLLMClient, no extra caller wiring needed")


def test_resilient_client_exposes_traces():
    """ResilientLLMClient must expose the primary's trace store, same
    pattern as .stats."""
    from llm_client import ResilientLLMClient, SafeFallbackClient, MockLLMClient

    class FakePrimaryWithTraces:
        def __init__(self):
            from tracing import TraceStore
            self.traces = TraceStore()

        def chat_structured(self, system, messages, dynamic_context=""):
            return {"reply": "ok", "profile_updates": {}, "trigger_recommendation": False, "show_map": None, "show_family_form": False}

    primary = FakePrimaryWithTraces()
    client = ResilientLLMClient(primary=primary, fallback=SafeFallbackClient())
    assert client.traces is primary.traces
    print("PASS: ResilientLLMClient correctly exposes the primary client's trace store")


if __name__ == "__main__":
    test_tracestore_records_success()
    test_tracestore_records_failure()
    test_tracestore_bounded_size()
    test_summary_stats()
    test_traces_wired_into_anthropic_client()
    test_resilient_client_exposes_traces()
    print("\nAll tracing tests passed.")
