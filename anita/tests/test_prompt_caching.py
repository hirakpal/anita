"""
Tests for prompt caching: verifies the static/dynamic system prompt split,
correct cache_control placement in the API request, and usage stats
tracking. Run with: python tests/test_prompt_caching.py
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_static_dynamic_separation():
    """The static system prompt must never change turn to turn; the
    dynamic context must reflect the current profile state each time."""
    from chat_assistant_prompt import CHAT_ASSISTANT_SYSTEM_PROMPT
    from orchestrator import ConversationState, _build_dynamic_profile_context

    state = ConversationState()
    dynamic_1 = _build_dynamic_profile_context(state.profile)

    state.profile["traveller_identity"]["name"] = "Test User"
    state.profile["trip"]["destination"]["confirmed"] = ["Testville"]
    dynamic_2 = _build_dynamic_profile_context(state.profile)

    assert isinstance(CHAT_ASSISTANT_SYSTEM_PROMPT, str)
    assert dynamic_1 != dynamic_2, "Dynamic context should change when profile changes"
    assert "Test User" in dynamic_2 and "Test User" not in dynamic_1
    print("PASS: static/dynamic separation works correctly")


def test_cache_control_placement():
    """The API request sent to Anthropic must place cache_control on the
    static block only, never on the dynamic block."""
    mock_anthropic_module = MagicMock()
    mock_client_instance = MagicMock()
    mock_anthropic_module.Anthropic.return_value = mock_client_instance

    mock_usage = MagicMock()
    mock_usage.cache_read_input_tokens = 15000
    mock_usage.cache_creation_input_tokens = 0
    mock_usage.input_tokens = 50
    mock_usage.output_tokens = 120

    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "record_turn"
    mock_tool_block.input = {
        "reply": "test", "profile_updates": {}, "trigger_recommendation": False, "show_map_destination": "",
    }
    mock_response = MagicMock()
    mock_response.content = [mock_tool_block]
    mock_response.usage = mock_usage
    mock_client_instance.messages.create.return_value = mock_response

    sys.modules["anthropic"] = mock_anthropic_module

    # Import after the mock is in place -- AnthropicLLMClient imports
    # anthropic lazily inside __init__, so this ordering works even if
    # llm_client was already imported earlier in the process.
    from llm_client import AnthropicLLMClient

    client = AnthropicLLMClient(api_key="test-key")
    client.chat_structured(
        system="STATIC", messages=[{"role": "user", "content": "hi"}], dynamic_context="DYNAMIC",
    )

    call_kwargs = mock_client_instance.messages.create.call_args.kwargs
    system_blocks = call_kwargs["system"]

    assert isinstance(system_blocks, list)
    assert system_blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in system_blocks[1]
    assert system_blocks[0]["text"] == "STATIC"
    assert system_blocks[1]["text"] == "DYNAMIC"
    print("PASS: cache_control correctly placed on static block only")


def test_cache_stats_tracking():
    """CacheStats should accumulate correctly across multiple calls."""
    from llm_client import CacheStats

    stats = CacheStats()

    usage_1 = MagicMock(cache_read_input_tokens=0, cache_creation_input_tokens=17000, input_tokens=50, output_tokens=100)
    stats.record(usage_1)
    assert stats.cache_creation_tokens == 17000
    assert stats.call_count == 1

    usage_2 = MagicMock(cache_read_input_tokens=17000, cache_creation_input_tokens=0, input_tokens=60, output_tokens=110)
    stats.record(usage_2)
    assert stats.cache_read_tokens == 17000
    assert stats.call_count == 2

    # Second call should show meaningful savings since it was a cache hit
    assert stats.estimated_savings_pct > 0
    print(f"PASS: cache stats tracked correctly (estimated savings: {stats.estimated_savings_pct}%)")


def test_no_dynamic_context_still_works():
    """chat_structured must still work with the default empty dynamic_context
    (backward compatible call site, e.g. tests or simple callers)."""
    mock_anthropic_module = MagicMock()
    mock_client_instance = MagicMock()
    mock_anthropic_module.Anthropic.return_value = mock_client_instance

    mock_usage = MagicMock(cache_read_input_tokens=0, cache_creation_input_tokens=0, input_tokens=10, output_tokens=10)
    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "record_turn"  # reserved kwarg in MagicMock() constructor -- must set after
    mock_tool_block.input = {
        "reply": "ok", "profile_updates": {}, "trigger_recommendation": False, "show_map_destination": "",
    }
    mock_response = MagicMock(content=[mock_tool_block], usage=mock_usage)
    mock_client_instance.messages.create.return_value = mock_response
    sys.modules["anthropic"] = mock_anthropic_module

    from llm_client import AnthropicLLMClient
    client = AnthropicLLMClient(api_key="test-key")
    result = client.chat_structured(system="STATIC", messages=[{"role": "user", "content": "hi"}])

    call_kwargs = mock_client_instance.messages.create.call_args.kwargs
    assert len(call_kwargs["system"]) == 1, "No second block should be added when dynamic_context is empty"
    assert result["reply"] == "ok"
    print("PASS: works correctly with no dynamic_context provided")


if __name__ == "__main__":
    test_static_dynamic_separation()
    test_cache_control_placement()
    test_cache_stats_tracking()
    test_no_dynamic_context_still_works()
    print("\nAll prompt caching tests passed.")
