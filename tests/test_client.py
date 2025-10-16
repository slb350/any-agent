"""Tests for Client class"""
import pytest
from any_agent import Client, AgentOptions
from any_agent.types import TextBlock, ToolUseBlock


def test_client_initialization():
    """Test Client class initialization"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1",
        max_turns=10
    )

    client = Client(options)

    assert client.options == options
    assert client.turn_count == 0
    assert len(client.message_history) == 0
    assert client.response_stream is None
    assert client._aggregator is None


def test_client_context_manager():
    """Test Client can be used as async context manager"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )

    client = Client(options)

    # Verify it has context manager methods
    assert hasattr(client, '__aenter__')
    assert hasattr(client, '__aexit__')


def test_client_history_property():
    """Test history property returns a copy"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )

    client = Client(options)

    # Add a message to history
    client.message_history.append({"role": "user", "content": "test"})

    # Get history
    history = client.history

    # Verify it's a copy
    assert history == client.message_history
    assert history is not client.message_history


def test_client_turn_metadata():
    """Test turn metadata property"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1",
        max_turns=5
    )

    client = Client(options)

    metadata = client.turn_metadata

    assert metadata["turn_count"] == 0
    assert metadata["max_turns"] == 5


def test_client_format_history_entry_text_only():
    """Test formatting history entry with text only"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )

    client = Client(options)

    blocks = [
        TextBlock(text="Hello "),
        TextBlock(text="world")
    ]

    entry = client._format_history_entry(blocks)

    assert entry["role"] == "assistant"
    assert entry["content"] == "Hello world"
    assert "tool_calls" not in entry


def test_client_format_history_entry_with_tools():
    """Test formatting history entry with tool calls"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )

    client = Client(options)

    blocks = [
        TextBlock(text="Let me search for that."),
        ToolUseBlock(id="call_123", name="search", input={"query": "test"})
    ]

    entry = client._format_history_entry(blocks)

    assert entry["role"] == "assistant"
    assert entry["content"] == "Let me search for that."
    assert "tool_calls" in entry
    assert len(entry["tool_calls"]) == 1
    assert entry["tool_calls"][0]["id"] == "call_123"
    assert entry["tool_calls"][0]["function"]["name"] == "search"
    # Verify JSON encoding
    import json
    assert json.loads(entry["tool_calls"][0]["function"]["arguments"]) == {"query": "test"}


def test_client_format_history_entry_tools_only():
    """Test formatting history entry with only tool calls"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )

    client = Client(options)

    blocks = [
        ToolUseBlock(id="call_abc", name="ping", input={})
    ]

    entry = client._format_history_entry(blocks)

    assert entry["role"] == "assistant"
    assert entry["content"] is None
    assert "tool_calls" in entry
    assert len(entry["tool_calls"]) == 1


@pytest.mark.asyncio
async def test_client_receive_messages_without_query():
    """Test that receive_messages raises error if query not called first"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )

    client = Client(options)

    with pytest.raises(RuntimeError, match="Call query\\(\\) first"):
        async for _ in client.receive_messages():
            pass
