"""Tests for Client class"""
from types import SimpleNamespace
from typing import Any

import pytest

from open_agent import Client, AgentOptions
from open_agent.types import TextBlock, ToolUseBlock


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

    with pytest.raises(RuntimeError, match="Call query\\(\\) or _continue_turn\\(\\) first"):
        async for _ in client.receive_messages():
            pass


@pytest.mark.asyncio
async def test_client_add_tool_result_with_string():
    """Tool results should be appended as tool messages"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )

    client = Client(options)
    client.message_history.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_123",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"}
            }
        ]
    })

    await client.add_tool_result("call_123", "result text")

    tool_message = client.message_history[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_123"
    assert tool_message["content"] == "result text"


@pytest.mark.asyncio
async def test_client_add_tool_result_with_dict():
    """Dict content should be JSON-encoded"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )

    client = Client(options)
    client.message_history.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"}
            }
        ]
    })

    await client.add_tool_result("call_abc", {"foo": "bar"})

    tool_message = client.message_history[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_abc"

    import json
    assert json.loads(tool_message["content"]) == {"foo": "bar"}


@pytest.mark.asyncio
async def test_client_add_tool_result_unknown_id():
    """Unknown tool_call_id should raise ValueError"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )

    client = Client(options)

    with pytest.raises(ValueError, match="Unknown tool_call_id"):
        await client.add_tool_result("missing", "data")


@pytest.mark.asyncio
async def test_client_add_tool_result_invalid_content_type():
    """Unsupported content types should raise TypeError"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )

    client = Client(options)
    client.message_history.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"}
            }
        ]
    })

    with pytest.raises(TypeError, match="content must be a str, dict, or list"):
        await client.add_tool_result("call_xyz", 123)


@pytest.mark.asyncio
async def test_client_add_tool_result_with_list_and_name():
    """List content should be JSON encoded and name propagated."""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )

    client = Client(options)
    client.message_history.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_list",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"}
            }
        ]
    })

    await client.add_tool_result("call_list", ["a", "b"], name="lookup")

    tool_message = client.message_history[-1]
    assert tool_message["name"] == "lookup"
    import json
    assert json.loads(tool_message["content"]) == ["a", "b"]


@pytest.mark.asyncio
async def test_client_query_failure_does_not_mutate_history(monkeypatch):
    """query() should rollback state if the transport raises."""
    class FailingMockClient:
        def __init__(self):
            self.closed = False
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        async def _create(self, **_: Any):
            raise RuntimeError("boom")

        async def close(self):
            self.closed = True

    failing_client = FailingMockClient()
    monkeypatch.setattr("open_agent.client.create_client", lambda _options: failing_client)

    options = AgentOptions(
        system_prompt="System",
        model="model",
        base_url="http://localhost:1234/v1"
    )

    client = Client(options)

    with pytest.raises(RuntimeError, match="boom"):
        await client.query("hello")

    assert client.message_history == []
    assert client.response_stream is None
    assert client._aggregator is None


@pytest.mark.asyncio
async def test_client_context_manager_closes_underlying_client(monkeypatch):
    """__aexit__ should close the created client to release transports."""
    class ClosingMockClient:
        def __init__(self):
            self.closed = False
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        async def _create(self, **_: Any):
            return []

        async def close(self):
            self.closed = True

    closing_client = ClosingMockClient()
    monkeypatch.setattr("open_agent.client.create_client", lambda _options: closing_client)

    options = AgentOptions(
        system_prompt="System",
        model="model",
        base_url="http://localhost:1234/v1"
    )

    async with Client(options):
        pass

    assert closing_client.closed
