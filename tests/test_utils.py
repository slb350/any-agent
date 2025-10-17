"""Tests for utils module"""
from types import SimpleNamespace

import pytest

from open_agent.utils import format_messages, create_client, ToolCallAggregator
from open_agent.types import AgentOptions, TextBlock, ToolUseBlock, ToolUseError


def test_format_messages_basic():
    """Test basic message formatting"""
    messages = format_messages("System", "User")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "System"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "User"


def test_format_messages_with_history():
    """Test message formatting with history"""
    history = [{"role": "user", "content": "Hi"}]
    messages = format_messages("System", "Again", history)
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hi"
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "Again"


def test_format_messages_empty_history():
    """Test message formatting with empty history"""
    messages = format_messages("System", "User", [])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_create_client():
    """Test client creation"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1",
        timeout=45.0,
        api_key="local-key"
    )
    client = create_client(options)
    assert str(client.base_url).rstrip('/') == "http://localhost:1234/v1"
    assert client.timeout == 45.0
    assert client.api_key == "local-key"


def test_tool_aggregator_basic():
    """Test tool call aggregator initialization"""
    aggregator = ToolCallAggregator()
    assert hasattr(aggregator, 'process_chunk')
    assert hasattr(aggregator, 'finalize_tools')
    assert len(aggregator.pending_tools) == 0


def test_tool_aggregator_no_content():
    """Test aggregator with chunk containing no content"""
    aggregator = ToolCallAggregator()

    # Create a mock chunk with no choices
    class MockChunk:
        choices = []

    result = aggregator.process_chunk(MockChunk())
    assert result is None


def test_tool_aggregator_text_content():
    """Test aggregator with text content"""
    aggregator = ToolCallAggregator()

    # Create a mock chunk with text content
    class MockDelta:
        content = "Hello world"
        tool_calls = None

    class MockChoice:
        delta = MockDelta()

    class MockChunk:
        choices = [MockChoice()]

    result = aggregator.process_chunk(MockChunk())
    assert isinstance(result, TextBlock)
    assert result.text == "Hello world"


def test_tool_aggregator_accumulative_streaming():
    """Providers that resend full text should only emit the delta."""
    aggregator = ToolCallAggregator()

    def chunk(text):
        delta = SimpleNamespace(content=text, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

    first = aggregator.process_chunk(chunk("Hello"))
    second = aggregator.process_chunk(chunk("Hello world"))

    assert isinstance(first, TextBlock)
    assert first.text == "Hello"
    assert isinstance(second, TextBlock)
    assert second.text == " world"


def test_tool_aggregator_process_tool_call_chunks():
    """Aggregator should combine streamed tool call fragments."""
    aggregator = ToolCallAggregator()

    def chunk(tool_id=None, name=None, arguments=None):
        function = SimpleNamespace(name=name, arguments=arguments)
        tool_call = SimpleNamespace(index=0, id=tool_id, function=function)
        delta = SimpleNamespace(content=None, tool_calls=[tool_call])
        return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

    # First chunk provides id, name, and partial arguments
    aggregator.process_chunk(chunk(tool_id="call_123", name="search", arguments='{"query": "new'))
    # Second chunk completes arguments
    aggregator.process_chunk(chunk(arguments='s"}'))

    results = aggregator.finalize_tools()
    assert len(results) == 1
    tool = results[0]
    assert isinstance(tool, ToolUseBlock)
    assert tool.id == "call_123"
    assert tool.name == "search"
    assert tool.input == {"query": "news"}


def test_tool_aggregator_finalize_empty():
    """Test finalizing with no pending tools"""
    aggregator = ToolCallAggregator()
    results = aggregator.finalize_tools()
    assert len(results) == 0


def test_tool_aggregator_finalize_missing_fields():
    """Test finalizing with missing required fields"""
    aggregator = ToolCallAggregator()

    # Manually add an incomplete tool
    aggregator.pending_tools[0] = {
        "id": None,
        "name": "test",
        "arguments_buffer": "{}"
    }

    results = aggregator.finalize_tools()
    assert len(results) == 1
    assert isinstance(results[0], ToolUseError)
    assert "missing required fields" in results[0].error.lower()


def test_tool_aggregator_finalize_invalid_json():
    """Test finalizing with invalid JSON arguments"""
    aggregator = ToolCallAggregator()

    # Manually add a tool with invalid JSON
    aggregator.pending_tools[0] = {
        "id": "call_123",
        "name": "test_tool",
        "arguments_buffer": "{invalid json"
    }

    results = aggregator.finalize_tools()
    assert len(results) == 1
    assert isinstance(results[0], ToolUseError)
    assert "Invalid JSON" in results[0].error


def test_tool_aggregator_finalize_valid_tool():
    """Test finalizing with a valid tool call"""
    aggregator = ToolCallAggregator()

    # Manually add a complete tool
    aggregator.pending_tools[0] = {
        "id": "call_abc",
        "name": "search",
        "arguments_buffer": '{"query": "test"}'
    }

    results = aggregator.finalize_tools()
    assert len(results) == 1
    assert isinstance(results[0], ToolUseBlock)
    assert results[0].id == "call_abc"
    assert results[0].name == "search"
    assert results[0].input == {"query": "test"}


def test_tool_aggregator_finalize_empty_args():
    """Test finalizing with empty arguments"""
    aggregator = ToolCallAggregator()

    # Add a tool with no arguments
    aggregator.pending_tools[0] = {
        "id": "call_def",
        "name": "ping",
        "arguments_buffer": ""
    }

    results = aggregator.finalize_tools()
    assert len(results) == 1
    assert isinstance(results[0], ToolUseBlock)
    assert results[0].input == {}


def test_tool_aggregator_clears_after_finalize():
    """Test that aggregator clears state after finalization"""
    aggregator = ToolCallAggregator()

    aggregator.pending_tools[0] = {
        "id": "call_xyz",
        "name": "test",
        "arguments_buffer": "{}"
    }

    results = aggregator.finalize_tools()
    assert len(results) == 1

    # State should be cleared
    assert len(aggregator.pending_tools) == 0

    # Second finalize should return empty
    results = aggregator.finalize_tools()
    assert len(results) == 0
