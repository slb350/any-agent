"""Tests for automatic tool execution feature"""
import json
from types import SimpleNamespace
from typing import Any

import pytest

from open_agent import Client, AgentOptions, tool
from open_agent.types import TextBlock, ToolUseBlock, ToolUseError
from open_agent.hooks import HookDecision, HOOK_PRE_TOOL_USE, HOOK_POST_TOOL_USE


# --- Test: Tool Registry Validation ---

def test_tool_registry_duplicate_names():
    """Tool registry should reject duplicate tool names"""
    @tool("duplicate", "First tool", {})
    async def tool1(args):
        return {"result": "one"}

    @tool("duplicate", "Second tool", {})
    async def tool2(args):
        return {"result": "two"}

    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1",
        tools=[tool1, tool2],
        auto_execute_tools=True
    )

    with pytest.raises(ValueError, match="Duplicate tool name 'duplicate'"):
        Client(options)


def test_tool_registry_builds_correctly():
    """Tool registry should build correctly with unique names"""
    @tool("tool_a", "First tool", {})
    async def tool_a_fn(args):
        return {"result": "a"}

    @tool("tool_b", "Second tool", {})
    async def tool_b_fn(args):
        return {"result": "b"}

    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1",
        tools=[tool_a_fn, tool_b_fn],
        auto_execute_tools=True
    )

    client = Client(options)

    assert len(client._tool_registry) == 2
    assert "tool_a" in client._tool_registry
    assert "tool_b" in client._tool_registry
    assert client._tool_registry["tool_a"] == tool_a_fn
    assert client._tool_registry["tool_b"] == tool_b_fn


# --- Test: Auto-Execution Basic Flow ---

@pytest.mark.asyncio
async def test_auto_execution_basic_flow(monkeypatch):
    """Auto-execution should execute tools and continue conversation"""

    # Define test tool
    @tool("add", "Add two numbers", {"a": int, "b": int})
    async def add(args):
        return {"result": args["a"] + args["b"]}

    # Track tool execution
    tool_executions = []
    original_execute = add.execute
    async def tracked_execute(args):
        tool_executions.append(args)
        return await original_execute(args)
    add.execute = tracked_execute

    # Mock responses: tool call, then text-only response
    call_count = [0]

    class MockResponse:
        def __init__(self, chunks):
            self.chunks = chunks

        def __aiter__(self):
            # Return async iterator
            async def _gen():
                for chunk in self.chunks:
                    yield chunk
            return _gen()

    async def mock_create(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: return tool use
            return MockResponse([
                SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(
                        tool_calls=[SimpleNamespace(
                            index=0,
                            id="call_1",
                            function=SimpleNamespace(name="add", arguments='{"a": 5, "b": 3}')
                        )]
                    ),
                    finish_reason=None
                )])
            ])
        else:
            # Second call: return text response
            return MockResponse([
                SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="The result is 8"),
                    finish_reason=None
                )])
            ])

    class MockClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=mock_create)
            )
        async def close(self):
            pass

    monkeypatch.setattr("open_agent.client.create_client", lambda _: MockClient())

    options = AgentOptions(
        system_prompt="You are a calculator",
        model="test-model",
        base_url="http://localhost:1234/v1",
        tools=[add],
        auto_execute_tools=True
    )

    async with Client(options) as client:
        await client.query("What is 5 + 3?")

        blocks = []
        async for block in client.receive_messages():
            blocks.append(block)

        # Should have: ToolUseBlock, then TextBlock
        assert len(blocks) == 2
        assert isinstance(blocks[0], ToolUseBlock)
        assert blocks[0].name == "add"
        assert isinstance(blocks[1], TextBlock)
        assert "8" in blocks[1].text

        # Verify tool was executed
        assert len(tool_executions) == 1
        assert tool_executions[0] == {"a": 5, "b": 3}


# --- Test: Hook Integration ---

@pytest.mark.asyncio
async def test_auto_execution_pretooluse_hook_blocks_execution(monkeypatch):
    """PreToolUse hook should be able to block tool execution in auto mode"""

    @tool("blocked_tool", "This tool will be blocked", {})
    async def blocked_tool(args):
        pytest.fail("Tool should not execute - hook blocked it")
        return {}

    # Hook that blocks execution
    async def blocking_hook(event):
        if event.tool_name == "blocked_tool":
            return HookDecision(continue_=False, reason="Blocked by policy")
        return None

    # Mock response: tool call only
    class MockResponse:
        def __init__(self, chunks):
            self.chunks = chunks
        def __aiter__(self):
            async def _gen():
                for chunk in self.chunks:
                    yield chunk
            return _gen()

    async def mock_create(**kwargs):
        return MockResponse([
            SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(
                    tool_calls=[SimpleNamespace(
                        index=0,
                        id="call_1",
                        function=SimpleNamespace(name="blocked_tool", arguments='{}')
                    )]
                ),
                finish_reason=None
            )])
        ])

    class MockClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=mock_create)
            )
        async def close(self):
            pass

    monkeypatch.setattr("open_agent.client.create_client", lambda _: MockClient())

    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1",
        tools=[blocked_tool],
        auto_execute_tools=True,
        hooks={HOOK_PRE_TOOL_USE: [blocking_hook]}
    )

    async with Client(options) as client:
        await client.query("Test")

        blocks = []
        async for block in client.receive_messages():
            blocks.append(block)

        # Should receive ToolUseError instead of ToolUseBlock
        assert len(blocks) == 1
        assert isinstance(blocks[0], ToolUseError)
        assert "Blocked by policy" in blocks[0].error


@pytest.mark.asyncio
async def test_auto_execution_posttooluse_hook_observes_execution(monkeypatch):
    """PostToolUse hook should fire after tool execution in auto mode"""

    @tool("observed_tool", "Tool with observation", {})
    async def observed_tool(args):
        return {"result": "success"}

    # Track PostToolUse hook calls
    post_tool_calls = []

    async def observing_hook(event):
        post_tool_calls.append({
            "tool_name": event.tool_name,
            "result": event.tool_result
        })
        return HookDecision(continue_=True, reason="Observed")

    # Mock responses
    call_count = [0]

    class MockResponse:
        def __init__(self, chunks):
            self.chunks = chunks
        def __aiter__(self):
            async def _gen():
                for chunk in self.chunks:
                    yield chunk
            return _gen()

    async def mock_create(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return MockResponse([
                SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(
                        tool_calls=[SimpleNamespace(
                            index=0,
                            id="call_1",
                            function=SimpleNamespace(name="observed_tool", arguments='{}')
                        )]
                    ),
                    finish_reason=None
                )])
            ])
        else:
            return MockResponse([
                SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="Done"),
                    finish_reason=None
                )])
            ])

    class MockClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=mock_create)
            )
        async def close(self):
            pass

    monkeypatch.setattr("open_agent.client.create_client", lambda _: MockClient())

    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1",
        tools=[observed_tool],
        auto_execute_tools=True,
        hooks={HOOK_POST_TOOL_USE: [observing_hook]}
    )

    async with Client(options) as client:
        await client.query("Test")

        blocks = []
        async for block in client.receive_messages():
            blocks.append(block)

        # Verify hook was called
        assert len(post_tool_calls) == 1
        assert post_tool_calls[0]["tool_name"] == "observed_tool"
        assert post_tool_calls[0]["result"] == {"result": "success"}


# --- Test: Max Iterations Limit ---

@pytest.mark.asyncio
async def test_auto_execution_max_iterations_limit(monkeypatch):
    """Auto-execution should respect max_tool_iterations limit"""

    @tool("infinite_tool", "Tool that always returns another tool call", {})
    async def infinite_tool(args):
        return {"result": "iteration"}

    # Mock: always return tool calls (would loop forever without limit)
    class MockResponse:
        def __init__(self, chunks):
            self.chunks = chunks
        def __aiter__(self):
            async def _gen():
                for chunk in self.chunks:
                    yield chunk
            return _gen()

    async def mock_create(**kwargs):
        return MockResponse([
            SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(
                    tool_calls=[SimpleNamespace(
                        index=0,
                        id=f"call_{id(kwargs)}",
                        function=SimpleNamespace(name="infinite_tool", arguments='{}')
                    )]
                ),
                finish_reason=None
            )])
        ])

    class MockClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=mock_create)
            )
        async def close(self):
            pass

    monkeypatch.setattr("open_agent.client.create_client", lambda _: MockClient())

    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1",
        tools=[infinite_tool],
        auto_execute_tools=True,
        max_tool_iterations=3  # Should stop after 3 iterations
    )

    async with Client(options) as client:
        await client.query("Test")

        tool_blocks = []
        async for block in client.receive_messages():
            if isinstance(block, ToolUseBlock):
                tool_blocks.append(block)

        # Should have exactly max_tool_iterations + 1 tool calls
        # (3 iterations in loop + 1 final response when limit hit)
        assert len(tool_blocks) == 4


# --- Test: Error Handling ---

@pytest.mark.asyncio
async def test_auto_execution_unknown_tool_error(monkeypatch):
    """Unknown tool should add error to history with tool name"""

    # Mock response with unknown tool, then text response
    call_count = [0]

    class MockResponse:
        def __init__(self, chunks):
            self.chunks = chunks
        def __aiter__(self):
            async def _gen():
                for chunk in self.chunks:
                    yield chunk
            return _gen()

    async def mock_create(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call: unknown tool
            return MockResponse([
                SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(
                        tool_calls=[SimpleNamespace(
                            index=0,
                            id="call_1",
                            function=SimpleNamespace(name="unknown_tool", arguments='{}')
                        )]
                    ),
                    finish_reason=None
                )])
            ])
        else:
            # Second call: text response (after error added)
            return MockResponse([
                SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="I encountered an error"),
                    finish_reason=None
                )])
            ])

    class MockClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=mock_create)
            )
        async def close(self):
            pass

    monkeypatch.setattr("open_agent.client.create_client", lambda _: MockClient())

    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1",
        tools=[],  # No tools registered
        auto_execute_tools=True
    )

    async with Client(options) as client:
        await client.query("Test")

        # Should not raise error, should add error to history
        blocks = []
        async for block in client.receive_messages():
            blocks.append(block)

        # Check that error was added to history
        tool_messages = [m for m in client.message_history if m.get("role") == "tool"]
        assert len(tool_messages) == 1

        error_content = json.loads(tool_messages[0]["content"])
        assert "error" in error_content
        assert "tool" in error_content
        assert error_content["tool"] == "unknown_tool"


@pytest.mark.asyncio
async def test_auto_execution_tool_execution_error(monkeypatch):
    """Tool execution failure should add error to history with tool name"""

    @tool("failing_tool", "Tool that raises exception", {})
    async def failing_tool(args):
        raise ValueError("Tool execution failed")

    # Mock responses
    call_count = [0]

    class MockResponse:
        def __init__(self, chunks):
            self.chunks = chunks
        def __aiter__(self):
            async def _gen():
                for chunk in self.chunks:
                    yield chunk
            return _gen()

    async def mock_create(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return MockResponse([
                SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(
                        tool_calls=[SimpleNamespace(
                            index=0,
                            id="call_1",
                            function=SimpleNamespace(name="failing_tool", arguments='{}')
                        )]
                    ),
                    finish_reason=None
                )])
            ])
        else:
            return MockResponse([
                SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="Error handled"),
                    finish_reason=None
                )])
            ])

    class MockClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=mock_create)
            )
        async def close(self):
            pass

    monkeypatch.setattr("open_agent.client.create_client", lambda _: MockClient())

    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1",
        tools=[failing_tool],
        auto_execute_tools=True
    )

    async with Client(options) as client:
        await client.query("Test")

        blocks = []
        async for block in client.receive_messages():
            blocks.append(block)

        # Check that error was added to history
        tool_messages = [m for m in client.message_history if m.get("role") == "tool"]
        assert len(tool_messages) == 1

        error_content = json.loads(tool_messages[0]["content"])
        assert "error" in error_content
        assert "tool" in error_content
        assert error_content["tool"] == "failing_tool"
        assert "Tool execution failed" in error_content["error"]


# --- Test: Backward Compatibility ---

@pytest.mark.asyncio
async def test_backward_compatibility_manual_mode_default():
    """auto_execute_tools should default to False for backward compatibility"""

    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )

    assert options.auto_execute_tools is False


@pytest.mark.asyncio
async def test_manual_mode_still_works(monkeypatch):
    """Manual mode should work unchanged when auto_execute_tools=False"""

    @tool("manual_tool", "Tool for manual execution", {})
    async def manual_tool(args):
        return {"result": "manual"}

    # Mock response with tool call
    class MockResponse:
        def __init__(self, chunks):
            self.chunks = chunks
        def __aiter__(self):
            async def _gen():
                for chunk in self.chunks:
                    yield chunk
            return _gen()

    async def mock_create(**kwargs):
        return MockResponse([
            SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(
                    tool_calls=[SimpleNamespace(
                        index=0,
                        id="call_1",
                        function=SimpleNamespace(name="manual_tool", arguments='{}')
                    )]
                ),
                finish_reason=None
            )])
        ])

    class MockClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=mock_create)
            )
        async def close(self):
            pass

    monkeypatch.setattr("open_agent.client.create_client", lambda _: MockClient())

    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1",
        tools=[manual_tool],
        auto_execute_tools=False  # Explicit manual mode
    )

    async with Client(options) as client:
        await client.query("Test")

        blocks = []
        async for block in client.receive_messages():
            blocks.append(block)

        # Should receive ToolUseBlock, NOT execute automatically
        assert len(blocks) == 1
        assert isinstance(blocks[0], ToolUseBlock)
        assert blocks[0].name == "manual_tool"

        # Tool should not have been executed yet
        # In manual mode, caller must execute and call add_tool_result()
