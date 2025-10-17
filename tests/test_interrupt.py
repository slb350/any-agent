"""Tests for interrupt capability"""
import asyncio
from types import SimpleNamespace
import pytest
from open_agent import Client, AgentOptions, tool
from open_agent.types import TextBlock, ToolUseBlock


@pytest.mark.asyncio
async def test_interrupt_no_operation():
    """Test interrupt() is safe when no operation in progress"""
    options = AgentOptions(
        model="test-model",
        system_prompt="Test",
        base_url="http://test",
        api_key="test"
    )

    client = Client(options)

    # Should be safe to call interrupt with no operation
    await client.interrupt()

    # Verify flag state - interrupt() sets the flag even if no operation is running
    assert client._interrupted == True


@pytest.mark.asyncio
async def test_interrupt_basic(monkeypatch):
    """Test basic interrupt during streaming response"""

    class MockResponse:
        def __init__(self, chunks):
            self.chunks = chunks
            self._closed = False

        def __aiter__(self):
            async def _gen():
                for chunk in self.chunks:
                    if self._closed:
                        break
                    yield chunk
            return _gen()

        async def aclose(self):
            self._closed = True

    # Create multiple text chunks
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="First "),
            finish_reason=None
        )]),
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="Second "),
            finish_reason=None
        )]),
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="Third"),
            finish_reason=None
        )])
    ]

    async def mock_create(**kwargs):
        return MockResponse(chunks)

    class MockClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=mock_create)
            )

        async def close(self):
            pass

    monkeypatch.setattr("open_agent.client.create_client", lambda _: MockClient())

    options = AgentOptions(
        model="test-model",
        system_prompt="Test",
        base_url="http://test",
        api_key="test"
    )

    client = Client(options)
    await client.query("Test")

    # Start receiving and interrupt after first block
    count = 0
    async for block in client.receive_messages():
        count += 1
        if count == 1:
            await client.interrupt()
            break

    # Should have received at least one block before interrupt
    assert count >= 1

    # Verify client state is clean
    assert client.response_stream is None
    assert client._aggregator is None


@pytest.mark.asyncio
async def test_interrupt_before_receive(monkeypatch):
    """Interrupt after query but before receive_messages starts"""

    class MockResponse:
        def __init__(self, chunks):
            self.chunks = chunks

        def __aiter__(self):
            async def _gen():
                for chunk in self.chunks:
                    yield chunk
            return _gen()

        async def aclose(self):
            pass

    async def mock_create(**kwargs):
        return MockResponse([
            SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content="Should not be seen"),
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
        model="test-model",
        system_prompt="Test",
        base_url="http://test",
        api_key="test"
    )

    async with Client(options) as client:
        await client.query("Test")
        await client.interrupt()  # Interrupt before receiving

        blocks = []
        async for block in client.receive_messages():
            blocks.append(block)

        assert blocks == []


@pytest.mark.asyncio
async def test_interrupt_multiple_times_idempotent(monkeypatch):
    """Test calling interrupt() multiple times is safe"""

    class MockResponse:
        def __init__(self, chunks):
            self.chunks = chunks

        def __aiter__(self):
            async def _gen():
                for chunk in self.chunks:
                    yield chunk
            return _gen()

        async def aclose(self):
            pass

    async def mock_create(**kwargs):
        return MockResponse([
            SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content="Test response"),
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
        model="test-model",
        system_prompt="Test",
        base_url="http://test",
        api_key="test"
    )

    client = Client(options)
    await client.query("Test")

    # Call interrupt multiple times
    await client.interrupt()
    await client.interrupt()
    await client.interrupt()

    # Client should still work after multiple interrupts
    # Flag should be reset by query()
    assert client._interrupted == True

    await client.query("Test 2")
    assert client._interrupted == False


@pytest.mark.asyncio
async def test_interrupt_then_query_again(monkeypatch):
    """Test client can be reused after interrupt"""

    class MockResponse:
        def __init__(self, chunks):
            self.chunks = chunks

        def __aiter__(self):
            async def _gen():
                for chunk in self.chunks:
                    yield chunk
            return _gen()

        async def aclose(self):
            pass

    call_count = [0]

    async def mock_create(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return MockResponse([
                SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="First response"),
                    finish_reason=None
                )])
            ])
        else:
            return MockResponse([
                SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="Second response"),
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
        model="test-model",
        system_prompt="Test",
        base_url="http://test",
        api_key="test"
    )

    client = Client(options)

    # First query
    await client.query("First")
    await client.interrupt()

    # Second query should work
    await client.query("Second")

    response = []
    async for block in client.receive_messages():
        response.append(block)

    assert len(response) == 1
    assert response[0].text == "Second response"


@pytest.mark.asyncio
async def test_interrupt_with_context_manager(monkeypatch):
    """Test interrupt works properly with async context manager"""

    class MockResponse:
        def __init__(self, chunks):
            self.chunks = chunks

        def __aiter__(self):
            async def _gen():
                for chunk in self.chunks:
                    yield chunk
            return _gen()

        async def aclose(self):
            pass

    async def mock_create(**kwargs):
        return MockResponse([
            SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content="Test"),
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
        model="test-model",
        system_prompt="Test",
        base_url="http://test",
        api_key="test"
    )

    async with Client(options) as client:
        await client.query("Test")
        await client.interrupt()

    # Should clean up properly (no exceptions)


@pytest.mark.asyncio
async def test_interrupt_during_auto_execution(monkeypatch):
    """Test interrupt during auto-execution stops tool loop"""

    tools_executed = []

    @tool("tool1", "First tool", {})
    async def tool1(args):
        tools_executed.append("tool1")
        await asyncio.sleep(0.01)
        return {"result": "tool1 done"}

    @tool("tool2", "Second tool", {})
    async def tool2(args):
        tools_executed.append("tool2")
        return {"result": "tool2 done"}

    call_count = [0]

    class MockResponse:
        def __init__(self, chunks):
            self.chunks = chunks

        def __aiter__(self):
            async def _gen():
                for chunk in self.chunks:
                    yield chunk
            return _gen()

        async def aclose(self):
            pass

    async def mock_create(**kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First response: call tool1
            return MockResponse([
                SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(
                        tool_calls=[SimpleNamespace(
                            index=0,
                            id="call_1",
                            function=SimpleNamespace(name="tool1", arguments='{}')
                        )]
                    ),
                    finish_reason=None
                )])
            ])
        elif call_count[0] == 2:
            # Second response: call tool2
            return MockResponse([
                SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(
                        tool_calls=[SimpleNamespace(
                            index=0,
                            id="call_2",
                            function=SimpleNamespace(name="tool2", arguments='{}')
                        )]
                    ),
                    finish_reason=None
                )])
            ])
        else:
            # Final response: text
            return MockResponse([
                SimpleNamespace(choices=[SimpleNamespace(
                    delta=SimpleNamespace(content="All done"),
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
        model="test-model",
        system_prompt="Test",
        base_url="http://test",
        api_key="test",
        tools=[tool1, tool2],
        auto_execute_tools=True,
        max_tool_iterations=5
    )

    client = Client(options)
    await client.query("Test")

    # Receive blocks and interrupt after seeing first tool
    tool_blocks_seen = 0
    async for block in client.receive_messages():
        if isinstance(block, ToolUseBlock):
            tool_blocks_seen += 1
            if tool_blocks_seen == 1:
                await client.interrupt()

    # Should have seen only 1 tool block before interrupt
    assert tool_blocks_seen == 1
    # Should have executed at most 1 tool
    assert len(tools_executed) <= 1


@pytest.mark.asyncio
async def test_interrupt_from_different_task(monkeypatch):
    """Test interrupt() called from separate asyncio task"""

    class MockResponse:
        def __init__(self, chunks):
            self.chunks = chunks

        def __aiter__(self):
            async def _gen():
                for chunk in self.chunks:
                    await asyncio.sleep(0.01)  # Slow streaming
                    yield chunk
            return _gen()

        async def aclose(self):
            pass

    # Create many chunks for slow streaming
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=f"Chunk {i} "),
            finish_reason=None
        )])
        for i in range(100)
    ]

    async def mock_create(**kwargs):
        return MockResponse(chunks)

    class MockClient:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=mock_create)
            )

        async def close(self):
            pass

    monkeypatch.setattr("open_agent.client.create_client", lambda _: MockClient())

    options = AgentOptions(
        model="test-model",
        system_prompt="Test",
        base_url="http://test",
        api_key="test"
    )

    client = Client(options)

    async def stream_task():
        await client.query("Test")
        count = 0
        async for block in client.receive_messages():
            count += 1
        return count

    async def interrupt_task():
        await asyncio.sleep(0.05)  # Let streaming start
        await client.interrupt()

    # Run both tasks concurrently
    stream_count, _ = await asyncio.gather(
        stream_task(),
        interrupt_task(),
        return_exceptions=True
    )

    # Stream was interrupted (should be less than full 100 chunks)
    if isinstance(stream_count, int):
        assert stream_count < 100


@pytest.mark.asyncio
async def test_interrupt_resets_on_next_query(monkeypatch):
    """Test that query() resets interrupt flag"""

    class MockResponse:
        def __init__(self, chunks):
            self.chunks = chunks

        def __aiter__(self):
            async def _gen():
                for chunk in self.chunks:
                    yield chunk
            return _gen()

        async def aclose(self):
            pass

    async def mock_create(**kwargs):
        return MockResponse([
            SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content="Test response"),
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
        model="test-model",
        system_prompt="Test",
        base_url="http://test",
        api_key="test"
    )

    client = Client(options)

    # Set interrupted flag manually
    client._interrupted = True

    # Start new query
    await client.query("Test")

    # Flag should be reset by query()
    assert client._interrupted == False
