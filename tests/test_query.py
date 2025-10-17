"""Tests for query function"""
from types import SimpleNamespace
from typing import Any, List

import pytest

from any_agent import query, AgentOptions
from any_agent.types import AssistantMessage, TextBlock, ToolUseBlock


class MockAsyncStream:
    """Simple async iterator over pre-defined chunks."""

    def __init__(self, chunks: List[Any]):
        self._chunks = chunks
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


class MockOpenAIClient:
    """Stub AsyncOpenAI client for testing query streaming."""

    def __init__(self, chunks: list[Any]):
        self.closed = False
        self._chunks = chunks
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self._create
            )
        )

    async def _create(self, **_: Any):
        return MockAsyncStream(self._chunks)

    async def close(self):
        self.closed = True


def _make_text_chunk(text: str) -> Any:
    delta = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


def _make_tool_chunk(
    *,
    index: int,
    tool_id: str | None = None,
    name: str | None = None,
    arguments: str | None = None
) -> Any:
    function = SimpleNamespace(name=name, arguments=arguments)
    tool_call = SimpleNamespace(
        index=index,
        id=tool_id,
        function=function
    )
    delta = SimpleNamespace(content=None, tool_calls=[tool_call])
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


@pytest.mark.asyncio
async def test_query_streams_text_and_tool(monkeypatch):
    """query() should yield text and finalize tool calls correctly."""
    chunks = [
        _make_text_chunk("Hello "),
        _make_text_chunk("world!"),
        _make_tool_chunk(index=0, tool_id="call_1", name="search", arguments='{"query": "news"'),
        _make_tool_chunk(index=0, arguments=' }')
    ]

    mock_client = MockOpenAIClient(chunks)

    monkeypatch.setattr(
        "any_agent.client.create_client",
        lambda options: mock_client
    )

    options = AgentOptions(
        system_prompt="System",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )

    result = query(prompt="Hello?", options=options)
    collected = []

    async for message in result:
        assert isinstance(message, AssistantMessage)
        collected.append(message)

    # Expect two yields for text streaming and a third after finalize
    assert len(collected) == 3
    final_message = collected[-1]
    text_payload = "".join(
        block.text for block in final_message.content if isinstance(block, TextBlock)
    )
    assert text_payload == "Hello world!"

    tool_blocks = [
        block
        for block in final_message.content
        if isinstance(block, ToolUseBlock)
    ]
    assert len(tool_blocks) == 1
    assert tool_blocks[0].id == "call_1"
    assert tool_blocks[0].name == "search"
    assert tool_blocks[0].input == {"query": "news"}

    # Ensure the client was closed after streaming
    assert mock_client.closed

@pytest.mark.asyncio
async def test_query_propagates_errors_and_closes_client(monkeypatch):
    """Errors from OpenAI client should propagate while still closing the client."""
    class FailingClient:
        def __init__(self):
            self.closed = False
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=self._create
                )
            )

        async def _create(self, **_: Any):
            raise RuntimeError("network issue")

        async def close(self):
            self.closed = True

    failing_client = FailingClient()
    monkeypatch.setattr("any_agent.client.create_client", lambda _options: failing_client)

    options = AgentOptions(
        system_prompt="System",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )

    with pytest.raises(RuntimeError, match="network issue"):
        async for _ in query("Hi", options):
            pass

    assert failing_client.closed
