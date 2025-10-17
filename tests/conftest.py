"""Shared test fixtures for hooks tests"""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Iterable, List

import pytest

from open_agent import AgentOptions, tool


@pytest.fixture
def fake_client_no_tools(monkeypatch):
    """
    Provide a fake OpenAI client without tools for testing hooks.
    Returns just monkeypatch-ready setup without enqueuing responses.
    """
    response_queue: deque[Iterable[SimpleNamespace]] = deque()

    class FakeResponse:
        def __init__(self, chunks: Iterable[SimpleNamespace]):
            self._iterator = iter(chunks)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._iterator)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class FakeCompletions:
        def __init__(self, queue: deque[Iterable[SimpleNamespace]]):
            self._queue = queue

        async def create(self, *args, **kwargs):
            # Default response: simple text
            if not self._queue:
                # Auto-enqueue a simple response
                response = [_make_chunk(content="Test response")]
                return FakeResponse(response)
            return FakeResponse(self._queue.popleft())

    class FakeClient:
        def __init__(self, queue: deque[Iterable[SimpleNamespace]]):
            self.chat = SimpleNamespace(completions=FakeCompletions(queue))

        async def close(self):
            return None

    def make_client(_options: AgentOptions):
        return FakeClient(response_queue)

    monkeypatch.setattr("open_agent.utils.create_client", make_client)
    monkeypatch.setattr("open_agent.client.create_client", make_client)

    return {}


@pytest.fixture
def fake_client_with_tools(monkeypatch):
    """
    Provide a fake OpenAI client with tools for testing tool hooks.
    Returns a dict with the calculator tool and auto-responding fake client.
    """
    response_queue: deque[Iterable[SimpleNamespace]] = deque()

    # Define a simple calculator tool for testing
    @tool(
        name="calculator",
        description="Perform arithmetic",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    )
    async def calculator(arguments: dict) -> str:
        return str(arguments.get("a", 0) + arguments.get("b", 0))

    class FakeResponse:
        def __init__(self, chunks: Iterable[SimpleNamespace]):
            self._iterator = iter(chunks)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._iterator)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

    class FakeCompletions:
        def __init__(self, queue: deque[Iterable[SimpleNamespace]]):
            self._queue = queue

        async def create(self, *args, **kwargs):
            # Default response: tool call
            if not self._queue:
                # Auto-enqueue a tool call response
                response = _tool_call_chunks(
                    tool_id="tool-123",
                    name="calculator",
                    arguments='{"a": 5, "b": 10}',
                )
                return FakeResponse(response)
            return FakeResponse(self._queue.popleft())

    class FakeClient:
        def __init__(self, queue: deque[Iterable[SimpleNamespace]]):
            self.chat = SimpleNamespace(completions=FakeCompletions(queue))

        async def close(self):
            return None

    def make_client(_options: AgentOptions):
        return FakeClient(response_queue)

    monkeypatch.setattr("open_agent.utils.create_client", make_client)
    monkeypatch.setattr("open_agent.client.create_client", make_client)

    return {"tools": [calculator]}


def _make_chunk(
    *,
    content: str | None = None,
    tool_calls: Iterable[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    """Helper to create a chunk compatible with ToolCallAggregator."""
    delta = SimpleNamespace(content=content, tool_calls=list(tool_calls) if tool_calls else None)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


def _tool_call_chunks(
    *,
    tool_id: str,
    name: str,
    arguments: str,
    text_prefix: str | None = None,
) -> List[SimpleNamespace]:
    """Create streaming chunks that optionally emit text before a tool call."""
    chunks: list[SimpleNamespace] = []
    if text_prefix is not None:
        chunks.append(_make_chunk(content=text_prefix))

    tool_call = SimpleNamespace(
        index=0,
        id=tool_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    chunks.append(_make_chunk(tool_calls=[tool_call]))
    return chunks
