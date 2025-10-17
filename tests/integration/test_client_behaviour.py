"""Integration-style tests using lightweight fakes instead of real servers."""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Iterable, List, Tuple

import pytest

from open_agent import AgentOptions, Client, TextBlock, ToolUseBlock, query


@pytest.fixture
def agent_options() -> AgentOptions:
    """Provide fresh AgentOptions for each test."""
    return AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="fake-model",
        base_url="http://fake-server.local/v1",
        max_turns=5,
        max_tokens=128,
    )


@pytest.fixture
def fake_openai(monkeypatch):
    """
    Replace `create_client` with a fake that serves queued streaming responses.

    Tests call `enqueue_response` with an iterable of fake streaming chunks.
    Each call to `chat.completions.create` pops the next response off the queue.
    """
    response_queue: deque[Iterable[SimpleNamespace]] = deque()
    call_log: list[Tuple[tuple, dict]] = []

    class FakeResponse:
        """Async iterator yielding preset streaming chunks."""

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
            call_log.append((args, kwargs))
            if not self._queue:
                raise AssertionError("Fake client received an unexpected request")
            return FakeResponse(self._queue.popleft())

    class FakeClient:
        def __init__(self, queue: deque[Iterable[SimpleNamespace]]):
            self.chat = SimpleNamespace(completions=FakeCompletions(queue))

        async def close(self):
            return None

    def enqueue_response(*responses: Iterable[SimpleNamespace]):
        for response in responses:
            response_queue.append(response)

    def make_client(_options: AgentOptions):
        return FakeClient(response_queue)

    monkeypatch.setattr("open_agent.utils.create_client", make_client)
    monkeypatch.setattr("open_agent.client.create_client", make_client)

    return {"enqueue": enqueue_response, "calls": call_log}


def text_chunks(*parts: str) -> List[SimpleNamespace]:
    """Create streaming chunks that emit text pieces in order."""
    return [make_chunk(content=part) for part in parts]


def tool_call_chunks(
    *,
    tool_id: str,
    name: str,
    arguments: str,
    text_prefix: str | None = None,
) -> List[SimpleNamespace]:
    """Create streaming chunks that optionally emit text before a tool call."""
    chunks: list[SimpleNamespace] = []
    if text_prefix is not None:
        chunks.append(make_chunk(content=text_prefix))

    tool_call = SimpleNamespace(
        index=0,
        id=tool_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    chunks.append(make_chunk(tool_calls=[tool_call]))
    return chunks


def make_chunk(
    *,
    content: str | None = None,
    tool_calls: Iterable[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    """Helper to create a chunk compatible with ToolCallAggregator."""
    delta = SimpleNamespace(content=content, tool_calls=list(tool_calls) if tool_calls else None)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


async def collect_blocks(generator):
    """Consume an async generator and return the yielded items."""
    items = []
    async for item in generator:
        items.append(item)
    return items


@pytest.mark.asyncio
async def test_query_streams_text_blocks(agent_options, fake_openai):
    fake_openai["enqueue"](text_chunks("Hello", " world"))

    messages = []
    async for message in query("Say hello", agent_options):
        messages.extend(message.content)

    combined_text = "".join(block.text for block in messages if isinstance(block, TextBlock))
    assert combined_text == "Hello world"

    # Ensure the request parameters were forwarded correctly
    assert len(fake_openai["calls"]) == 1
    _, kwargs = fake_openai["calls"][0]
    assert kwargs["model"] == agent_options.model
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][-1]["content"] == "Say hello"


@pytest.mark.asyncio
async def test_client_multi_turn_tracks_history(agent_options, fake_openai):
    fake_openai["enqueue"](
        text_chunks("Paris"),
        text_chunks("Approximately 2 million."),
        text_chunks("Yes, Paris is larger by population."),
    )

    async with Client(agent_options) as client:
        await client.query("What is the capital of France?")
        blocks1 = await collect_blocks(client.receive_messages())
        assert "".join(block.text for block in blocks1 if isinstance(block, TextBlock)) == "Paris"

        await client.query("What is its population?")
        blocks2 = await collect_blocks(client.receive_messages())
        assert "2 million" in "".join(
            block.text for block in blocks2 if isinstance(block, TextBlock)
        )

        await client.query("Is it bigger than Lyon?")
        blocks3 = await collect_blocks(client.receive_messages())
        assert any(
            "larger" in block.text for block in blocks3 if isinstance(block, TextBlock)
        )

        assert client.turn_count == 3
        assert client.turn_metadata == {"turn_count": 3, "max_turns": agent_options.max_turns}

        # History should alternate user/assistant entries
        assert len(client.history) == 6
        assert client.history[0]["role"] == "user"
        assert client.history[1]["role"] == "assistant"


@pytest.mark.asyncio
async def test_client_handles_tool_call_and_results(agent_options, fake_openai):
    enqueue = fake_openai["enqueue"]

    enqueue(
        tool_call_chunks(
            tool_id="call-1",
            name="calculate",
            arguments='{"expression": "2+2"}',
            text_prefix="Let me calculate that.",
        ),
        text_chunks("The result is 4."),
    )

    async with Client(agent_options) as client:
        await client.query("Please compute 2+2.")
        blocks = await collect_blocks(client.receive_messages())

        text_output = "".join(block.text for block in blocks if isinstance(block, TextBlock))
        assert text_output == "Let me calculate that."

        tool_blocks = [block for block in blocks if isinstance(block, ToolUseBlock)]
        assert len(tool_blocks) == 1
        tool_block = tool_blocks[0]
        assert tool_block.name == "calculate"
        assert tool_block.input == {"expression": "2+2"}

        client.add_tool_result(tool_block.id, {"result": 4}, name=tool_block.name)

        await client.query("Continue with the result.")
        follow_up_blocks = await collect_blocks(client.receive_messages())
        follow_up_text = "".join(
            block.text for block in follow_up_blocks if isinstance(block, TextBlock)
        )
        assert follow_up_text == "The result is 4."

        # Tool result should be recorded in history for subsequent turns
        tool_entries = [msg for msg in client.history if msg["role"] == "tool"]
        assert len(tool_entries) == 1
        tool_entry = tool_entries[0]
        assert tool_entry["tool_call_id"] == tool_block.id
        assert tool_entry["content"] == '{"result": 4}'

        # Second request should include tool response in messages
        assert len(fake_openai["calls"]) == 2
        second_call_kwargs = fake_openai["calls"][1][1]
        roles = [msg["role"] for msg in second_call_kwargs["messages"]]
        assert "tool" in roles


@pytest.mark.asyncio
async def test_query_yields_tool_block(agent_options, fake_openai):
    fake_openai["enqueue"](
        tool_call_chunks(
            tool_id="call-xyz",
            name="lookup",
            arguments='{"topic": "weather"}',
            text_prefix=None,
        )
    )

    tool_blocks = []
    async for message in query("Use a tool", agent_options):
        tool_blocks.extend(block for block in message.content if isinstance(block, ToolUseBlock))

    assert len(tool_blocks) == 1
    assert tool_blocks[0].name == "lookup"
    assert tool_blocks[0].input == {"topic": "weather"}
