# Implementation Plan: Any-Agent SDK

## Overview

Phased implementation plan to build a minimal, working SDK for local/self-hosted LLMs with claude-agent-sdk-compatible API.

**Target**: 3 weeks to validated MVP

**What we're NOT building** (Agent responsibility):
- ❌ Persistent storage/database
- ❌ Memory management/RAG
- ❌ Tool execution
- ❌ Custom schemas

Instead, we provide conversation primitives (`client.history`, `client.turn_metadata`) that agents can use with their own storage solutions (like your `copy_editor.database.py`).

## Phase 1: Foundation (Week 1, Days 1-3)

### Day 1: Project Setup & Types

**Goal**: Get project structure and type definitions in place

**Tasks**:
1. ✅ Create project structure
2. ✅ Set up pyproject.toml
3. ✅ Archive old docs
4. ✅ Update CLAUDE.md and README.md
5. 🔨 Implement types.py

**Deliverables**:

`pyproject.toml`:
```toml
[project]
name = "any-agent"
version = "0.1.0"
description = "Claude Agent SDK-style API for local/self-hosted LLMs"
requires-python = ">=3.10"
dependencies = [
    "openai>=1.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[build-system]
requires = ["setuptools>=68.0.0", "wheel"]
build-backend = "setuptools.build_meta"
```

`any_agent/types.py`:
```python
"""Type definitions matching claude-agent-sdk patterns"""
from dataclasses import dataclass, field
from typing import Literal

@dataclass
class TextBlock:
    """Text content from assistant"""
    text: str
    type: Literal["text"] = "text"

@dataclass
class ToolUseBlock:
    """Tool call from assistant"""
    id: str
    name: str
    input: dict
    type: Literal["tool_use"] = "tool_use"

@dataclass
class AssistantMessage:
    """Full assistant message"""
    role: Literal["assistant"] = "assistant"
    content: list[TextBlock | ToolUseBlock] = field(default_factory=list)

@dataclass
class ToolUseError:
    """Error when tool call fails to parse"""
    error: str
    raw_data: str | None = None
    type: Literal["tool_use_error"] = "tool_use_error"

@dataclass
class AgentOptions:
    """Configuration for agent"""
    system_prompt: str
    model: str
    base_url: str
    max_turns: int = 1
    max_tokens: int | None = 4096  # Default 4096, None uses provider default
    temperature: float = 0.7
    api_key: str = "not-needed"
```

**Success Criteria**:
- [ ] Project installs with `pip install -e .`
- [ ] types.py imports without errors
- [ ] All dataclasses construct correctly

### Day 2: Utilities & OpenAI Client

**Goal**: OpenAI client creation and response parsing

**Tasks**:
1. Implement create_client()
2. Implement format_messages()
3. Implement parse_chunk()
4. Write unit tests

**Deliverables**:

`any_agent/utils.py`:
```python
"""OpenAI client utilities"""
import json
import logging
from typing import Any
from openai import AsyncOpenAI
from .types import AgentOptions, TextBlock, ToolUseBlock, ToolUseError

logger = logging.getLogger(__name__)

def create_client(options: AgentOptions) -> AsyncOpenAI:
    """Create configured AsyncOpenAI client"""
    return AsyncOpenAI(
        base_url=options.base_url,
        api_key=options.api_key,
        timeout=60.0
    )

def format_messages(
    system_prompt: str,
    user_prompt: str,
    history: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Format messages for OpenAI API"""
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_prompt})
    return messages


class ToolCallAggregator:
    """
    Stateful aggregator for streaming tool calls.

    OpenAI streams tool calls incrementally:
    - Arguments arrive as partial JSON strings across multiple chunks
    - id and name may appear in separate chunks
    - Multiple tools use 'index' to track which tool is being updated
    """

    def __init__(self):
        self.pending_tools: dict[int, dict[str, Any]] = {}

    def process_chunk(self, chunk) -> TextBlock | None:
        """
        Process a streaming chunk, accumulating tool calls.
        Returns TextBlock if there's text content, None otherwise.
        Tool calls are accumulated internally until finalized.
        """
        try:
            if not chunk.choices:
                return None

            delta = chunk.choices[0].delta

            # Handle text content
            if hasattr(delta, 'content') and delta.content:
                return TextBlock(text=delta.content)

            # Handle tool call deltas
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                for tc in delta.tool_calls:
                    index = tc.index

                    # Initialize tool slot if needed
                    if index not in self.pending_tools:
                        self.pending_tools[index] = {
                            "id": None,
                            "name": None,
                            "arguments_buffer": ""
                        }

                    tool = self.pending_tools[index]

                    # Update id if present
                    if hasattr(tc, 'id') and tc.id:
                        tool["id"] = tc.id

                    # Update name if present
                    if hasattr(tc, 'function') and hasattr(tc.function, 'name') and tc.function.name:
                        tool["name"] = tc.function.name

                    # Accumulate arguments
                    if hasattr(tc, 'function') and hasattr(tc.function, 'arguments') and tc.function.arguments:
                        tool["arguments_buffer"] += tc.function.arguments

            return None

        except Exception as e:
            logger.warning(f"Failed to process chunk: {e}")
            return None

    def finalize_tools(self) -> list[ToolUseBlock | ToolUseError]:
        """
        Finalize all pending tool calls, parsing accumulated JSON arguments.
        Returns list of completed ToolUseBlocks or ToolUseErrors.
        """
        results: list[ToolUseBlock | ToolUseError] = []

        for index, tool in sorted(self.pending_tools.items()):
            # Validate required fields
            if not tool["id"] or not tool["name"]:
                logger.error(f"Tool at index {index} missing id or name: {tool}")
                results.append(ToolUseError(
                    error=f"Tool call missing required fields (id={tool['id']}, name={tool['name']})",
                    raw_data=str(tool)
                ))
                continue

            # Parse arguments JSON
            try:
                if tool["arguments_buffer"]:
                    input_dict = json.loads(tool["arguments_buffer"])
                else:
                    input_dict = {}

                results.append(ToolUseBlock(
                    id=tool["id"],
                    name=tool["name"],
                    input=input_dict
                ))

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool arguments JSON: {e}")
                logger.error(f"Raw buffer: {tool['arguments_buffer']}")
                results.append(ToolUseError(
                    error=f"Invalid JSON in tool arguments: {e}",
                    raw_data=tool["arguments_buffer"]
                ))

        # Clear state for next turn
        self.pending_tools.clear()

        return results
```

`tests/test_utils.py`:
```python
"""Tests for utils module"""
import pytest
from any_agent.utils import format_messages, create_client, ToolCallAggregator
from any_agent.types import AgentOptions, TextBlock, ToolUseBlock, ToolUseError

def test_format_messages_basic():
    messages = format_messages("System", "User")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

def test_format_messages_with_history():
    history = [{"role": "user", "content": "Hi"}]
    messages = format_messages("System", "Again", history)
    assert len(messages) == 3
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hi"

def test_create_client():
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )
    client = create_client(options)
    assert client.base_url == "http://localhost:1234/v1"

def test_tool_aggregator_basic():
    """Test tool call aggregation across multiple chunks"""
    aggregator = ToolCallAggregator()

    # Simulate streaming chunks (mock objects would be used in practice)
    # For now, verify API exists
    assert hasattr(aggregator, 'process_chunk')
    assert hasattr(aggregator, 'finalize_tools')
```

**Success Criteria**:
- [ ] All tests pass
- [ ] create_client() returns AsyncOpenAI
- [ ] format_messages() handles history
- [ ] ToolCallAggregator accumulates tool calls correctly
- [ ] finalize_tools() produces ToolUseBlocks or ToolUseErrors

### Day 3: Simple query() Function

**Goal**: Working single-turn query

**Tasks**:
1. Implement query() function
2. Test with LM Studio (manual)
3. Write integration test (mocked)

**Deliverables**:

`any_agent/client.py`:
```python
"""Main client implementation"""
import logging
from typing import AsyncGenerator, Any
from .types import AgentOptions, AssistantMessage, TextBlock, ToolUseBlock, ToolUseError
from .utils import create_client, format_messages, ToolCallAggregator

logger = logging.getLogger(__name__)

async def query(
    prompt: str,
    options: AgentOptions
) -> AsyncGenerator[AssistantMessage, None]:
    """
    Simple single-turn query.

    Usage:
        options = AgentOptions(...)
        result = query("Hello", options)
        async for msg in result:
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(block.text)
                elif isinstance(block, ToolUseBlock):
                    print(f"Tool: {block.name}")
                elif isinstance(block, ToolUseError):
                    print(f"Tool error: {block.error}")
    """
    client = create_client(options)
    messages = format_messages(options.system_prompt, prompt)

    try:
        response = await client.chat.completions.create(
            model=options.model,
            messages=messages,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            stream=True
        )

        current_message = AssistantMessage(content=[])
        aggregator = ToolCallAggregator()

        async for chunk in response:
            # Process text blocks immediately
            text_block = aggregator.process_chunk(chunk)
            if text_block:
                current_message.content.append(text_block)
                yield current_message

        # Finalize any pending tool calls
        tool_blocks = aggregator.finalize_tools()
        if tool_blocks:
            current_message.content.extend(tool_blocks)
            yield current_message

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise
```

`any_agent/__init__.py`:
```python
"""Any-Agent SDK - Claude Agent SDK-style API for local LLMs"""
from .client import query, Client
from .types import AgentOptions, TextBlock, ToolUseBlock, ToolUseError, AssistantMessage

__all__ = [
    "query",
    "Client",
    "AgentOptions",
    "TextBlock",
    "ToolUseBlock",
    "ToolUseError",
    "AssistantMessage",
]
```

`examples/simple_lmstudio.py`:
```python
"""Simple query example with LM Studio"""
import asyncio
from any_agent import query, AgentOptions

async def main():
    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="qwen2.5-32b-instruct",
        base_url="http://localhost:1234/v1",
        max_turns=1,
        temperature=0.7
    )

    print("Querying LM Studio...")
    result = query(prompt="What is 2+2?", options=options)

    response_text = ""
    async for msg in result:
        for block in msg.content:
            if isinstance(block, TextBlock):
                response_text += block.text
                print(block.text, end="", flush=True)

    print(f"\n\nFull response: {response_text}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Success Criteria**:
- [ ] query() returns async generator
- [ ] Streams TextBlocks from LM Studio
- [ ] example/simple_lmstudio.py works
- [ ] Clean error messages on connection failure

## Phase 2: Multi-Turn Client (Week 1, Days 4-7)

### Day 4: Client Class Structure

**Goal**: Implement Client class skeleton

**Tasks**:
1. Implement Client.__init__
2. Implement Client.query()
3. Implement Client.receive_messages()
4. Add message history tracking

**Deliverables**:

Add to `any_agent/client.py`:
```python
class Client:
    """
    Multi-turn conversation client.

    Usage:
        async with Client(options) as client:
            await client.query("Hello")
            async for msg in client.receive_messages():
                # Process
    """

    def __init__(self, options: AgentOptions):
        self.options = options
        self.client = create_client(options)
        self.message_history: list[dict[str, Any]] = []
        self.turn_count = 0
        self.response_stream = None
        self._aggregator: ToolCallAggregator | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup
        pass

    async def query(self, prompt: str):
        """Send query and prepare to receive messages"""
        messages = format_messages(
            self.options.system_prompt,
            prompt,
            self.message_history
        )

        # Add user message to history
        self.message_history.append({
            "role": "user",
            "content": prompt
        })

        self.response_stream = await self.client.chat.completions.create(
            model=self.options.model,
            messages=messages,
            max_tokens=self.options.max_tokens,
            temperature=self.options.temperature,
            stream=True
        )

        # Initialize aggregator for this turn
        self._aggregator = ToolCallAggregator()

    async def receive_messages(self) -> AsyncGenerator[TextBlock | ToolUseBlock | ToolUseError, None]:
        """Stream individual blocks from response"""
        if not self.response_stream:
            raise RuntimeError("Call query() first")
        if not self._aggregator:
            raise RuntimeError("Aggregator not initialized")

        assistant_blocks: list[TextBlock | ToolUseBlock | ToolUseError] = []

        # Stream text blocks
        async for chunk in self.response_stream:
            text_block = self._aggregator.process_chunk(chunk)
            if text_block:
                assistant_blocks.append(text_block)
                yield text_block

        # Finalize tool calls
        tool_blocks = self._aggregator.finalize_tools()
        if tool_blocks:
            assistant_blocks.extend(tool_blocks)
            for tool_block in tool_blocks:
                yield tool_block

        # Add assistant response to history with proper structure
        # Preserve both text and tool calls for OpenAI API compatibility
        history_entry = self._format_history_entry(assistant_blocks)
        self.message_history.append(history_entry)

        self.turn_count += 1

        # Check max turns
        if self.turn_count >= self.options.max_turns:
            logger.info(f"Reached max_turns ({self.options.max_turns})")

    def _format_history_entry(
        self,
        blocks: list[TextBlock | ToolUseBlock | ToolUseError]
    ) -> dict[str, Any]:
        """
        Format assistant response for message history.
        Handles mixed text + tool calls per OpenAI API format.
        """
        # Separate text and tools
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in blocks:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": str(block.input)  # OpenAI expects JSON string
                    }
                })
            # Skip ToolUseError - don't preserve malformed calls in history

        # Build history entry per OpenAI format
        if tool_calls:
            # Assistant used tools
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(text_parts) if text_parts else None,
                "tool_calls": tool_calls
            }
        else:
            # Text-only response
            entry = {
                "role": "assistant",
                "content": "".join(text_parts)
            }

        return entry

    @property
    def history(self) -> list[dict[str, Any]]:
        """Get full conversation history for agent storage"""
        return self.message_history.copy()

    @property
    def turn_metadata(self) -> dict[str, Any]:
        """Get conversation metadata for agent tracking"""
        return {
            "turn_count": self.turn_count,
            "max_turns": self.options.max_turns
        }
```

**Success Criteria**:
- [ ] Client tracks message history
- [ ] Client tracks turn count
- [ ] receive_messages() yields blocks
- [ ] Context manager works

### Day 5-6: Multi-Turn Testing

**Goal**: Validate multi-turn conversations

**Tasks**:
1. Write multi-turn example
2. Test with LM Studio
3. Test with Ollama
4. Add tests

**Deliverables**:

`examples/ollama_chat.py`:
```python
"""Multi-turn chat example with Ollama"""
import asyncio
from any_agent import Client, AgentOptions, TextBlock, ToolUseBlock

async def main():
    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="llama3.1:70b",
        base_url="http://localhost:11434/v1",
        max_turns=5,
        temperature=0.7
    )

    async with Client(options) as client:
        # Turn 1
        print("User: What's the capital of France?")
        await client.query("What's the capital of France?")

        async for block in client.receive_messages():
            if isinstance(block, TextBlock):
                print(f"Assistant: {block.text}", end="", flush=True)

        print("\n")

        # Turn 2
        print("User: What's its population?")
        await client.query("What's its population?")

        async for block in client.receive_messages():
            if isinstance(block, TextBlock):
                print(f"Assistant: {block.text}", end="", flush=True)

        print(f"\n\nTotal turns: {client.turn_count}")

if __name__ == "__main__":
    asyncio.run(main())
```

**Success Criteria**:
- [ ] Multi-turn maintains context
- [ ] Works with LM Studio
- [ ] Works with Ollama
- [ ] max_turns enforced

### Day 7: Tool Monitoring

**Goal**: Yield ToolUseBlocks for user to monitor

**Tasks**:
1. Test with function calling model
2. Verify ToolUseBlocks parse correctly
3. Document tool monitoring pattern

**Deliverables**:

`examples/tool_monitoring.py`:
```python
"""Tool use monitoring example"""
import asyncio
from any_agent import Client, AgentOptions, TextBlock, ToolUseBlock

async def main():
    options = AgentOptions(
        system_prompt="You can use tools to help answer questions.",
        model="qwen2.5-32b-instruct",
        base_url="http://localhost:1234/v1",
        max_turns=3
    )

    async with Client(options) as client:
        await client.query("What's the weather in Paris?")

        async for block in client.receive_messages():
            if isinstance(block, TextBlock):
                print(f"Text: {block.text}")

            elif isinstance(block, ToolUseBlock):
                print(f"🔧 Tool: {block.name}")
                print(f"   Input: {block.input}")
                # User can log, execute, or ignore

if __name__ == "__main__":
    asyncio.run(main())
```

**Success Criteria**:
- [ ] ToolUseBlocks yield correctly
- [ ] User can monitor tool calls
- [ ] Example demonstrates pattern

## Phase 3: Validation (Week 2)

### Week 2 Goals

1. Port copy_editor agent from stories project
2. Compare output to claude-agent-sdk version
3. Fix any issues discovered
4. Document migration process

**Tasks**:

1. Create `examples/copy_editor_port.py`
2. Copy logic from `~/Dev/stories/.agents/copy_editor/agent.py`
3. Replace claude-agent-sdk imports with any-agent
4. Test on sample chapter
5. Compare results

**Success Metric**:
- [ ] < 5 lines changed from original
- [ ] Output quality comparable
- [ ] Migration documented

## Phase 4: Polish & Release (Week 3)

### Documentation

- [ ] Complete API reference
- [ ] Add troubleshooting guide
- [ ] Document common patterns
- [ ] Add model recommendations

### Packaging

- [ ] Set up GitHub Actions CI
- [ ] Configure PyPI publishing
- [ ] Add LICENSE file
- [ ] Release v0.1.0

### Testing

- [ ] 80%+ code coverage
- [ ] Integration tests with mocked responses
- [ ] Manual validation with 3+ models

## Success Criteria

Project is ready for release when:

- [x] CLAUDE.md and README.md updated
- [ ] query() function works
- [ ] Client class works multi-turn
- [ ] Tool monitoring works
- [ ] Copy editor agent ports cleanly
- [ ] Works with LM Studio, Ollama, llama.cpp
- [ ] Tests pass
- [ ] Documentation complete
- [ ] PyPI package published

## Development Commands

```bash
# Setup
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black any_agent/
ruff check any_agent/

# Test with LM Studio
python examples/simple_lmstudio.py

# Test with Ollama
python examples/ollama_chat.py
```

## Notes

- Focus on MVP - don't add features not in scope
- Keep it simple - 300-400 lines total
- Test early and often with real local models
- Document as you go
