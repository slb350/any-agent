# Implementation Plan: Any-Agent SDK

## Overview

Phased implementation plan to build a minimal, working SDK for local/self-hosted LLMs with claude-agent-sdk-compatible API.

**Target**: 3 weeks to validated MVP

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
class AgentOptions:
    """Configuration for agent"""
    system_prompt: str
    model: str
    base_url: str
    max_turns: int = 1
    max_tokens: int = 8000
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
import logging
from openai import AsyncOpenAI
from .types import AgentOptions, TextBlock, ToolUseBlock

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
    history: list[dict] | None = None
) -> list[dict]:
    """Format messages for OpenAI API"""
    messages = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_prompt})
    return messages

def parse_chunk(chunk) -> TextBlock | ToolUseBlock | None:
    """
    Parse streaming chunk into message block.

    OpenAI chunks look like:
    chunk.choices[0].delta.content = "text"
    chunk.choices[0].delta.tool_calls = [{...}]
    """
    try:
        if not chunk.choices:
            return None

        delta = chunk.choices[0].delta

        # Text content
        if hasattr(delta, 'content') and delta.content:
            return TextBlock(text=delta.content)

        # Tool calls
        if hasattr(delta, 'tool_calls') and delta.tool_calls:
            tool_call = delta.tool_calls[0]
            return ToolUseBlock(
                id=tool_call.id,
                name=tool_call.function.name,
                input=tool_call.function.arguments
            )

        return None

    except Exception as e:
        logger.warning(f"Failed to parse chunk: {e}")
        return None
```

`tests/test_utils.py`:
```python
"""Tests for utils module"""
import pytest
from any_agent.utils import format_messages, create_client
from any_agent.types import AgentOptions

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
```

**Success Criteria**:
- [ ] All tests pass
- [ ] create_client() returns AsyncOpenAI
- [ ] format_messages() handles history
- [ ] parse_chunk() extracts text and tool calls

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
from typing import AsyncGenerator
from .types import AgentOptions, AssistantMessage, TextBlock, ToolUseBlock
from .utils import create_client, format_messages, parse_chunk

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

        async for chunk in response:
            block = parse_chunk(chunk)
            if block:
                current_message.content.append(block)
                yield current_message

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise
```

`any_agent/__init__.py`:
```python
"""Any-Agent SDK - Claude Agent SDK-style API for local LLMs"""
from .client import query
from .types import AgentOptions, TextBlock, ToolUseBlock, AssistantMessage

__all__ = [
    "query",
    "AgentOptions",
    "TextBlock",
    "ToolUseBlock",
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
        self.message_history: list[dict] = []
        self.turn_count = 0
        self.response_stream = None

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

    async def receive_messages(self) -> AsyncGenerator[TextBlock | ToolUseBlock, None]:
        """Stream individual blocks from response"""
        if not self.response_stream:
            raise RuntimeError("Call query() first")

        assistant_content = []

        async for chunk in self.response_stream:
            block = parse_chunk(chunk)
            if block:
                assistant_content.append(block)
                yield block

        # Add assistant response to history
        # Reconstruct full text
        full_text = ""
        for block in assistant_content:
            if isinstance(block, TextBlock):
                full_text += block.text

        self.message_history.append({
            "role": "assistant",
            "content": full_text
        })

        self.turn_count += 1

        # Check max turns
        if self.turn_count >= self.options.max_turns:
            logger.info(f"Reached max_turns ({self.options.max_turns})")
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
