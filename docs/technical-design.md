# Technical Design: Open Agent SDK

## Overview

Open Agent SDK provides a clean, streaming API for local/self-hosted LLMs via OpenAI-compatible endpoints.

**Core Goal**: Minimal wrapper around AsyncOpenAI that provides familiar patterns from claude-agent-sdk.

Note: While the SDK targets local/self-hosted endpoints, it also works with local OpenAI-compatible gateways (e.g., Ollama) that proxy requests to cloud models. In that setup, Open Agent SDK still points to the local gateway `base_url`; credentials and routing are handled by the gateway.

## Architecture

```
┌──────────────────────────────────────┐
│  User's Agent Code                   │
│  (Copy Editor, Market Analysis)      │
└───────────────┬──────────────────────┘
                │
                │ import query, Client, AgentOptions
                │
┌───────────────▼──────────────────────┐
│  any_agent/                          │
│  ├── client.py                       │
│  │   ├── query()      ◄── Single turn
│  │   └── Client       ◄── Multi-turn │
│  ├── types.py                        │
│  │   ├── AgentOptions                │
│  │   ├── TextBlock                   │
│  │   ├── ToolUseBlock                │
│  │   └── AssistantMessage            │
│  └── utils.py                        │
│      ├── create_client()             │
│      ├── format_messages()           │
│      └── parse_response()            │
└───────────────┬──────────────────────┘
                │
                │ AsyncOpenAI client
                │
┌───────────────▼──────────────────────┐
│  OpenAI Python SDK                   │
│  (AsyncOpenAI)                       │
└───────────────┬──────────────────────┘
                │
                │ HTTP streaming
                │
┌───────────────▼──────────────────────┐
│  Local Model Servers                 │
│  ├── LM Studio (localhost:1234)      │
│  ├── Ollama (localhost:11434)        │
│  ├── llama.cpp server                │
│  └── vLLM, Text Gen WebUI, etc.      │
└──────────────────────────────────────┘
```

## Core Components

### 1. types.py - Data Structures

**AgentOptions** - Configuration dataclass
```python
@dataclass
class AgentOptions:
    system_prompt: str
    model: str
    base_url: str
    max_turns: int = 1
    max_tokens: int | None = 4096  # Default 4096, None uses provider default
    temperature: float = 0.7
    api_key: str = "not-needed"
```

**Message Types** - Match claude-agent-sdk patterns
```python
from typing import Literal
from dataclasses import dataclass

@dataclass
class TextBlock:
    text: str
    type: Literal["text"] = "text"

@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: Literal["tool_use"] = "tool_use"

@dataclass
class ToolUseError:
    error: str
    raw_data: str | None = None
    type: Literal["tool_use_error"] = "tool_use_error"

@dataclass
class AssistantMessage:
    role: Literal["assistant"] = "assistant"
    content: list[TextBlock | ToolUseBlock | ToolUseError]
```

### 2. utils.py - OpenAI Client Helpers

**create_client()**
```python
def create_client(options: AgentOptions) -> AsyncOpenAI:
    """Create configured AsyncOpenAI client"""
    return AsyncOpenAI(
        base_url=options.base_url,
        api_key=options.api_key
    )
```

**format_messages()**
```python
def format_messages(
    system_prompt: str,
    user_prompt: str,
    history: list = None
) -> list[dict]:
    """Format messages for OpenAI API"""
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_prompt})
    return messages
```

**ToolCallAggregator.process_chunk()**
```python
def process_chunk(self, chunk) -> TextBlock | None:
    """Accumulate tool calls and emit only new text deltas."""
    delta = chunk.choices[0].delta

    new_text = self._extract_new_text(delta)  # Handles cumulative + incremental streams
    if new_text:
        return TextBlock(text=new_text)

    # Accumulate tool fragments keyed by their index
    for tool_delta in delta.tool_calls or []:
        pending = self.pending_tools.setdefault(tool_delta.index, {...})
        ...
```

**ToolCallAggregator.finalize_tools()**
```python
def finalize_tools(self) -> list[ToolUseBlock | ToolUseError]:
    """Parse accumulated JSON arguments and clear state for the next turn."""
    results = []
    for tool in self.pending_tools.values():
        if missing required fields:
            results.append(ToolUseError(...))
            continue

        try:
            payload = json.loads(tool["arguments_buffer"] or "{}")
        except json.JSONDecodeError as exc:
            results.append(ToolUseError(error=f"Invalid JSON: {exc}", ...))
        else:
            results.append(ToolUseBlock(..., input=payload))

    self.pending_tools.clear()
    self._text_accumulator = ""
    return results
```

### 3. client.py - Main API

**query() - Simple Single-Turn**
```python
async def query(
    prompt: str,
    options: AgentOptions
) -> AsyncGenerator[AssistantMessage, None]:
    """
    Simple query function for single-turn requests.

    Usage:
        result = query(prompt="Hello", options=options)
        async for msg in result:
            # Process messages
    """
    client = create_client(options)
    messages = format_messages(options.system_prompt, prompt)

    response = await client.chat.completions.create(
        model=options.model,
        messages=messages,
        max_tokens=options.max_tokens,
        temperature=options.temperature,
        stream=True
    )

    async for chunk in response:
        text_block = aggregator.process_chunk(chunk)
        if text_block:
            # Emit a fresh AssistantMessage for each delta so consumers
            # only see new content once.
            yield AssistantMessage(content=[text_block])

    for tool_block in aggregator.finalize_tools():
        yield AssistantMessage(content=[tool_block])
```

**Client - Multi-Turn Conversations**
```python
class Client:
    """
    Multi-turn conversation client with tool monitoring.

    Usage:
        async with Client(options) as client:
            await client.query("Hello")
            async for msg in client.receive_messages():
                # Process messages
    """

    def __init__(self, options: AgentOptions):
        self.options = options
        self.client = create_client(options)
        self.message_history = []
        self.turn_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup if needed
        pass

    async def query(self, prompt: str):
        """Send a query and start receiving messages"""
        messages = format_messages(
            self.options.system_prompt,
            prompt,
            self.message_history
        )

        self.response_stream = await self.client.chat.completions.create(
            model=self.options.model,
            messages=messages,
            max_tokens=self.options.max_tokens,
            temperature=self.options.temperature,
            stream=True
        )

    async def receive_messages(self) -> AsyncGenerator[TextBlock | ToolUseBlock, None]:
        """Stream messages from the model"""
        async for chunk in self.response_stream:
            text_block = self._aggregator.process_chunk(chunk)
            if text_block:
                assistant_blocks.append(text_block)
                yield text_block

        for tool_block in self._aggregator.finalize_tools():
            assistant_blocks.append(tool_block)
            yield tool_block

        self.turn_count += 1

        # Check max turns
        if self.turn_count >= self.options.max_turns:
            # Optionally raise StopIteration or set a flag
            pass
```

## Design Decisions

### 1. Streaming by Default

**Decision**: Always use streaming API

**Rationale**:
- Matches claude-agent-sdk behavior
- Better UX for long responses
- Tool monitoring during generation
- Can accumulate for non-streaming use case

### 2. Message Types

**Decision**: Match claude-agent-sdk message structure

**Rationale**:
- Familiar to developers migrating from claude-agent-sdk
- Clean abstraction over OpenAI's chunk format
- Easy to extend (add ToolResultBlock later)

**OpenAI Format** (what we receive):
```json
{
  "choices": [{
    "delta": {
      "content": "Hello",
      "tool_calls": [...]
    }
  }]
}
```

**Our Format** (what we yield):
```python
AssistantMessage(
    content=[
        TextBlock(text="Hello"),
        ToolUseBlock(name="search", input={...})
    ]
)
```

### 3. No Built-in Tool Execution

**Decision**: Don't execute tools automatically (Phase 1)

**Rationale**:
- Keep MVP simple
- User's agents log/monitor tools anyway (see market_analysis)
- Can add optional tool execution in Phase 2

**User's responsibility**:
```python
async for msg in client.receive_messages():
    if isinstance(msg, ToolUseBlock):
        # User logs it
        log_tool_use(msg.name, msg.input)
        # User can execute if they want
        # (we don't auto-execute)

        tool_result = await execute_tool(msg.name, msg.input)
        client.add_tool_result(
            tool_call_id=msg.id,
            content=tool_result  # dict/list will be JSON encoded automatically
        )

# Next assistant turn sees tool result in history
await client.query("Here is what the tool returned...")
```

### 4. Minimal State Management

**Decision**: Client tracks message history for multi-turn, nothing else

**Rationale**:
- User's agents don't need complex state
- Can add session persistence later if needed
- Keep it simple for MVP

### 5. No LiteLLM Dependency

**Decision**: Use AsyncOpenAI directly

**Rationale**:
- We only support OpenAI-compatible endpoints
- LiteLLM adds complexity we don't need
- Fewer dependencies = easier maintenance
- Direct control over streaming behavior

### 6. No Built-in Storage/Memory

**Decision**: Don't provide database, persistence, or memory management

**Rationale**:
- Agents have domain-specific storage needs
- Copy editor needs: issues, severity, trends
- Style analyzer needs: patterns, voice tracking
- Market analysis needs: comp titles, research
- Forcing a schema would limit agents

**What we provide instead**: Conversation primitives

```python
class Client:
    @property
    def history(self) -> list[dict]:
        """Agent can read full history to store"""
        return self.message_history.copy()

    @property
    def turn_metadata(self) -> dict:
        """Agent can track conversation state"""
        return {
            "turn_count": self.turn_count
        }
```

**Agent's responsibility**:
```python
from any_agent import Client, AgentOptions
from my_agent.database import MyCustomDatabase

class MyAgent:
    def __init__(self):
        self.db = MyCustomDatabase("data.db")  # Agent controls schema

    async def run(self):
        async with Client(options) as client:
            # ... conversation ...

            # Agent stores what it needs
            self.db.save_conversation(client.history)
            self.db.save_metadata(client.turn_metadata)
```

**Future possibility**: Optional `any_agent.extras.storage` for basic conversation storage:
- Install separately: `pip install open-agent-sdk[storage]`
- Simple schema for developers who don't need custom
- But NOT in core - keeps main package minimal

### 7. Tool-Calling Streaming Semantics

OpenAI-style streaming delivers tool calls incrementally. Arguments are streamed as partial JSON strings over multiple chunks, and metadata like `id` and `function.name` can appear in separate chunks. To handle this robustly:

**What streaming chunks look like**
```json
// Chunk 1
{
  "choices": [{
    "delta": {
      "tool_calls": [
        {"index": 0, "id": "call_abc", "function": {"name": "search", "arguments": "{\"q\": \"par"}}
      ]
    }
  }]
}
// Chunk 2
{
  "choices": [{
    "delta": {
      "tool_calls": [
        {"index": 0, "function": {"arguments": "is\"}"}}
      ]
    }
  }]
}
// Final chunk
{
  "choices": [{"finish_reason": "tool_calls"}]
}
```

**Aggregation strategy (MVP)**
- Maintain an in-memory map keyed by `index` for the current assistant turn: `pending_tools[index] = { id, name, arguments_buffer }`.
- For each chunk:
  - If `delta.content` exists, yield text as usual.
  - If `delta.tool_calls` exists, for each entry:
    - Use `tc.index` to select/update the correct pending tool.
    - Set `id` and `function.name` when they first appear.
    - Append `function.arguments` to `arguments_buffer` when present (it is a partial JSON string).
- Finalization:
  - When `finish_reason == "tool_calls"` (or after the stream ends), finalize each pending tool:
    - Attempt to parse `arguments_buffer` as JSON. If parsing fails, keep buffering until the assistant turn ends; as a last resort, log a warning and skip emitting the tool call for this turn (to avoid emitting malformed `input`).
    - Emit a single completed `ToolUseBlock` per tool with `id`, `name`, and parsed `input: dict`.
- Emission policy (MVP): emit only completed `ToolUseBlock`s. Do not emit partial tool deltas; this keeps the interface aligned with claude-agent-sdk-style blocks.

**Provider differences and fallbacks**
- Not all local endpoints implement tool/function calling. In those cases, only `TextBlock`s will be produced.
- Some providers emit `id` late or omit it; rely primarily on `index` for aggregation, set `id` once available.
- Some providers may end the stream with `finish_reason == "stop"` even when tool calls were emitted; in this case, finalize pending tools at end-of-stream.

**Error handling**
- If `arguments_buffer` never forms valid JSON by stream end, emit a `ToolUseError` block with the error message and raw buffer data
- If `id` or `name` are missing, emit a `ToolUseError` block describing what's missing
- User code can check for `isinstance(block, ToolUseError)` to handle tool parsing failures
- `ToolUseError` blocks are NOT preserved in message history (only valid tool calls are)
- Keep aggregation state per assistant turn; clear state once finalized

**Future enhancement options**
- Add an optional “tool delta” event (e.g., `ToolUseDelta`) for real-time monitoring of partial arguments.
- Provide a lenient JSON fixer for common truncation issues, gated behind an option (e.g., `allow_lenient_tool_json`).

## Implementation Complexity

**Total LOC estimate**: ~300-400 lines

**Breakdown**:
- types.py: ~50 lines (dataclasses)
- utils.py: ~100 lines (client creation, parsing)
- client.py: ~150 lines (query + Client class)
- __init__.py: ~10 lines (exports)
- Tests: ~200 lines

## Error Handling

**Connection errors**:
- Let AsyncOpenAI exceptions bubble up
- User can catch and retry

**Malformed responses**:
- parse_chunk() returns None if unparseable
- Log warning but continue streaming

**Max turns exceeded**:
- Client tracks turn_count
- Stop iteration when max_turns reached

## Future Enhancements (Out of Scope for MVP)

- ⏳ Automatic tool execution
- ⏳ Context window management
- ⏳ Message history trimming
- ⏳ Session persistence (SQLite)
- ⏳ Retry logic with exponential backoff
- ⏳ Token counting/usage tracking
- ⏳ Richer tool-calling semantics and function definition mapping across providers

## Testing Strategy

**Unit Tests**:
- test_types.py - Dataclass construction
- test_utils.py - Message formatting, chunk parsing
- test_client.py - query() and Client behavior

**Integration Tests**:
- Test against real LM Studio (if running)
- Test against Ollama (if running)
- Mock AsyncOpenAI for CI/CD

**Validation**:
- Port copy_editor agent
- Verify < 5 lines changed
- Compare output quality

## Dependencies

Minimal external dependencies:

```toml
[project]
dependencies = [
    "openai>=1.0.0",     # AsyncOpenAI client
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]
```

Optional:
- pydantic for stricter types (can use dataclasses instead)

## Success Metrics

The technical design succeeds when:

1. ✅ query() matches claude-agent-sdk.query() behavior
2. ✅ Client matches claude-agent-sdk.ClaudeSDKClient behavior
3. ✅ Copy editor agent ports with < 5 lines changed
4. ✅ Total codebase < 500 lines (excluding tests)
5. ✅ Works with LM Studio, Ollama, llama.cpp
6. ✅ No crashes on malformed responses
7. ✅ Clean async/await patterns throughout
