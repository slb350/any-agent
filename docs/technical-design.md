# Technical Design: Any-Agent SDK

## Overview

Any-Agent SDK provides a claude-agent-sdk-compatible API for local/self-hosted LLMs via OpenAI-compatible endpoints.

**Core Goal**: Minimal wrapper around AsyncOpenAI that provides familiar patterns from claude-agent-sdk.

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
    max_tokens: int = 8000
    temperature: float = 0.7
    api_key: str = "not-needed"
```

**Message Types** - Match claude-agent-sdk patterns
```python
@dataclass
class TextBlock:
    text: str
    type: str = "text"

@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict
    type: str = "tool_use"

@dataclass
class AssistantMessage:
    role: str = "assistant"
    content: list[TextBlock | ToolUseBlock]
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

**parse_chunk()**
```python
def parse_chunk(chunk) -> TextBlock | ToolUseBlock | None:
    """Parse streaming chunk into message blocks"""
    # Extract delta from chunk.choices[0].delta
    # Convert to TextBlock or ToolUseBlock
    # Return None if no content
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

    # Stream and parse chunks
    current_message = AssistantMessage(content=[])

    async for chunk in response:
        block = parse_chunk(chunk)
        if block:
            current_message.content.append(block)
            yield current_message
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
            block = parse_chunk(chunk)
            if block:
                yield block

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
- ⏳ Support for function calling (OpenAI format)

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
