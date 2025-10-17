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
│  open_agent/                          │
│  ├── client.py                       │
│  │   ├── query()      ◄── Single turn
│  │   └── Client       ◄── Multi-turn │
│  ├── types.py                        │
│  │   ├── AgentOptions                │
│  │   ├── TextBlock                   │
│  │   ├── ToolUseBlock                │
│  │   ├── ToolResultBlock             │
│  │   └── AssistantMessage            │
│  ├── tools.py                        │
│  │   ├── @tool       ◄── Decorator   │
│  │   ├── Tool        ◄── Definition  │
│  │   └── Schema conversion           │
│  ├── context.py      ◄── v0.2.3      │
│  │   ├── estimate_tokens()           │
│  │   └── truncate_messages()         │
│  ├── hooks.py        ◄── v0.2.4      │
│  │   ├── PreToolUseEvent             │
│  │   ├── PostToolUseEvent            │
│  │   ├── UserPromptSubmitEvent       │
│  │   └── HookDecision                │
│  └── utils.py                        │
│      ├── create_client()             │
│      ├── format_messages()           │
│      ├── format_tools()              │
│      └── ToolCallAggregator          │
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
    tools: list[Tool] = field(default_factory=list)  # v0.2.2+
    auto_execute_tools: bool = False                 # v0.3.0+ Auto tool execution
    max_tool_iterations: int = 5                     # v0.3.0+ Safety limit for auto mode
    max_turns: int = 1
    max_tokens: int | None = 4096  # Default 4096, None uses provider default
    temperature: float = 0.7
    timeout: float = 60.0
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
class ToolResultBlock:
    """Tool execution result (v0.2.2+)"""
    tool_use_id: str
    content: str | dict[str, Any] | list[Any]
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"

@dataclass
class AssistantMessage:
    role: Literal["assistant"] = "assistant"
    content: list[TextBlock | ToolUseBlock | ToolUseError | ToolResultBlock]
```

### 2. tools.py - Tool System (v0.2.2+)

**Tool Definition**
```python
@dataclass
class Tool:
    """Tool definition for OpenAI-compatible function calling"""
    name: str
    description: str
    input_schema: dict[str, type] | dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[Any]]

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _convert_schema_to_openai(self.input_schema)
            }
        }

    async def execute(self, arguments: dict[str, Any]) -> Any:
        """Execute the tool with given arguments"""
        return await self.handler(arguments)
```

**@tool Decorator**
```python
def tool(
    name: str,
    description: str,
    input_schema: dict[str, type] | dict[str, Any]
) -> Callable:
    """
    Decorator for defining tools with OpenAI-compatible function calling.

    Examples:
        # Simple schema with Python types
        @tool("get_weather", "Get weather", {"location": str, "units": str})
        async def get_weather(args):
            return {"temp": 72, "conditions": "sunny"}

        # Sync handlers automatically wrapped
        @tool("shout", "Uppercase", {"text": str})
        def shout(args):
            return args["text"].upper()

        # Optional parameters
        @tool("search", "Search", {
            "query": str,
            "limit": {"type": "integer", "default": 10}
        })
        async def search(args):
            return {"results": []}
    """
    def decorator(handler):
        # Auto-wrap sync handlers
        if inspect.iscoroutinefunction(handler):
            async_handler = handler
        else:
            async def async_wrapper(arguments):
                return handler(arguments)
            async_handler = async_wrapper

        return Tool(name, description, input_schema, async_handler)
    return decorator
```

**Schema Conversion**
```python
def _convert_schema_to_openai(input_schema: dict) -> dict:
    """
    Convert input schema to OpenAI parameters format.

    Handles:
    - Simple Python types: {"param": str} → {"type": "string"}
    - Full JSON Schema (passed through unchanged)
    - Optional parameters via:
      - "default" field
      - "required": False
      - "optional": True (convenience flag)
    """
    # Check if already JSON Schema
    if "type" in input_schema and "properties" in input_schema:
        return input_schema

    # Convert simple type mapping
    properties = {}
    required_params = []

    for param_name, param_type in input_schema.items():
        if isinstance(param_type, type):
            properties[param_name] = _type_to_json_schema(param_type)
            required_params.append(param_name)
        elif isinstance(param_type, dict):
            # Extract optionality flags
            schema = dict(param_type)
            optional = schema.pop("optional", False)
            required = schema.pop("required", None)
            properties[param_name] = schema

            # Determine if required
            if required is True:
                required_params.append(param_name)
            elif required is False or optional or "default" in schema:
                pass  # Optional
            else:
                required_params.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required_params
    }
```

### 3. utils.py - OpenAI Client Helpers

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

**format_tools()** (v0.2.2+)
```python
def format_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """
    Convert Tool instances to OpenAI function calling format.

    Returns:
        List of tool definitions in format:
        [
            {
                "type": "function",
                "function": {
                    "name": "tool_name",
                    "description": "Tool description",
                    "parameters": {...}
                }
            },
            ...
        ]
    """
    return [tool.to_openai_format() for tool in tools]
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

### 4. client.py - Main API

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

    # Build request parameters
    request_params = {
        "model": options.model,
        "messages": messages,
        "max_tokens": options.max_tokens,
        "temperature": options.temperature,
        "stream": True,
    }

    # Add tools if configured (v0.2.2+)
    if options.tools:
        request_params["tools"] = format_tools(options.tools)

    response = await client.chat.completions.create(**request_params)

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
        # Automatic tool execution (recommended, v0.3.0+)
        options = AgentOptions(
            system_prompt="...",
            model="...",
            base_url="...",
            tools=[add, multiply],
            auto_execute_tools=True  # Tools execute automatically
        )
        async with Client(options) as client:
            await client.query("What's 25 + 17?")
            async for msg in client.receive_messages():
                # Tools execute automatically, just process responses
                if isinstance(msg, TextBlock):
                    print(msg.text)

        # Manual tool execution (advanced)
        async with Client(options) as client:
            await client.query("Hello")
            async for msg in client.receive_messages():
                if isinstance(msg, ToolUseBlock):
                    result = await tool.execute(msg.input)
                    await client.add_tool_result(msg.id, result, msg.name)
                    await client.query("")  # Continue conversation
    """

    def __init__(self, options: AgentOptions):
        self.options = options
        self.client = create_client(options)
        self.message_history = []
        self.turn_count = 0
        self._tool_registry = {}  # v0.3.0+ Built at init, validates no duplicates

        # Build tool registry with duplicate validation
        if options.tools:
            for tool in options.tools:
                if tool.name in self._tool_registry:
                    raise ValueError(f"Duplicate tool name: {tool.name}")
                self._tool_registry[tool.name] = tool

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

    async def _continue_turn(self):
        """
        Continue conversation without re-firing hooks (v0.3.0+).

        Used internally by auto-execution to feed tool results back
        without triggering UserPromptSubmit hooks.
        """
        messages = format_messages(
            self.options.system_prompt,
            "",  # Empty prompt for continuation (mirrors manual query("") pattern)
            self.message_history
        )
        # ... rest of API call

    async def _receive_once(self) -> list[TextBlock | ToolUseBlock | ToolUseError]:
        """
        Consume stream once and return all blocks (v0.3.0+).

        Helper for auto-execution to ensure stream is fully consumed
        before executing tools and continuing.
        """
        blocks = []
        async for chunk in self.response_stream:
            text_block = self._aggregator.process_chunk(chunk)
            if text_block:
                blocks.append(text_block)

        for tool_block in self._aggregator.finalize_tools():
            blocks.append(tool_block)

        return blocks

    async def _auto_execute_loop(self) -> AsyncGenerator[TextBlock | ToolUseBlock | ToolUseError, None]:
        """
        Automatic tool execution loop (v0.3.0+).

        Executes tools and feeds results back to the model automatically
        until a text-only response or max_tool_iterations is reached.

        Safety limits:
        - max_tool_iterations (default 5) prevents infinite loops
        - Unknown tools yield ToolUseError for monitoring
        - Execution failures yield ToolUseError for monitoring

        Error handling:
        - Unknown tools: Pass error dict to add_tool_result(), yield ToolUseError
        - Execution failures: Pass error dict to add_tool_result(), yield ToolUseError
        """
        iteration_count = 0

        while iteration_count < self.options.max_tool_iterations:
            blocks = await self._receive_once()

            # Yield all blocks to user
            for block in blocks:
                yield block

            # Check if any tools need execution
            tool_blocks = [b for b in blocks if isinstance(b, ToolUseBlock)]
            if not tool_blocks:
                break  # No tools, we're done

            # Execute all tools in this batch
            for tool_block in tool_blocks:
                tool_name = tool_block.name
                tool = self._tool_registry.get(tool_name)

                if not tool:
                    # Unknown tool - yield error and pass to model
                    error_payload = {
                        "error": f"Tool '{tool_name}' not found in registry",
                        "tool": tool_name
                    }
                    yield ToolUseError(
                        error=f"Tool '{tool_name}' not found in registry",
                        raw_data=str(tool_block.input)
                    )
                    await self.add_tool_result(
                        tool_call_id=tool_block.id,
                        content=error_payload,
                        name=tool_name
                    )
                    continue

                try:
                    result = await tool.execute(tool_block.input)
                    await self.add_tool_result(
                        tool_call_id=tool_block.id,
                        content=result,
                        name=tool_name
                    )
                except Exception as exc:
                    # Execution error - yield error and pass to model
                    error_payload = {
                        "error": str(exc),
                        "tool": tool_name
                    }
                    yield ToolUseError(
                        error=str(exc),
                        raw_data=str(tool_block.input)
                    )
                    await self.add_tool_result(
                        tool_call_id=tool_block.id,
                        content=error_payload,
                        name=tool_name
                    )

            # Continue conversation with tool results
            await self._continue_turn()
            iteration_count += 1

    async def receive_messages(self) -> AsyncGenerator[TextBlock | ToolUseBlock | ToolUseError, None]:
        """
        Stream messages from the model.

        In auto_execute_tools mode (v0.3.0+), tools execute automatically
        and results feed back until text-only response or max_tool_iterations.

        In manual mode, yields ToolUseBlock for user to execute.
        """
        if self.options.auto_execute_tools:
            # Automatic mode - orchestrate tool execution
            async for block in self._auto_execute_loop():
                yield block
        else:
            # Manual mode - yield blocks as they arrive
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

### 5. context.py - Context Management Utilities (v0.2.3+)

**Philosophy**: Opt-in utilities for manual history management, NOT automatic compaction.

**Why manual?**
- Domain-specific needs vary significantly (copy editor ≠ research agent ≠ code reviewer)
- Token counting accuracy varies by model family (Qwen, Llama, Mistral have different tokenizers)
- Risk of silently breaking context by removing important messages or tool chains
- Natural model limits exist regardless of compaction strategy (8k-32k tokens)
- Users understand their domain better than generic heuristics

**estimate_tokens()**
```python
def estimate_tokens(
    messages: list[dict[str, Any]],
    model: str = "gpt-3.5-turbo"
) -> int:
    """
    Estimate token count for message list.

    Uses tiktoken if available (~90-95% accurate for GPT models,
    ~70-85% accurate for other families), otherwise falls back to
    character-based approximation (~75-85% accurate).

    Always include 10-20% safety margin when checking limits.

    Examples:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful"},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> tokens = estimate_tokens(messages)
        >>> if tokens > 28000:  # 85% of 32k limit
        ...     print("Need to truncate!")
    """
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)
        # Count tokens with message overhead
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # Message formatting overhead
            # ... count content tokens
        return num_tokens
    except ImportError:
        # Fallback: 1 token ≈ 4 characters
        total_chars = sum(len(str(v)) for m in messages for v in m.values())
        return total_chars // 4
```

**truncate_messages()**
```python
def truncate_messages(
    messages: list[dict[str, Any]],
    keep: int = 10,
    preserve_system: bool = True
) -> list[dict[str, Any]]:
    """
    Truncate message history, keeping recent messages.

    Simple truncation that preserves system prompt and keeps
    last N messages. Does NOT attempt to preserve tool chains
    or identify important context.

    Args:
        messages: List of message dicts (OpenAI format)
        keep: Number of recent messages to keep (default: 10)
        preserve_system: Keep system message if present (default: True)

    Returns:
        Truncated message list (new list, original unchanged)

    Examples:
        >>> # Manual truncation at natural breakpoint
        >>> if estimate_tokens(client.history) > 28000:
        ...     client.message_history = truncate_messages(
        ...         client.history, keep=10
        ...     )

        >>> # Clear history except system prompt
        >>> client.message_history = truncate_messages(
        ...     client.history, keep=0
        ... )
    """
    if not messages or len(messages) <= keep:
        return messages.copy()

    has_system = (
        preserve_system and
        messages and
        messages[0].get("role") == "system"
    )

    if has_system:
        return [messages[0]] + messages[-keep:] if keep > 0 else [messages[0]]
    else:
        return messages[-keep:] if keep > 0 else []
```

**Recommended Usage Patterns**:

1. **Stateless Agents** (Best for single-task agents):
```python
for task in tasks:
    async with Client(options) as client:
        await client.query(task)
        # Client disposed, fresh context for next task
```

2. **Manual Truncation** (At natural breakpoints):
```python
from open_agent.context import truncate_messages

async with Client(options) as client:
    for task in tasks:
        await client.query(task)
        # Truncate after each major milestone
        client.message_history = truncate_messages(client.history, keep=5)
```

3. **Token Budget Monitoring**:
```python
from open_agent.context import estimate_tokens, truncate_messages

MAX_TOKENS = 28000  # 85% of 32k limit

async with Client(options) as client:
    while True:
        await client.query(user_input)
        if estimate_tokens(client.history) > MAX_TOKENS:
            client.message_history = truncate_messages(client.history, keep=10)
```

4. **External Memory (RAG-lite)**:
```python
# Store important facts externally, keep conversation context small
knowledge_base = {}
async with Client(options) as client:
    # Research phase
    await client.query("Research topic X")
    knowledge_base["topic_x"] = extract_facts(response)

    # Clear history
    client.message_history = truncate_messages(client.history, keep=0)

    # Analysis phase with stored knowledge
    await client.query(f"Based on: {knowledge_base}, analyze Y")
```

**Optional Dependency**:
```bash
# Better token estimation with tiktoken
pip install open-agent-sdk[context]

# Without tiktoken, uses character-based fallback
pip install open-agent-sdk
```

**Testing**:
- 17 comprehensive tests in `tests/test_context.py`
- Tests for tiktoken and fallback modes
- Edge cases (keep=0, empty messages, preserve_system=False)
- Integration tests with realistic conversation workflows

**Documentation**:
- `examples/context_management.py` - 4 detailed usage patterns
- `docs/context-utilities-design.md` - Full design rationale

### 6. hooks.py - Lifecycle Hooks System (v0.2.4+)

**Philosophy**: "Claude parity, but local-first" - familiar lifecycle hooks without CLI subprocess complexity.

**Why hooks?**
- **Security gates**: Block dangerous operations before they execute
- **Audit logging**: Track all tool executions for compliance
- **Input validation**: Sanitize user prompts before processing
- **Monitoring**: Observe agent behavior in production
- **Control flow**: Modify tool inputs or redirect operations

**Design principles**:
- Pythonic over protocol - dataclasses and coroutines, not JSON messages
- Inline execution - hooks run synchronously on the event loop
- Explicit control - users decide what to monitor and when to intervene
- No blocking I/O - spawn tasks for heavy work
- Short-circuit behavior - first non-None decision wins

**Core Types**

```python
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

@dataclass
class HookDecision:
    """
    Decision from a hook about whether to continue execution.

    Return HookDecision from a hook to:
    - Block execution (continue_=False)
    - Modify inputs (modified_input, modified_prompt)
    - Document reasoning (reason)

    Return None from a hook to allow by default.
    Raise an exception to abort entirely.
    """
    continue_: bool = True
    modified_input: dict[str, Any] | None = None
    modified_prompt: str | None = None
    reason: str | None = None

@dataclass
class PreToolUseEvent:
    """Fired before a tool is executed (or yielded to user)."""
    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str
    history: list[dict[str, Any]]

@dataclass
class PostToolUseEvent:
    """Fired after a tool result is added to history."""
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: Any
    tool_use_id: str
    history: list[dict[str, Any]]

@dataclass
class UserPromptSubmitEvent:
    """Fired when user submits a prompt (before API call)."""
    prompt: str
    history: list[dict[str, Any]]

# Type alias for hook handlers
HookHandler = Callable[[HookEvent], Awaitable[HookDecision | None]]
```

**Hook Constants**

```python
# Hook names for registration
HOOK_PRE_TOOL_USE = "pre_tool_use"
HOOK_POST_TOOL_USE = "post_tool_use"
HOOK_USER_PROMPT_SUBMIT = "user_prompt_submit"
```

**Hook Registration**

Hooks are registered in `AgentOptions`:

```python
from open_agent import (
    AgentOptions, Client,
    PreToolUseEvent, HookDecision,
    HOOK_PRE_TOOL_USE, HOOK_POST_TOOL_USE
)

async def security_gate(event: PreToolUseEvent) -> HookDecision | None:
    """Block writes to system directories."""
    if event.tool_name == "file_writer":
        path = event.tool_input.get("path", "")
        if "/etc/" in path or "/sys/" in path:
            return HookDecision(
                continue_=False,
                reason=f"Cannot write to system path: {path}"
            )
    return None  # Allow by default

async def audit_logger(event: PostToolUseEvent) -> None:
    """Log all tool executions."""
    logging.info(f"Tool executed: {event.tool_name} -> {event.tool_result}")
    return None  # Observational only

options = AgentOptions(
    system_prompt="You are a helpful assistant",
    model="qwen2.5-32b-instruct",
    base_url="http://localhost:1234/v1",
    tools=[file_writer, web_search],
    hooks={
        HOOK_PRE_TOOL_USE: [security_gate],
        HOOK_POST_TOOL_USE: [audit_logger],
    }
)
```

**Hook Execution Flow**

Hooks are executed sequentially with short-circuit behavior:

```python
async def _run_hooks(
    self,
    hook_name: str,
    event: HookEvent
) -> HookDecision | None:
    """
    Run registered hooks for the given event.

    Returns first non-None HookDecision, or None if all allow.
    Raises if any hook raises an exception.
    """
    if not self.options.hooks:
        return None

    handlers = self.options.hooks.get(hook_name, [])
    for handler in handlers:
        decision = await handler(event)
        if decision is not None:
            return decision  # Short-circuit on first decision

    return None  # All hooks passed (allow)
```

**Integration Points**

**1. PreToolUse Hook** - Fires before yielding ToolUseBlock

```python
# In Client.receive_messages()
for tool_block in self._aggregator.finalize_tools():
    # Fire PreToolUse hook
    event = PreToolUseEvent(
        tool_name=tool_block.name,
        tool_input=tool_block.input,
        tool_use_id=tool_block.id,
        history=self.message_history.copy()
    )

    decision = await self._run_hooks(HOOK_PRE_TOOL_USE, event)

    if decision and not decision.continue_:
        # Hook blocked execution
        raise RuntimeError(f"Tool use blocked: {decision.reason}")

    if decision and decision.modified_input:
        # Hook modified input
        tool_block = ToolUseBlock(
            id=tool_block.id,
            name=tool_block.name,
            input=decision.modified_input
        )

    assistant_blocks.append(tool_block)
    yield tool_block
```

**2. PostToolUse Hook** - Fires after adding tool result to history

```python
# In Client.add_tool_result()
async def add_tool_result(
    self,
    tool_call_id: str,
    content: Any,
    name: str | None = None
) -> None:
    """Add tool result to conversation history."""
    # Add to history
    self.message_history.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": name,
        "content": json.dumps(content) if isinstance(content, dict) else str(content)
    })

    # Fire PostToolUse hook (observational)
    if self.options.hooks:
        # Find original tool call from history
        tool_input = self._find_tool_input(tool_call_id, name)

        event = PostToolUseEvent(
            tool_name=name or "unknown",
            tool_input=tool_input,
            tool_result=content,
            tool_use_id=tool_call_id,
            history=self.message_history.copy()
        )

        await self._run_hooks(HOOK_POST_TOOL_USE, event)
```

**3. UserPromptSubmit Hook** - Fires before sending prompt to API

```python
# In Client.query()
async def query(self, prompt: str) -> None:
    """Send a query to the model."""
    # Fire UserPromptSubmit hook
    if self.options.hooks:
        event = UserPromptSubmitEvent(
            prompt=prompt,
            history=self.message_history.copy()
        )

        decision = await self._run_hooks(HOOK_USER_PROMPT_SUBMIT, event)

        if decision and not decision.continue_:
            raise RuntimeError(f"Prompt blocked: {decision.reason}")

        if decision and decision.modified_prompt:
            prompt = decision.modified_prompt

    # Build messages and send to API
    self.message_history.append({"role": "user", "content": prompt})
    # ... rest of query logic
```

**Works with Both Client and query()**

Hooks work with both multi-turn Client and single-turn query():

```python
# Multi-turn Client
async with Client(options) as client:
    await client.query("Write to /etc/config")  # UserPromptSubmit fires
    async for block in client.receive_messages():
        if isinstance(block, ToolUseBlock):  # PreToolUse fires
            result = await tool.execute(block.input)
            await client.add_tool_result(block.id, result)  # PostToolUse fires

# Single-turn query()
async for msg in query("Write to /etc/config", options):
    # Same hooks fire inline
    pass
```

**Hook Patterns**

**Pattern 1: Security Gates**

```python
async def security_gate(event: PreToolUseEvent) -> HookDecision | None:
    """Block dangerous operations."""
    if event.tool_name == "delete_file":
        return HookDecision(
            continue_=False,
            reason="Delete operations require manual approval"
        )
    return None
```

**Pattern 2: Input Redirection**

```python
async def redirect_to_sandbox(event: PreToolUseEvent) -> HookDecision | None:
    """Redirect file operations to sandbox."""
    if event.tool_name == "file_writer":
        path = event.tool_input.get("path", "")
        if not path.startswith("/tmp/"):
            safe_path = f"/tmp/sandbox/{path.lstrip('/')}"
            return HookDecision(
                modified_input={"path": safe_path, "content": event.tool_input.get("content", "")},
                reason="Redirected to sandbox"
            )
    return None
```

**Pattern 3: Audit Logging**

```python
audit_log = []

async def audit_logger(event: PostToolUseEvent) -> None:
    """Log all tool executions for compliance."""
    audit_log.append({
        "timestamp": datetime.now(),
        "tool": event.tool_name,
        "input": event.tool_input,
        "result": str(event.tool_result)[:100]
    })
    return None  # Observational only
```

**Pattern 4: Input Sanitization**

```python
async def sanitize_input(event: UserPromptSubmitEvent) -> HookDecision | None:
    """Add safety instructions to risky prompts."""
    if "delete" in event.prompt.lower() or "write" in event.prompt.lower():
        safe_prompt = event.prompt + " (Please confirm this is safe before proceeding)"
        return HookDecision(
            modified_prompt=safe_prompt,
            reason="Added safety warning"
        )
    return None
```

**Breaking Change**

Making `Client.add_tool_result()` async was required for PostToolUse hook support:

```python
# Old (v0.2.3 and earlier)
client.add_tool_result(tool_id, result)

# New (v0.2.4+)
await client.add_tool_result(tool_id, result)
```

**Testing**

- 14 comprehensive tests in `tests/test_hooks.py` (118 total tests)
- PreToolUse tests: allow, block, modify input
- PostToolUse tests: observe results, logging
- UserPromptSubmit tests: allow, block, modify prompt
- Multiple hooks sequencing
- Exception handling
- Event data validation
- Works with both Client and query() contexts

**Examples**

`examples/hooks_example.py` demonstrates 4 comprehensive patterns:
1. Security gates - blocking/redirecting dangerous operations
2. Audit logging - compliance tracking
3. Input sanitization - validation and safety
4. Combined hooks - layered control

**Design Trade-offs**

**vs Claude SDK**:
- Claude SDK: Hooks communicate with CLI subprocess via control protocol (JSON messages)
- Our SDK: Hooks run inline on event loop (Pythonic coroutines)
- Trade-off: No subprocess overhead, simpler integration, but no CLI interception

**Sequential execution**:
- Multiple hooks run in order, first non-None decision wins
- Allows layered control (security → validation → logging)
- Predictable behavior with clear short-circuit semantics

**Observational PostToolUse**:
- PostToolUse fires after tool already executed
- Can log/observe but can't block (tool already ran)
- Design choice: Tool execution control happens in PreToolUse

**No matcher DSL**:
- No pattern matching or filtering syntax
- Handlers branch inside coroutines with if/elif
- Simpler implementation, more explicit control

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

### 3. Tool System Design (v0.2.2+)

**Decision**: Provide `@tool` decorator with direct OpenAI format (not MCP)

**Rationale**:
- Target audience uses OpenAI-compatible endpoints (LM Studio, Ollama, etc.)
- These servers already understand OpenAI's native tools format
- No need for MCP middleware layer - simpler and lighter
- Can add MCP bridge later as optional feature

**Tool Definition**:
```python
from open_agent import tool, Client, AgentOptions

@tool("get_weather", "Get current weather", {"location": str, "units": str})
async def get_weather(args):
    return {
        "temperature": 72,
        "conditions": "sunny",
        "units": args["units"]
    }

# Register with agent
options = AgentOptions(
    system_prompt="You are a helpful assistant",
    model="qwen2.5-32b-instruct",
    base_url="http://localhost:1234/v1",
    tools=[get_weather]  # Tools sent to API automatically
)
```

**Execution Pattern** (user-controlled):
```python
async for msg in client.receive_messages():
    if isinstance(msg, ToolUseBlock):
        # User controls execution
        result = await get_weather.execute(msg.input)

        # Send result back to agent
        client.add_tool_result(
            tool_call_id=msg.id,
            content=result
        )

        # Continue conversation
        await client.query("")
```

**Design Features**:
- **Sync handler support**: Auto-wraps `def` functions to `async def`
- **Optional parameters**: Support via `default`, `required: False`, or `optional: True`
- **Schema flexibility**: Simple Python types or full JSON Schema
- **Type safety**: Automatic conversion with validation
- **Clean API**: Matches Claude SDK's `@tool` ergonomics

### 4. Automatic Tool Execution (Opt-In, v0.3.0+)

**Decision**: Provide both automatic and manual tool execution modes

**Rationale**:
- **Automatic mode** (`auto_execute_tools=True`) - Default "pit of success" path
  - Tools execute automatically without boilerplate
  - Simplifies common use cases significantly
  - Safety limits prevent runaway loops (`max_tool_iterations`)
  - ToolUseError blocks allow monitoring of failures
- **Manual mode** (default, `auto_execute_tools=False`) - Advanced control
  - User explicitly executes each tool
  - Required for approval workflows
  - Custom execution logic per agent
  - Backward compatible with existing code

**Automatic Mode Example**:
```python
options = AgentOptions(
    system_prompt="You are a helpful assistant",
    model="qwen2.5-32b-instruct",
    base_url="http://localhost:1234/v1",
    tools=[add, multiply],
    auto_execute_tools=True,      # Enable automatic execution
    max_tool_iterations=10         # Safety limit
)

async with Client(options) as client:
    await client.query("What's 25 + 17, then multiply by 3?")
    async for block in client.receive_messages():
        # Tools execute automatically in the background
        if isinstance(block, TextBlock):
            print(block.text)  # Just get the final answer
```

**Manual Mode Example** (for custom execution logic):
```python
async with Client(options) as client:
    await client.query("What's 25 + 17?")
    async for block in client.receive_messages():
        if isinstance(block, ToolUseBlock):
            # User controls execution
            result = await tool.execute(block.input)
            await client.add_tool_result(block.id, result, block.name)
            await client.query("")  # Continue conversation
```

**Design Features**:
- Default `False` for backward compatibility
- Tool registry built at init with duplicate validation
- Error handling: Unknown tools and execution failures yield `ToolUseError`
- Hooks work identically in both modes (PreToolUse/PostToolUse fire correctly)
- Clean separation: auto mode via `_auto_execute_loop()`, manual via original flow

### 5. Minimal State Management

**Decision**: Client tracks message history for multi-turn, nothing else

**Rationale**:
- User's agents don't need complex state
- Can add session persistence later if needed
- Keep it simple for MVP

### 6. No LiteLLM Dependency

**Decision**: Use AsyncOpenAI directly

**Rationale**:
- We only support OpenAI-compatible endpoints
- LiteLLM adds complexity we don't need
- Fewer dependencies = easier maintenance
- Direct control over streaming behavior

### 7. No Built-in Storage/Memory

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
from open_agent import Client, AgentOptions
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

**Future possibility**: Optional `open_agent.extras.storage` for basic conversation storage:
- Install separately: `pip install open-agent-sdk[storage]`
- Simple schema for developers who don't need custom
- But NOT in core - keeps main package minimal

### 8. Manual Context Management, Not Automatic (v0.2.3+)

**Decision**: Provide opt-in utilities (`estimate_tokens`, `truncate_messages`), NOT automatic compaction

**Rationale**:
- **Domain-specific strategies**: Copy editing agents need different truncation than research agents
  - Copy editor: Keep recent edits + system rules
  - Research agent: External DB + summarized facts
  - Code reviewer: Preserve file context chains
- **Token counting inaccuracy**: Each model family has different tokenizers
  - GPT models: tiktoken ~90-95% accurate
  - Llama/Qwen/Mistral: ~70-85% accurate (different tokenizers)
  - 10-20% safety margins required
- **Context breaking risk**: Silently removing messages can:
  - Break tool call/result pairs mid-conversation
  - Remove critical information the model needs
  - Cause confusion when context suddenly changes
- **Natural limits exist**: Model context windows are fixed (8k-32k)
  - Compaction doesn't bypass fundamental limits
  - If you need massive context, use RAG/vector DB
- **User knowledge**: Users understand their domain better than generic heuristics

**What we provide instead**:
```python
from open_agent.context import estimate_tokens, truncate_messages

# Manual monitoring
tokens = estimate_tokens(client.history)
if tokens > 28000:  # User decides threshold
    # User decides strategy
    client.message_history = truncate_messages(client.history, keep=10)
```

**Benefits of manual approach**:
- No silent mutations (explicit is better than implicit)
- Users stay in control
- Domain-specific strategies easy to implement
- Clear, tested primitives to build on
- Migration path if auto-compaction is needed later

**Alternative approaches documented**:
1. Stateless agents - Fresh context per task
2. Manual truncation - At natural breakpoints
3. Token monitoring - Periodic checks with budget
4. External memory - RAG-lite pattern

**Comparison to Claude SDK**:
- Claude SDK: Automatic compaction via CLI + PreCompact hook
- Our SDK: Manual utilities + 4 documented patterns
- Trade-off: Less "magic" but more control and clarity

### 9. Tool-Calling Streaming Semantics

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

**Total LOC**: ~1300 lines (v0.3.0)

**Breakdown**:
- types.py: ~100 lines (dataclasses + Tool types)
- tools.py: ~270 lines (decorator, schema conversion)
- utils.py: ~220 lines (client, formatting, aggregation)
- client.py: ~365 lines (query + Client class + auto-execution) **[v0.3.0: +100 lines]**
- context.py: ~170 lines (token estimation, truncation) **[v0.2.3]**
- hooks.py: ~136 lines (event types, hook decision) **[v0.2.4]**
- __init__.py: ~15 lines (exports)
- Tests: ~1965 lines (comprehensive coverage) **[v0.3.0: +200 lines]**

**v0.2.2 additions** (Tool System):
- tools.py: 270 lines
- tests/test_tools.py: 267 lines
- examples/calculator_tools.py: 129 lines
- examples/simple_tool.py: 81 lines

**v0.2.3 additions** (Context Utilities):
- context.py: 170 lines
- tests/test_context.py: 230 lines
- examples/context_management.py: 240 lines

**v0.2.4 additions** (Hooks System):
- hooks.py: 136 lines
- tests/test_hooks.py: 435 lines
- examples/hooks_example.py: 320 lines
- Updated client.py for hook integration (~30 lines added)

**v0.3.0 additions** (Automatic Tool Execution):
- Updated client.py: +100 lines (_tool_registry, _continue_turn, _receive_once, _auto_execute_loop)
- tests/test_auto_execution.py: 200 lines (10 comprehensive tests)
- Updated examples/calculator_tools.py: Now demonstrates both modes
- Removed examples/calculator_auto.py: Redundant after consolidation

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

## Future Enhancements

**Completed (v0.2.2)** - Tool System:
- ✅ Tool system with `@tool` decorator
- ✅ Automatic Python type → JSON Schema conversion
- ✅ Sync handler support
- ✅ Optional parameter handling
- ✅ Tool result injection

**Completed (v0.2.3)** - Context Utilities:
- ✅ Token counting/estimation (tiktoken + fallback)
- ✅ Manual history truncation utilities
- ✅ Context management documentation (4 patterns)
- ⏸️ **Intentionally NOT building**: Automatic compaction (see design decision #8)

**Completed (v0.2.4)** - Hooks System:
- ✅ PreToolUse hook - intercept/control tool execution before it happens
- ✅ PostToolUse hook - monitor tool results after execution
- ✅ UserPromptSubmit hook - sanitize/validate user input
- ✅ Pythonic event dataclasses (PreToolUseEvent, PostToolUseEvent, UserPromptSubmitEvent)
- ✅ HookDecision for controlling execution flow
- ✅ Sequential execution with short-circuit behavior
- ✅ Works with both Client and query() contexts
- ✅ 14 comprehensive tests + 4 example patterns
- ⏸️ **Intentionally NOT building**: Stop/SubagentStop hooks (not essential for local LLMs)

**Completed (v0.3.0)** - Automatic Tool Execution:
- ✅ `auto_execute_tools=True` flag in AgentOptions
- ✅ `max_tool_iterations` safety limit (default 5)
- ✅ Tool registry with duplicate validation at init
- ✅ `_continue_turn()` for hook-free conversation continuation
- ✅ `_receive_once()` helper for stream consumption
- ✅ `_auto_execute_loop()` for orchestration with error handling
- ✅ ToolUseError yields for unknown tools and execution failures
- ✅ Structured error dicts to `add_tool_result()` for PostToolUse hooks
- ✅ Works seamlessly with hooks (PreToolUse/PostToolUse fire correctly)
- ✅ Backward compatible (defaults to `False` for manual mode)
- ✅ 10 comprehensive tests covering all scenarios
- ✅ Consolidated example showing both modes

**Planned (Phase 4+)**:
- ⏳ Permission system (allowedTools, disallowedTools)
- ⏳ Session persistence (SQLite)
- ⏳ Retry logic with exponential backoff
- ⏳ ThinkingBlock support for chain-of-thought
- ⏳ Optional MCP bridge for external tool servers
- ⏳ Message type improvements (SystemMessage, UserMessage, etc.)

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
context = [
    "tiktoken>=0.5.0",   # Better token estimation (v0.2.3+)
]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "tiktoken>=0.5.0",   # For testing context utilities
]
```

**Optional dependencies**:
- `tiktoken` - Better token estimation for context management (~90-95% accurate for GPT models, ~70-85% for others)
  - Without: Falls back to character-based estimation (~75-85% accurate)
  - Install: `pip install open-agent-sdk[context]`
- `pydantic` - Stricter types (can use dataclasses instead)

## Success Metrics

The technical design succeeds when:

**Phase 1 (v0.1.0-0.2.1):**
1. ✅ query() matches claude-agent-sdk.query() behavior
2. ✅ Client matches claude-agent-sdk.ClaudeSDKClient behavior
3. ✅ Works with LM Studio, Ollama, llama.cpp
4. ✅ No crashes on malformed responses
5. ✅ Clean async/await patterns throughout
6. ✅ Comprehensive test coverage (85 tests)

**Phase 2 - Tool System (v0.2.2):**
1. ✅ @tool decorator with ergonomic API
2. ✅ Automatic schema conversion (Python types → JSON Schema)
3. ✅ Sync and async handler support
4. ✅ Optional parameter handling (3 methods)
5. ✅ Direct OpenAI tools format (no MCP overhead)
6. ✅ Tool examples (calculator, simple_tool)
7. ✅ 16 comprehensive tool tests
8. ✅ Production-ready quality

**Phase 3 - Context Utilities (v0.2.3):**
1. ✅ `estimate_tokens()` with tiktoken + fallback
2. ✅ `truncate_messages()` simple truncation
3. ✅ Optional `[context]` dependency for better accuracy
4. ✅ 17 comprehensive context tests (102 total)
5. ✅ 4 documented usage patterns in examples
6. ✅ Design rationale documented (manual > automatic)
7. ✅ Lean implementation (170 LOC)
8. ✅ No silent mutations (explicit control)

**Phase 4 - Hooks System (v0.2.4):**
1. ✅ PreToolUse/PostToolUse/UserPromptSubmit hooks
2. ✅ Pythonic event dataclasses (not JSON protocol)
3. ✅ HookDecision for controlling execution flow
4. ✅ Sequential execution with short-circuit behavior
5. ✅ Works with both Client and query() contexts
6. ✅ 14 comprehensive hook tests (118 total)
7. ✅ 4 example patterns (security, audit, sanitization, combined)
8. ✅ Production-ready for logging, monitoring, security
9. ✅ Breaking change documented (async add_tool_result)
10. ✅ Local-first design (inline execution, no subprocess)

**Phase 5 - Automatic Tool Execution (v0.3.0):**
1. ✅ `auto_execute_tools=True` flag with backward compatibility
2. ✅ `max_tool_iterations` safety limit (default 5)
3. ✅ Tool registry with duplicate validation
4. ✅ `_continue_turn()` avoids hook re-firing
5. ✅ `_receive_once()` ensures stream fully consumed
6. ✅ `_auto_execute_loop()` orchestrates execution with error handling
7. ✅ ToolUseError yields for unknown tools and execution failures
8. ✅ Structured error dicts for PostToolUse hook integration
9. ✅ 10 comprehensive auto-execution tests (128 total)
10. ✅ Consolidated example showing both automatic and manual modes
11. ✅ Hooks work identically in both modes
12. ✅ "Pit of success" design - auto mode is easiest path

**Future Phases:**
- ⏳ Permission system for tool control
- ⏳ ThinkingBlock for chain-of-thought
- ⏳ Message type improvements (SystemMessage, UserMessage)
