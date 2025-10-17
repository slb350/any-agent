# Open Agent SDK

> Lightweight Python SDK for local/self-hosted LLMs via OpenAI-compatible endpoints

[![PyPI version](https://badge.fury.io/py/open-agent-sdk.svg)](https://pypi.org/project/open-agent-sdk/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Open Agent SDK provides a clean, streaming API for working with OpenAI-compatible local model servers, making it easy to build AI agents with your own hardware.

**Use Case**: Build powerful AI agents using local Qwen/Llama/Mistral models without cloud API costs or data privacy concerns.

**Solution**: Drop-in similar API that works with LM Studio, Ollama, llama.cpp, and any OpenAI-compatible endpoint—complete with streaming, tool call aggregation, and a helper for returning tool results back to the model.

## Supported Providers

### ✅ Supported (OpenAI-Compatible Endpoints)

- **LM Studio** - `http://localhost:1234/v1`
- **Ollama** - `http://localhost:11434/v1`
- **llama.cpp server** - OpenAI-compatible mode
- **vLLM** - OpenAI-compatible API
- **Text Generation WebUI** - OpenAI extension
- **Any OpenAI-compatible local endpoint**
- **Local gateways proxying cloud models** - e.g., Ollama or custom gateways that route to cloud providers

### ❌ Not Supported (Use Official SDKs)

- **Claude/OpenAI direct** - Use their official SDKs, unless you proxy through a local OpenAI-compatible gateway
- **Cloud provider SDKs** - Bedrock, Vertex, Azure, etc. (proxied via local gateway is fine)

## Quick Start

### Installation

```bash
pip install open-agent-sdk
```

For development:

```bash
git clone https://github.com/slb350/open-agent-sdk.git
cd open-agent-sdk
pip install -e .
```

### Simple Query (LM Studio)

```python
import asyncio
from open_agent import query, AgentOptions

async def main():
    options = AgentOptions(
        system_prompt="You are a professional copy editor",
        model="qwen2.5-32b-instruct",
        base_url="http://localhost:1234/v1",
        max_turns=1,
        temperature=0.1
    )

    result = query(prompt="Analyze this text...", options=options)

    response_text = ""
    async for msg in result:
        if hasattr(msg, 'content'):
            for block in msg.content:
                if hasattr(block, 'text'):
                    response_text += block.text

    print(response_text)

asyncio.run(main())
```

### Multi-Turn Conversation (Ollama)

```python
from open_agent import Client, AgentOptions, TextBlock, ToolUseBlock
from open_agent.config import get_base_url

def run_my_tool(name: str, params: dict) -> dict:
    # Replace with your tool execution logic
    return {"result": f"stub output for {name}"}

async def main():
    options = AgentOptions(
        system_prompt="You are a helpful assistant",
        model="kimi-k2:1t-cloud",  # Use your available Ollama model
        base_url=get_base_url(provider="ollama"),
        max_turns=10
    )

    async with Client(options) as client:
        await client.query("What's the capital of France?")

        async for msg in client.receive_messages():
            if isinstance(msg, TextBlock):
                print(f"Assistant: {msg.text}")
            elif isinstance(msg, ToolUseBlock):
                print(f"Tool used: {msg.name}")
                tool_result = run_my_tool(msg.name, msg.input)
                client.add_tool_result(msg.id, tool_result)

asyncio.run(main())
```

See `examples/tool_use_agent.py` for progressively richer patterns (manual loop, helper function, and reusable agent class) demonstrating `add_tool_result()` in context.

### Function Calling with Tools

Define tools using the `@tool` decorator for clean, type-safe function calling:

```python
from open_agent import tool, Client, AgentOptions, TextBlock, ToolUseBlock

# Define tools
@tool("get_weather", "Get current weather", {"location": str, "units": str})
async def get_weather(args):
    return {
        "temperature": 72,
        "conditions": "sunny",
        "units": args["units"]
    }

@tool("calculate", "Perform calculation", {"a": float, "b": float, "op": str})
async def calculate(args):
    ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b}
    result = ops[args["op"]](args["a"], args["b"])
    return {"result": result}

# Enable automatic tool execution (recommended)
options = AgentOptions(
    system_prompt="You are a helpful assistant with access to tools.",
    model="qwen2.5-32b-instruct",
    base_url="http://localhost:1234/v1",
    tools=[get_weather, calculate],
    auto_execute_tools=True,      # 🔥 Tools execute automatically
    max_tool_iterations=10         # Safety limit for tool loops
)

async with Client(options) as client:
    await client.query("What's 25 + 17?")

    # Simply iterate - tools execute automatically!
    async for block in client.receive_messages():
        if isinstance(block, ToolUseBlock):
            print(f"Tool called: {block.name}")
        elif isinstance(block, TextBlock):
            print(f"Response: {block.text}")
```

**Advanced: Manual Tool Execution**

For custom execution logic or result interception:

```python
# Disable auto-execution
options = AgentOptions(
    system_prompt="You are a helpful assistant with access to tools.",
    model="qwen2.5-32b-instruct",
    base_url="http://localhost:1234/v1",
    tools=[get_weather, calculate],
    auto_execute_tools=False  # Manual mode
)

async with Client(options) as client:
    await client.query("What's 25 + 17?")

    async for block in client.receive_messages():
        if isinstance(block, ToolUseBlock):
            # You execute the tool manually
            tool = {"calculate": calculate, "get_weather": get_weather}[block.name]
            result = await tool.execute(block.input)

            # Return result to agent
            await client.add_tool_result(block.id, result)

            # Continue conversation
            await client.query("")
```

**Key Features:**
- **Automatic execution** (v0.3.0+) - Tools run automatically with safety limits
- **Type-safe schemas** - Simple Python types (`str`, `int`, `float`, `bool`) or full JSON Schema
- **OpenAI-compatible** - Works with any OpenAI function calling endpoint
- **Clean decorator API** - Similar to Claude SDK's `@tool`
- **Hook integration** - PreToolUse/PostToolUse hooks work in both modes

See `examples/calculator_tools.py` and `examples/simple_tool.py` for complete examples.

## Context Management

Local models have fixed context windows (typically 8k-32k tokens). The SDK provides **opt-in utilities** for manual history management—no silent mutations, you stay in control.

### Token Estimation & Truncation

```python
from open_agent import Client, AgentOptions
from open_agent.context import estimate_tokens, truncate_messages

async with Client(options) as client:
    # Long conversation...
    for i in range(50):
        await client.query(f"Question {i}")
        async for msg in client.receive_messages():
            pass

    # Check token usage
    tokens = estimate_tokens(client.history)
    print(f"Context size: ~{tokens} tokens")

    # Manually truncate when needed
    if tokens > 28000:
        client.message_history = truncate_messages(client.history, keep=10)
```

### Recommended Patterns

**1. Stateless Agents** (Best for single-task agents):
```python
# Process each task independently - no history accumulation
for task in tasks:
    async with Client(options) as client:
        await client.query(task)
        # Client disposed, fresh context for next task
```

**2. Manual Truncation** (At natural breakpoints):
```python
from open_agent.context import truncate_messages

async with Client(options) as client:
    for task in tasks:
        await client.query(task)
        # Truncate after each major task
        client.message_history = truncate_messages(client.history, keep=5)
```

**3. External Memory** (RAG-lite for research agents):
```python
# Store important facts in database, keep conversation context small
database = {}
async with Client(options) as client:
    await client.query("Research topic X")
    # Save response to database
    database["topic_x"] = extract_facts(response)

    # Clear history, query database when needed
    client.message_history = truncate_messages(client.history, keep=0)
```

### Why Manual?

The SDK **intentionally** does not auto-compact history because:
- **Domain-specific needs**: Copy editors need different strategies than research agents
- **Token accuracy varies**: Each model family has different tokenizers
- **Risk of breaking context**: Silently removing messages could break tool chains
- **Natural limits exist**: Compaction doesn't bypass model context windows

### Installing Token Estimation

For better token estimation accuracy (optional):

```bash
pip install open-agent-sdk[context]  # Adds tiktoken
```

Without `tiktoken`, falls back to character-based approximation (~75-85% accurate).

See `examples/context_management.py` for complete patterns and usage.

## Lifecycle Hooks

Monitor and control agent behavior at key execution points with Pythonic lifecycle hooks—no subprocess overhead or JSON protocols.

### Quick Example

```python
from open_agent import (
    AgentOptions, Client,
    PreToolUseEvent, PostToolUseEvent, UserPromptSubmitEvent,
    HookDecision,
    HOOK_PRE_TOOL_USE, HOOK_POST_TOOL_USE, HOOK_USER_PROMPT_SUBMIT
)

# Security gate - block dangerous operations
async def security_gate(event: PreToolUseEvent) -> HookDecision | None:
    if event.tool_name == "delete_file":
        return HookDecision(
            continue_=False,
            reason="Delete operations require approval"
        )
    return None  # Allow by default

# Audit logger - track all tool executions
async def audit_logger(event: PostToolUseEvent) -> None:
    print(f"Tool executed: {event.tool_name} -> {event.tool_result}")
    return None

# Input sanitizer - validate user prompts
async def sanitize_input(event: UserPromptSubmitEvent) -> HookDecision | None:
    if "DELETE" in event.prompt.upper():
        return HookDecision(
            continue_=False,
            reason="Dangerous keywords detected"
        )
    return None

# Register hooks in AgentOptions
options = AgentOptions(
    system_prompt="You are a helpful assistant",
    model="qwen2.5-32b-instruct",
    base_url="http://localhost:1234/v1",
    tools=[my_file_tool, my_search_tool],
    hooks={
        HOOK_PRE_TOOL_USE: [security_gate],
        HOOK_POST_TOOL_USE: [audit_logger],
        HOOK_USER_PROMPT_SUBMIT: [sanitize_input],
    }
)

async with Client(options) as client:
    await client.query("Write to /etc/config")  # UserPromptSubmit fires
    async for block in client.receive_messages():
        if isinstance(block, ToolUseBlock):  # PreToolUse fires
            result = await tool.execute(block.input)
            await client.add_tool_result(block.id, result)  # PostToolUse fires
```

### Hook Types

**PreToolUse** - Fires before tool execution (or yielding to user)
- **Block operations**: Return `HookDecision(continue_=False, reason="...")`
- **Modify inputs**: Return `HookDecision(modified_input={...}, reason="...")`
- **Allow**: Return `None`

**PostToolUse** - Fires after tool result added to history
- **Observational only** (tool already executed)
- Use for audit logging, metrics, result validation
- Return `None` (decision ignored for PostToolUse)

**UserPromptSubmit** - Fires before sending prompt to API
- **Block prompts**: Return `HookDecision(continue_=False, reason="...")`
- **Modify prompts**: Return `HookDecision(modified_prompt="...", reason="...")`
- **Allow**: Return `None`

### Common Patterns

**Pattern 1: Redirect to Sandbox**

```python
async def redirect_to_sandbox(event: PreToolUseEvent) -> HookDecision | None:
    """Redirect file operations to safe directory."""
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

**Pattern 2: Compliance Audit Log**

```python
audit_log = []

async def compliance_logger(event: PostToolUseEvent) -> None:
    """Log all tool executions for compliance."""
    audit_log.append({
        "timestamp": datetime.now(),
        "tool": event.tool_name,
        "input": event.tool_input,
        "result": str(event.tool_result)[:100],
        "user": get_current_user()
    })
    return None
```

**Pattern 3: Safety Instructions**

```python
async def add_safety_warning(event: UserPromptSubmitEvent) -> HookDecision | None:
    """Add safety instructions to risky prompts."""
    if "write" in event.prompt.lower() or "delete" in event.prompt.lower():
        safe_prompt = event.prompt + " (Please confirm this is safe before proceeding)"
        return HookDecision(
            modified_prompt=safe_prompt,
            reason="Added safety warning"
        )
    return None
```

### Hook Execution Flow

- Hooks run **sequentially** in the order registered
- **First non-None decision wins** (short-circuit behavior)
- Hooks run **inline on event loop** (spawn tasks for heavy work)
- Works with both **Client** and **query()** function

### Breaking Change (v0.2.4)

`Client.add_tool_result()` is now async to support PostToolUse hooks:

```python
# Old (v0.2.3 and earlier)
client.add_tool_result(tool_id, result)

# New (v0.2.4+)
await client.add_tool_result(tool_id, result)
```

### Why Hooks?

- **Security gates**: Block dangerous operations before they execute
- **Audit logging**: Track all tool executions for compliance
- **Input validation**: Sanitize user prompts before processing
- **Monitoring**: Observe agent behavior in production
- **Control flow**: Modify tool inputs or redirect operations

See `examples/hooks_example.py` for 4 comprehensive patterns (security, audit, sanitization, combined).

## Interrupt Capability

Cancel long-running operations cleanly without corrupting client state. Perfect for timeouts, user cancellations, or conditional interruptions.

### Quick Example

```python
from open_agent import Client, AgentOptions
import asyncio

async def main():
    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="qwen2.5-32b-instruct",
        base_url="http://localhost:1234/v1"
    )

    async with Client(options) as client:
        await client.query("Write a detailed 1000-word essay...")

        # Timeout after 5 seconds
        try:
            async def collect_messages():
                async for block in client.receive_messages():
                    print(block.text, end="", flush=True)

            await asyncio.wait_for(collect_messages(), timeout=5.0)
        except asyncio.TimeoutError:
            await client.interrupt()  # Clean cancellation
            print("\n⚠️ Operation timed out!")

        # Client is still usable after interrupt
        await client.query("Short question?")
        async for block in client.receive_messages():
            print(block.text)
```

### Common Patterns

**1. Timeout-Based Interruption**

```python
try:
    await asyncio.wait_for(process_messages(client), timeout=10.0)
except asyncio.TimeoutError:
    await client.interrupt()
    print("Operation timed out")
```

**2. Conditional Interruption**

```python
# Stop if response contains specific content
full_text = ""
async for block in client.receive_messages():
    full_text += block.text
    if "error" in full_text.lower():
        await client.interrupt()
        break
```

**3. User Cancellation (from separate task)**

```python
async def stream_task():
    await client.query("Long task...")
    async for block in client.receive_messages():
        print(block.text, end="")

async def cancel_button_task():
    await asyncio.sleep(2.0)  # User waits 2 seconds
    await client.interrupt()  # User clicks cancel

# Run both concurrently
await asyncio.gather(stream_task(), cancel_button_task())
```

**4. Interrupt During Auto-Execution**

```python
options = AgentOptions(
    tools=[slow_tool, fast_tool],
    auto_execute_tools=True,
    max_tool_iterations=10
)

async with Client(options) as client:
    await client.query("Use tools...")

    tool_count = 0
    async for block in client.receive_messages():
        if isinstance(block, ToolUseBlock):
            tool_count += 1
            if tool_count >= 2:
                await client.interrupt()  # Stop after 2 tools
                break
```

### How It Works

When you call `client.interrupt()`:
1. **Active stream closure** - HTTP stream closed immediately (not just a flag)
2. **Clean state** - Client remains in valid state for reuse
3. **Partial output** - Text blocks flushed to history, incomplete tools skipped
4. **Idempotent** - Safe to call multiple times
5. **Concurrent-safe** - Can be called from separate asyncio tasks

### Example

See `examples/interrupt_demo.py` for 5 comprehensive patterns:
- Timeout-based interruption
- Conditional interruption
- Auto-execution interruption
- Concurrent interruption (simulated cancel button)
- Interrupt and retry

## 🚀 Practical Examples

We've included two production-ready agents that demonstrate real-world usage:

### 📝 Git Commit Agent
**[examples/git_commit_agent.py](examples/git_commit_agent.py)**

Analyzes your staged git changes and writes professional commit messages following conventional commit format.

```bash
# Stage your changes
git add .

# Run the agent
python examples/git_commit_agent.py

# Output:
# ✓ Found staged changes in 3 file(s)
# 🤖 Analyzing changes and generating commit message...
#
# 📝 Suggested commit message:
# feat(auth): Add OAuth2 integration with refresh tokens
#
# - Implement token refresh mechanism
# - Add secure cookie storage for tokens
# - Update login flow to support OAuth2 providers
# - Add tests for token expiration handling
```

**Features:**
- Analyzes diff to determine commit type (feat/fix/docs/etc)
- Writes clear, descriptive commit messages
- Interactive mode: accept, edit, or regenerate
- Follows conventional commit standards

### 📊 Log Analyzer Agent
**[examples/log_analyzer_agent.py](examples/log_analyzer_agent.py)**

Intelligently analyzes application logs to identify patterns, errors, and provide actionable insights.

```bash
# Analyze a log file
python examples/log_analyzer_agent.py /var/log/app.log

# Analyze with a specific time window
python examples/log_analyzer_agent.py app.log --since "2025-10-15T00:00:00" --until "2025-10-15T12:00:00"

# Interactive mode for drilling down
python examples/log_analyzer_agent.py app.log --interactive
```

**Features:**
- Automatic error pattern detection
- Time-based analysis (peak error times)
- Root cause suggestions
- Interactive mode for investigating specific issues
- Supports multiple log formats (JSON, Apache, syslog, etc)
- Time range filtering with `--since` / `--until`

**Sample Output:**
```
📊 Log Summary:
  Total entries: 45,231
  Errors: 127 (0.3%)
  Warnings: 892

🔴 Top Error Patterns:
  - Connection Error: 67 occurrences
  - NullPointerException: 23 occurrences
  - Timeout Error: 19 occurrences

⏰ Peak error time: 2025-10-15T14:00:00
   Errors in that hour: 43

🤖 ANALYSIS REPORT:
Main Issues (Priority Order):
1. Database connection pool exhaustion during peak hours
2. Unhandled null values in user authentication flow
3. External API timeouts affecting payment processing

Recommendations:
1. Increase connection pool size from 10 to 25
2. Add null checks in AuthService.validateUser() method
3. Implement circuit breaker for payment API with 30s timeout
```

### Why These Examples?

These agents demonstrate:
- **Practical Value**: Solve real problems developers face daily
- **Tool Integration**: Show how to integrate with system commands (git, file I/O)
- **Multi-turn Conversations**: Interactive modes for complex analysis
- **Structured Output**: Parse and format LLM responses for actionable results
- **Privacy-First**: Keep your code and logs local while getting AI assistance

## Configuration

Open Agent SDK uses config helpers to provide flexible configuration via environment variables, provider shortcuts, or explicit parameters:

### Environment Variables (Recommended)

```bash
export OPEN_AGENT_BASE_URL="http://localhost:1234/v1"
export OPEN_AGENT_MODEL="qwen/qwen3-30b-a3b-2507"
```

```python
from open_agent import AgentOptions
from open_agent.config import get_model, get_base_url

# Config helpers read from environment
options = AgentOptions(
    system_prompt="...",
    model=get_model(),      # Reads OPEN_AGENT_MODEL
    base_url=get_base_url() # Reads OPEN_AGENT_BASE_URL
)
```

### Provider Shortcuts

```python
from open_agent.config import get_base_url

# Use built-in defaults for common providers
options = AgentOptions(
    system_prompt="...",
    model="llama3.1:70b",
    base_url=get_base_url(provider="ollama")  # → http://localhost:11434/v1
)
```

**Available providers**: `lmstudio`, `ollama`, `llamacpp`, `vllm`

### Fallback Values

```python
# Provide fallbacks when env vars not set
options = AgentOptions(
    system_prompt="...",
    model=get_model("qwen2.5-32b-instruct"),         # Fallback model
    base_url=get_base_url(provider="lmstudio")       # Fallback URL
)
```

**Configuration Priority:**
- Environment variable (default behaviour)
- Fallback value passed to the config helper
- Provider default (for `base_url` only)

Need to force a specific model even when `OPEN_AGENT_MODEL` is set? Call `get_model("model-name", prefer_env=False)` to ignore the environment variable for that lookup.

**Benefits:**
- Switch between dev/prod by changing environment variables
- No hardcoded URLs or model names
- Per-agent overrides when needed

See [docs/configuration.md](docs/configuration.md) for complete guide.

## Why Not Just Use OpenAI Client?

**Without open-agent-sdk** (raw OpenAI client):
```python
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
response = await client.chat.completions.create(
    model="qwen2.5-32b-instruct",
    messages=[{"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt}],
    stream=True
)

async for chunk in response:
    # Complex parsing of chunks
    # Extract delta content
    # Handle tool calls manually
    # Track conversation state yourself
```

**With open-agent-sdk**:
```python
from open_agent import query, AgentOptions

options = AgentOptions(
    system_prompt=system_prompt,
    model="qwen2.5-32b-instruct",
    base_url="http://localhost:1234/v1"
)

result = query(prompt=user_prompt, options=options)
async for msg in result:
    # Clean message types (TextBlock, ToolUseBlock)
    # Automatic streaming and tool call handling
```

**Value**: Familiar patterns + Less boilerplate + Easy migration

## API Reference

### AgentOptions

```python
class AgentOptions:
    system_prompt: str                          # System prompt
    model: str                                  # Model name (required)
    base_url: str                               # OpenAI-compatible endpoint URL (required)
    tools: list[Tool] = []                      # Tool instances for function calling
    hooks: dict[str, list[HookHandler]] = None  # Lifecycle hooks for monitoring/control
    auto_execute_tools: bool = False            # Enable automatic tool execution (v0.3.0+)
    max_tool_iterations: int = 5                # Max tool calls per query in auto mode
    max_turns: int = 1                          # Max conversation turns
    max_tokens: int | None = 4096               # Tokens to generate (None uses provider default)
    temperature: float = 0.7                    # Sampling temperature
    timeout: float = 60.0                       # Request timeout in seconds
    api_key: str = "not-needed"                 # Most local servers don't need this
```

**Note**: Use config helpers (`get_model()`, `get_base_url()`) for environment variable and provider support.

### query()

Simple single-turn query function.

```python
async def query(prompt: str, options: AgentOptions) -> AsyncGenerator
```

Returns an async generator yielding messages.

### Client

Multi-turn conversation client with tool monitoring.

```python
async with Client(options: AgentOptions) as client:
    await client.query(prompt: str)
    async for msg in client.receive_messages():
        # Process messages
```

### Message Types

- `TextBlock` - Text content from model
- `ToolUseBlock` - Tool calls from model (has `id`, `name`, `input` fields)
- `ToolResultBlock` - Tool execution results to send back to model
- `ToolUseError` - Tool call parsing error (malformed JSON, missing fields)
- `AssistantMessage` - Full message wrapper

### Tool System

```python
@tool(name: str, description: str, input_schema: dict)
async def my_tool(args: dict) -> Any:
    """Tool handler function"""
    return result

# Tool class
class Tool:
    name: str
    description: str
    input_schema: dict[str, type] | dict[str, Any]
    handler: Callable[[dict], Awaitable[Any]]

    async def execute(arguments: dict) -> Any
    def to_openai_format() -> dict
```

**Schema formats:**
- Simple: `{"param": str, "count": int}` - All parameters required
- JSON Schema: Full schema with `type`, `properties`, `required`, etc.

### Hooks System

```python
# Event types
@dataclass
class PreToolUseEvent:
    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str
    history: list[dict[str, Any]]

@dataclass
class PostToolUseEvent:
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: Any
    tool_use_id: str
    history: list[dict[str, Any]]

@dataclass
class UserPromptSubmitEvent:
    prompt: str
    history: list[dict[str, Any]]

# Hook decision
@dataclass
class HookDecision:
    continue_: bool = True
    modified_input: dict[str, Any] | None = None
    modified_prompt: str | None = None
    reason: str | None = None

# Hook handler signature
HookHandler = Callable[[HookEvent], Awaitable[HookDecision | None]]

# Hook constants
HOOK_PRE_TOOL_USE = "pre_tool_use"
HOOK_POST_TOOL_USE = "post_tool_use"
HOOK_USER_PROMPT_SUBMIT = "user_prompt_submit"
```

**Hook behavior:**
- Return `None` to allow by default
- Return `HookDecision(continue_=False)` to block
- Return `HookDecision(modified_input={...})` to modify (PreToolUse)
- Return `HookDecision(modified_prompt="...")` to modify (UserPromptSubmit)
- Raise exception to abort entirely

## Recommended Models

**Local models** (LM Studio, Ollama, llama.cpp):
- **GPT-OSS-120B** - Best in class for speed and quality
- **Qwen 3 30B** - Excellent instruction following, good for most tasks
- **GPT-OSS-20B** - Solid all-around performance
- **Mistral 7B** - Fast and efficient for simple agents

**Cloud-proxied via local gateway** (Ollama cloud provider, custom gateway):
- **kimi-k2:1t-cloud** - Tested and working via Ollama gateway
- **deepseek-v3.1:671b-cloud** - High-quality reasoning model
- **qwen3-coder:480b-cloud** - Code-focused models
- Your `base_url` still points to localhost gateway (e.g., `http://localhost:11434/v1`)
- Gateway handles authentication and routing to cloud provider
- Useful when you need larger models than your hardware can run locally

**Architecture guidance:**
- Prefer MoE (Mixture of Experts) models over dense when available - significantly faster
- Start with 7B-30B models for most agent tasks - they're fast and capable
- Test models with your specific use case - the LLM landscape changes rapidly

## Project Structure

```
open-agent-sdk/
├── open_agent/
│   ├── __init__.py        # query, Client, AgentOptions exports
│   ├── client.py          # Streaming query(), Client, tool helper
│   ├── config.py          # Env/provider helpers
│   ├── context.py         # Token estimation and truncation utilities
│   ├── hooks.py           # Lifecycle hooks (PreToolUse, PostToolUse, UserPromptSubmit)
│   ├── tools.py           # Tool decorator and schema conversion
│   ├── types.py           # Dataclasses for options and blocks
│   └── utils.py           # OpenAI client + ToolCallAggregator
├── docs/
│   ├── configuration.md
│   ├── provider-compatibility.md
│   └── technical-design.md
├── examples/
│   ├── git_commit_agent.py     # 🌟 Practical: Git commit message generator
│   ├── log_analyzer_agent.py   # 🌟 Practical: Log file analyzer
│   ├── calculator_tools.py     # Function calling with @tool decorator
│   ├── simple_tool.py          # Minimal tool usage example
│   ├── tool_use_agent.py       # Complete tool use patterns
│   ├── context_management.py   # Manual history management patterns
│   ├── hooks_example.py        # Lifecycle hooks patterns (security, audit, sanitization)
│   ├── interrupt_demo.py       # Interrupt capability patterns (timeout, conditional, concurrent)
│   ├── simple_lmstudio.py      # Basic usage with LM Studio
│   ├── ollama_chat.py          # Multi-turn chat example
│   ├── config_examples.py      # Configuration patterns
│   └── simple_with_env.py      # Environment variable config
├── tests/
│   ├── integration/               # Integration-style tests using fakes
│   │   └── test_client_behaviour.py  # Streaming, multi-turn, tool flow coverage
│   ├── test_agent_options.py
│   ├── test_auto_execution.py     # Automatic tool execution
│   ├── test_client.py
│   ├── test_config.py
│   ├── test_context.py            # Context utilities (token estimation, truncation)
│   ├── test_hooks.py              # Lifecycle hooks (PreToolUse, PostToolUse, UserPromptSubmit)
│   ├── test_interrupt.py          # Interrupt capability (timeout, concurrent, reuse)
│   ├── test_query.py
│   ├── test_tools.py              # Tool decorator and schema conversion
│   └── test_utils.py
├── CHANGELOG.md
├── pyproject.toml
└── README.md
```

## Examples

### 🌟 Practical Agents (Production-Ready)
- **`git_commit_agent.py`** – Analyzes git diffs and writes professional commit messages
- **`log_analyzer_agent.py`** – Parses logs, finds patterns, suggests fixes with interactive mode
- **`tool_use_agent.py`** – Complete tool use patterns: manual, helper, and agent class

### Core SDK Usage
- `simple_lmstudio.py` – Minimal streaming query with hard-coded config (simplest quickstart)
- `simple_with_env.py` – Using environment variables with config helpers and fallbacks
- `config_examples.py` – Comprehensive reference: provider shortcuts, priority, and all config patterns
- `ollama_chat.py` – Multi-turn chat loop with Ollama, including tool-call logging
- `context_management.py` – Manual history management patterns (stateless, truncation, token monitoring, RAG-lite)
- `hooks_example.py` – Lifecycle hooks patterns (security gates, audit logging, input sanitization, combined)

### Integration Tests
Located in `tests/integration/`:
- `test_client_behaviour.py` – Fake AsyncOpenAI client covering streaming, multi-turn history, and tool-call flows without hitting real servers

## Development Status

**Released v0.1.0** – Core functionality is complete and available on PyPI. Multi-turn conversations, tool monitoring, and streaming are fully implemented.

### Roadmap

- [x] Project planning and architecture
- [x] Core `query()` and `Client` class
- [x] Tool monitoring + `Client.add_tool_result()` helper
- [x] Tool use example (`examples/tool_use_agent.py`)
- [x] PyPI release - Published as `open-agent-sdk`
- [ ] Provider compatibility matrix expansion
- [ ] Additional agent examples

### Tested Providers

- ✅ **Ollama** - Fully validated with `kimi-k2:1t-cloud` (cloud-proxied model)
- ✅ **LM Studio** - Fully validated with `qwen/qwen3-30b` model
- ✅ **llama.cpp** - Fully validated with TinyLlama 1.1B model

See [docs/provider-compatibility.md](docs/provider-compatibility.md) for detailed test results.

## Documentation

- [docs/technical-design.md](docs/technical-design.md) - Architecture details
- [docs/configuration.md](docs/configuration.md) - Configuration guide
- [docs/provider-compatibility.md](docs/provider-compatibility.md) - Provider test results
- [examples/](examples/) - Usage examples

## Testing

Integration-style tests run entirely against lightweight fakes, so they are safe to execute locally and in pre-commit:

```bash
python -m pytest tests/integration
```

Add `-k` or a specific path when you want to target a subset of the unit tests (`tests/test_client.py`, etc.). If you use a virtual environment, prefix commands with `./venv/bin/python -m`.

## Pre-commit Hooks

Install hooks once per clone:

```bash
pip install pre-commit
pre-commit install
```

Running `pre-commit run --all-files` will execute formatting checks and the integration tests (`python -m pytest tests/integration`) before you push changes.

## Requirements

- Python 3.10+
- openai 1.0+ (for AsyncOpenAI client)
- pydantic 2.0+ (for types, optional)
 - Some servers require a dummy `api_key`; set any non-empty string if needed

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- API design inspired by [claude-agent-sdk](https://github.com/anthropics/claude-agent-sdk-python)
- Built for local/open-source LLM enthusiasts

---

**Status**: Alpha - API stabilizing, feedback welcome

Star ⭐ this repo if you're building AI agents with local models!
