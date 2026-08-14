# Open Agent SDK (Python)

## Project Description

A lightweight Python SDK (v0.4.2) for building AI agents with local or cloud LLMs via OpenAI-compatible endpoints. Inspired by the Claude SDK API shape. Published to PyPI as `open-agent-sdk`.

## Repository Structure

```
open-agent-sdk/
├── open_agent/
│   ├── __init__.py      # Public API exports
│   ├── client.py        # query() function + Client class
│   ├── types.py         # AgentOptions, TextBlock, ToolUseBlock, ToolUseError, ToolResultBlock, AssistantMessage
│   ├── tools.py         # @tool decorator + Tool class
│   ├── hooks.py         # Hooks system (PreToolUse, PostToolUse, UserPromptSubmit)
│   ├── context.py       # Token estimation + truncation utilities (opt-in)
│   ├── config.py        # get_model(), get_base_url(), load_config_file() helpers; YAML config loading
│   └── utils.py         # create_client(), format_messages(), format_tools(), ToolCallAggregator
├── examples/            # Runnable examples
│   ├── simple_lmstudio.py
│   ├── simple_tool.py
│   ├── calculator_tools.py
│   ├── tool_use_agent.py
│   ├── hooks_example.py
│   ├── context_management.py
│   ├── interrupt_demo.py
│   ├── git_commit_agent.py
│   ├── log_analyzer_agent.py
│   ├── ollama_chat.py
│   ├── config_examples.py
│   └── simple_with_env.py
├── tests/               # 147 tests (pytest); conftest.py has shared fake-client fixtures
│   ├── conftest.py      # Shared fake-client fixtures
│   ├── integration/     # Integration-style tests using fake AsyncOpenAI client
│   │   └── test_client_behaviour.py
│   └── test_*.py        # Unit tests per module
├── docs/
│   ├── technical-design.md
│   ├── provider-compatibility.md
│   └── configuration.md
├── pyproject.toml       # pip/setuptools metadata, version = "0.4.2"
├── uv.lock              # Locked dependency pins for reproducible installs (uv)
├── .pre-commit-config.yaml  # Pre-commit hooks (whitespace checks + pytest tests/)
└── CHANGELOG.md
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.10+ |
| **HTTP** | openai>=1.0.0 (AsyncOpenAI) |
| **Optional** | tiktoken (`pip install open-agent-sdk[context]`), pyyaml (`pip install open-agent-sdk[yaml]`) |
| **Tests** | pytest, pytest-asyncio |
| **Linting** | ruff |
| **Formatting** | black |

## Common Commands

```bash
# Install for development
pip install -e .
pip install -e ".[dev]"   # includes test/lint deps

# Run tests
pytest tests/

# Lint
ruff check open_agent/ tests/

# Format
black open_agent/ tests/

# Run an example
python examples/git_commit_agent.py
```

## Public API

```python
from open_agent import (
    query,                   # Simple single-turn query
    Client,                  # Multi-turn conversation client
    AgentOptions,            # Configuration dataclass
    TextBlock,               # Text content block
    ToolUseBlock,            # Tool call request block
    ToolUseError,            # Malformed tool call error
    ToolResultBlock,         # Tool result to feed back
    AssistantMessage,        # Complete response wrapper
    tool,                    # @tool decorator
    Tool,                    # Tool definition class
    PreToolUseEvent,         # Hook: before tool execution
    PostToolUseEvent,        # Hook: after tool execution
    UserPromptSubmitEvent,   # Hook: before user input processed
    HookEvent,               # Union type of all hook events
    HookDecision,            # Hook return type (continue/block/modify)
    HookHandler,             # Type alias for hook handler functions
    HOOK_PRE_TOOL_USE,
    HOOK_POST_TOOL_USE,
    HOOK_USER_PROMPT_SUBMIT,
)
```

## Key Features

### Streaming API
```python
async for msg in query(prompt, options):
    for block in msg.content:
        if isinstance(block, TextBlock):
            print(block.text)
```

### Tool Use
```python
@tool("my_tool", "Description", {"param": str})
async def my_tool(args):
    return {"result": "..."}

options = AgentOptions(
    tools=[my_tool],
    auto_execute_tools=True,   # automatic execution (recommended)
    max_tool_iterations=10,    # safety limit
)
```

### Hooks
```python
async def pre_tool_hook(event: PreToolUseEvent):
    if event.tool_name == "dangerous_op":
        return HookDecision(continue_=False, reason="Blocked")

options = AgentOptions(
    hooks={HOOK_PRE_TOOL_USE: [pre_tool_hook]}
)
```

### Context Management (opt-in)
```python
from open_agent.context import estimate_tokens, truncate_messages
```

`estimate_tokens()` uses `tiktoken` when installed; falls back to a character-based approximation (~60-80% accurate) calibrated for English. Non-English text (especially CJK) can have significantly different token-to-character ratios — install `tiktoken` for multilingual accuracy.

### Multi-turn Client (two-step pattern)
```python
# query() prepares the stream; receive_messages() consumes it
await client.query("What's 2+2?")
async for block in client.receive_messages():
    if isinstance(block, TextBlock):
        print(block.text)
```

Use `async with` for automatic resource cleanup:
```python
async with Client(options) as client:
    await client.query("prompt")
    async for block in client.receive_messages():
        ...
```

Or call `await client.close()` explicitly when done.

### Interrupts
```python
# From a separate asyncio task:
await client.interrupt()
```

### Turn Metadata
```python
# Inspect conversation progress
meta = client.turn_metadata  # {"turn_count": int, "max_turns": int}
```

## AgentOptions Fields

Fields are listed in dataclass definition order (positional argument order).

| Field | Default | Description |
|-------|---------|-------------|
| `system_prompt` | required | System instructions defining agent role and behavior |
| `model` | required | Model name (provider-specific) |
| `base_url` | required | OpenAI-compatible endpoint (must start with http:// or https://) |
| `tools` | `[]` | Tool definitions |
| `hooks` | `None` | Lifecycle hooks dict (must appear before `auto_execute_tools` for positional arg compatibility) |
| `auto_execute_tools` | `False` | Auto-execute tools |
| `max_tool_iterations` | `5` | Safety limit for tool loops (auto mode only) |
| `max_turns` | `1` | Max conversation turns |
| `max_tokens` | `4096` | Max output tokens (None = provider default) |
| `temperature` | `0.7` | Sampling temperature |
| `timeout` | `60.0` | HTTP timeout (seconds) |
| `api_key` | `"not-needed"` | API key (local servers don't need one) |

## Supported Providers

All OpenAI-compatible endpoints:
- LM Studio: `http://localhost:1234/v1`
- Ollama: `http://localhost:11434/v1`
- llama.cpp server (OpenAI mode)
- vLLM, Text Generation WebUI
- Any local gateway proxying cloud models

## Development Rules

- TDD: Write failing tests first, implement, refactor
- All 147 tests must pass before committing (147 tests collected; run `pytest tests/` to verify)
- Run `ruff check` and `black` before committing
- No breaking changes to `AgentOptions` field order (positional arg compatibility)
- Manual mode (`auto_execute_tools=False`) must remain the default (backwards compat)
- `add_tool_result()` is async — always `await` it
- `PostToolUseEvent` handlers are **observation-only** — return values are ignored (no blocking or result modification); use `PreToolUseEvent` for interception
- `_interrupted` flag resets at the start of each `query()` call only (not `receive_messages()`)
- Context management is intentionally **opt-in** — no silent history mutations

## YAML Config

The SDK searches for YAML config files (in priority order, first file found wins):
1. `./open-agent.yaml` (project directory — highest priority)
2. `~/.config/open-agent/config.yaml` (XDG Base Directory standard)
3. `~/.open-agent.yaml` (home directory fallback)

PyYAML is an optional dependency (`pip install open-agent-sdk[yaml]` or `pip install pyyaml`). If not installed, config file loading silently returns `{}` and the SDK falls back to code/environment variable configuration.

**YAML error behavior**: Invalid YAML raises `yaml.YAMLError` (not caught — fail fast). File I/O errors (e.g., permission denied) also raise and are not caught. An empty YAML file returns `{}`.

Environment variables for runtime overrides (take precedence over config files when using `get_model()`/`get_base_url()` helpers):
- `OPEN_AGENT_MODEL` — override model name
- `OPEN_AGENT_BASE_URL` — override endpoint URL
