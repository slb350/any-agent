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
│   ├── config.py        # get_model(), get_base_url() helpers; YAML config loading
│   └── utils.py         # Shared internal utilities
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
├── tests/               # 147 tests (pytest)
├── docs/
│   ├── technical-design.md
│   ├── provider-compatibility.md
│   └── configuration.md
├── pyproject.toml       # uv/pip metadata, version = "0.4.2"
└── CHANGELOG.md
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.10+ |
| **HTTP** | openai>=1.0.0 (AsyncOpenAI) |
| **Optional** | tiktoken (context management), pyyaml (YAML config) |
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
    HookDecision,            # Hook return type (continue/block/modify)
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

### Interrupts
```python
# From a separate asyncio task:
client.interrupt()
```

## AgentOptions Fields

| Field | Default | Description |
|-------|---------|-------------|
| `model` | required | Model name |
| `base_url` | required | OpenAI-compatible endpoint |
| `system_prompt` | `""` | System message |
| `max_turns` | `1` | Max conversation turns |
| `temperature` | `None` | Sampling temperature |
| `max_tokens` | `None` | Max output tokens |
| `hooks` | `{}` | Lifecycle hooks dict |
| `tools` | `[]` | Tool definitions |
| `auto_execute_tools` | `False` | Auto-execute tools |
| `max_tool_iterations` | `5` | Safety limit for tool loops |
| `timeout` | `60` | HTTP timeout (seconds) |
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
- All 147 tests must pass before committing
- Run `ruff check` and `black` before committing
- No breaking changes to `AgentOptions` field order (positional arg compatibility)
- Manual mode (`auto_execute_tools=False`) must remain the default (backwards compat)
- `add_tool_result()` is async — always `await` it
- `_interrupted` flag resets at the start of each `query()` and `receive_messages()` call
- Context management is intentionally **opt-in** — no silent history mutations

## YAML Config

The SDK searches for YAML config files (in priority order, first file found wins):
1. `./open-agent.yaml` (project directory — highest priority)
2. `~/.config/open-agent/config.yaml` (XDG Base Directory standard)
3. `~/.open-agent.yaml` (home directory fallback)

PyYAML is an optional dependency (`pip install open-agent-sdk[yaml]` or `pip install pyyaml`). If not installed, config file loading silently returns `{}` and the SDK falls back to code/environment variable configuration.

Environment variables for runtime overrides (take precedence over config files when using `get_model()`/`get_base_url()` helpers):
- `OPEN_AGENT_MODEL` — override model name
- `OPEN_AGENT_BASE_URL` — override endpoint URL
- `OPEN_AGENT_API_KEY` — override API key
