# Any-Agent SDK

> Claude Agent SDK-style API for local/self-hosted LLMs via OpenAI-compatible endpoints

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Any-Agent SDK provides a lightweight wrapper around OpenAI-compatible local model servers, with the same ergonomics as `claude-agent-sdk`.

**Use Case**: You love the claude-agent-sdk workflow for building agents, but want to use local Qwen/Llama/Mistral models instead of paying for Claude API.

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
pip install any-agent  # Coming soon to PyPI
```

For development:

```bash
git clone https://github.com/yourusername/any-agent.git
cd any-agent
pip install -e .
```

### Simple Query (LM Studio)

```python
import asyncio
from any_agent import query, AgentOptions

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
from any_agent import Client, AgentOptions, TextBlock, ToolUseBlock
from any_agent.config import get_base_url

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

## Configuration

Any-Agent SDK uses config helpers to provide flexible configuration via environment variables, provider shortcuts, or explicit parameters:

### Environment Variables (Recommended)

```bash
export ANY_AGENT_BASE_URL="https://lmstudio.localbrandonfamily.com/v1"
export ANY_AGENT_MODEL="qwen/qwen3-30b-a3b-2507"
```

```python
from any_agent import AgentOptions
from any_agent.config import get_model, get_base_url

# Config helpers read from environment
options = AgentOptions(
    system_prompt="...",
    model=get_model(),      # Reads ANY_AGENT_MODEL
    base_url=get_base_url() # Reads ANY_AGENT_BASE_URL
)
```

### Provider Shortcuts

```python
from any_agent.config import get_base_url

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
- Explicit parameter (highest)
- Environment variable
- Fallback value passed to config helper
- Provider default (for base_url only)

**Benefits:**
- Switch between dev/prod by changing environment variables
- No hardcoded URLs or model names
- Per-agent overrides when needed

See [docs/configuration.md](docs/configuration.md) for complete guide.

## Why Not Just Use OpenAI Client?

**Without any-agent** (raw OpenAI client):
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

**With any-agent**:
```python
from any_agent import query, AgentOptions

options = AgentOptions(
    system_prompt=system_prompt,
    model="qwen2.5-32b-instruct",
    base_url="http://localhost:1234/v1"
)

result = query(prompt=user_prompt, options=options)
async for msg in result:
    # Clean message types (TextBlock, ToolUseBlock)
    # Same pattern as claude-agent-sdk
```

**Value**: Familiar patterns + Less boilerplate + Easy migration

## API Reference

### AgentOptions

```python
class AgentOptions:
    system_prompt: str                      # System prompt
    model: str                              # Model name (required)
    base_url: str                           # OpenAI-compatible endpoint URL (required)
    max_turns: int = 1                      # Max conversation turns
    max_tokens: int | None = 4096           # Tokens to generate (None uses provider default)
    temperature: float = 0.7                # Sampling temperature
    api_key: str = "not-needed"             # Most local servers don't need this
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
- `ToolUseBlock` - Tool calls from model
- `ToolUseError` - Tool call parsing error (malformed JSON, missing fields)
- `AssistantMessage` - Full message wrapper

## Recommended Models

**Local models** (LM Studio, Ollama, llama.cpp):
- **GPT-OSS-120B** - Best in class for speed and quality
- **Qwen 3 30B** - Excellent instruction following, good for most tasks
- **GPT-OSS-20B - Solid all-around performance
- **Mistral 7B - Fast and efficient for simple agents

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
any-agent/
├── any_agent/
│   ├── __init__.py        # query, Client, AgentOptions exports
│   ├── client.py          # Streaming query(), Client, tool helper
│   ├── config.py          # Env/provider helpers
│   ├── types.py           # Dataclasses for options and blocks
│   └── utils.py           # OpenAI client + ToolCallAggregator
├── docs/
│   ├── configuration.md
│   ├── implementation.md
│   ├── provider-compatibility.md
│   ├── roadmap.md
│   └── technical-design.md
├── examples/
│   ├── simple_lmstudio.py
│   ├── simple_with_env.py
│   ├── env_config_complete.py
│   ├── config_examples.py
│   ├── ollama_chat.py
│   ├── tool_use_agent.py
│   ├── test_lmstudio.py
│   ├── test_multiturn_network.py
│   ├── test_network_lmstudio.py
│   ├── test_ollama_kimi.py
│   └── test_timeout.py
├── tests/
│   ├── test_agent_options.py
│   ├── test_client.py
│   ├── test_config.py
│   ├── test_query.py
│   └── test_utils.py
├── CHANGELOG.md
├── CLAUDE.md
├── pyproject.toml
└── README.md
```

## Examples

- `examples/simple_lmstudio.py` – Minimal streaming query against a local LM Studio server.
- `examples/simple_with_env.py` – Same query pattern, but pulls model/URL via config helpers with fallbacks.
- `examples/env_config_complete.py` – Strict environment-variable configuration; raises if settings are missing.
- `examples/config_examples.py` – Shows the provider shortcuts and manual overrides side by side.
- `examples/ollama_chat.py` – Multi-turn chat loop with Ollama, including tool-call logging.
- `examples/tool_use_agent.py` – End-to-end tool flow: manual handling, helper function, and reusable agent class.
- `examples/test_lmstudio.py` – Comprehensive LM Studio smoke test covering streaming, timeouts, and errors.
- `examples/test_multiturn_network.py` – Network-only multi-turn regression test for remote endpoints.
- `examples/test_network_lmstudio.py` – Connectivity/timeout checks for LM Studio over the network.
- `examples/test_ollama_kimi.py` – Quick validation script for the `kimi-k2` model on Ollama.
- `examples/test_timeout.py` – Ensures custom timeout settings behave as expected.

## Development Status

**Currently in active development** – multi-turn support, tool monitoring, and tool-result helpers are implemented; we’re polishing examples and broadening provider coverage.

### Roadmap

- [x] Project planning and architecture
- [x] Core `query()` and `Client` class (tested with Ollama `kimi-k2:1t-cloud`)
- [x] Tool monitoring + `Client.add_tool_result()` helper
- [x] Tool use example (`examples/tool_use_agent.py`)
- [ ] Provider compatibility matrix
- [ ] PyPI release tooling

### Tested Providers

- ✅ **Ollama** - Validated with `kimi-k2:1t-cloud` (cloud-proxied model)
- ⏳ **LM Studio** - Pending test
- ⏳ **llama.cpp** - Pending test

See [docs/implementation.md](docs/implementation.md) for detailed plan.

## Documentation

- [CLAUDE.md](CLAUDE.md) - Project overview and context
- [docs/technical-design.md](docs/technical-design.md) - Architecture details
- [docs/implementation.md](docs/implementation.md) - Implementation plan
- [docs/roadmap.md](docs/roadmap.md) - Current milestones and future work
- [docs/provider-compatibility.md](docs/provider-compatibility.md) - Provider notes (tracked as we validate)
- [examples/](examples/) - Usage examples

## Testing

```bash
./venv/bin/python -m pytest
```

Add `-k` or a specific path when you want to target a subset of the suite (`tests/test_client.py`, etc.).

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

**Status**: Pre-alpha - API is subject to change

Star ⭐ this repo if you want claude-agent-sdk ergonomics for your local models!
