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
│   ├── types.py           # Dataclasses for options and blocks
│   └── utils.py           # OpenAI client + ToolCallAggregator
├── docs/
│   ├── configuration.md
│   ├── provider-compatibility.md
│   ├── roadmap.md
│   └── technical-design.md
├── examples/
│   ├── git_commit_agent.py     # 🌟 Practical: Git commit message generator
│   ├── log_analyzer_agent.py   # 🌟 Practical: Log file analyzer
│   ├── tool_use_agent.py       # Complete tool use patterns
│   ├── simple_lmstudio.py      # Basic usage with LM Studio
│   ├── ollama_chat.py          # Multi-turn chat example
│   ├── config_examples.py      # Configuration patterns
│   ├── simple_with_env.py      # Environment variable config
│   ├── env_config_complete.py  # Strict env config
│   ├── test_lmstudio.py        # Provider test suite
│   ├── test_llamacpp.py        # llama.cpp test suite
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
├── pyproject.toml
└── README.md
```

## Examples

### 🌟 Practical Agents (Production-Ready)
- **`git_commit_agent.py`** – Analyzes git diffs and writes professional commit messages
- **`log_analyzer_agent.py`** – Parses logs, finds patterns, suggests fixes with interactive mode
- **`tool_use_agent.py`** – Complete tool use patterns: manual, helper, and agent class

### Core SDK Usage
- `simple_lmstudio.py` – Minimal streaming query against a local LM Studio server
- `ollama_chat.py` – Multi-turn chat loop with Ollama, including tool-call logging
- `simple_with_env.py` – Query pattern using config helpers with fallbacks
- `config_examples.py` – Shows provider shortcuts and manual overrides side by side
- `env_config_complete.py` – Strict environment-variable configuration

### Provider Testing
- `test_lmstudio.py` – Comprehensive LM Studio test suite
- `test_llamacpp.py` – llama.cpp provider test suite
- `test_ollama_kimi.py` – Quick validation for Ollama with kimi-k2 model
- `test_multiturn_network.py` – Network multi-turn conversation tests
- `test_timeout.py` – Timeout configuration verification

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
- [docs/roadmap.md](docs/roadmap.md) - Current milestones and future work
- [docs/provider-compatibility.md](docs/provider-compatibility.md) - Provider test results
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

**Status**: Alpha - API stabilizing, feedback welcome

Star ⭐ this repo if you're building AI agents with local models!
