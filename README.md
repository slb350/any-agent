# Any-Agent SDK

> Claude Agent SDK-style API for local/self-hosted LLMs via OpenAI-compatible endpoints

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Any-Agent SDK provides a lightweight wrapper around OpenAI-compatible local model servers, with the same ergonomics as `claude-agent-sdk`.

**Use Case**: You love the claude-agent-sdk workflow for building agents, but want to use local Qwen/Llama/Mistral models instead of paying for Claude API.

**Solution**: Drop-in similar API that works with LM Studio, Ollama, llama.cpp, and any OpenAI-compatible endpoint.

## Supported Providers

✅ **LM Studio** - http://localhost:1234/v1
✅ **Ollama** - http://localhost:11434
✅ **llama.cpp server** - OpenAI-compatible mode
✅ **vLLM** - OpenAI-compatible API
✅ **Text Generation WebUI** - OpenAI extension
✅ **Any OpenAI-compatible local endpoint**

❌ **NOT for Claude/OpenAI** - Use their official SDKs instead
❌ **NOT for cloud providers** - This is for local/self-hosted models only

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
from any_agent import Client, AgentOptions

async def main():
    options = AgentOptions(
        system_prompt="You are a helpful assistant",
        model="llama3.1:70b",
        base_url="http://localhost:11434",
        max_turns=10
    )

    async with Client(options) as client:
        await client.query("What's the capital of France?")

        async for msg in client.receive_messages():
            if isinstance(msg, TextBlock):
                print(f"Assistant: {msg.text}")
            elif isinstance(msg, ToolUseBlock):
                print(f"Tool used: {msg.name}")

asyncio.run(main())
```

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
    system_prompt: str           # System prompt
    model: str                   # Model name (e.g., "qwen2.5-32b-instruct")
    base_url: str               # Endpoint (e.g., "http://localhost:1234/v1")
    max_turns: int = 1          # Max conversation turns
    max_tokens: int = 8000      # Max tokens to generate
    temperature: float = 0.7    # Sampling temperature
    api_key: str = "not-needed" # Most local servers don't need this
```

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
- `AssistantMessage` - Full message wrapper

## Recommended Models

**Fast & Efficient:**
- Qwen 2.5 7B/14B/32B - Excellent instruction following
- Llama 3.2 3B/8B - Good for simple tasks
- Mistral 7B v0.3 - Solid all-around

**Larger/Better:**
- Qwen 2.5 72B - Near GPT-4 quality
- Llama 3.1 70B - Very capable
- DeepSeek-V3 - If you have the VRAM

**For Copy Editing/Analysis:**
- Qwen 2.5 32B+ - Excellent for detailed analysis
- Command-R+ 104B - If resources allow

## Project Structure

```
any-agent/
├── any_agent/
│   ├── __init__.py      # Main exports: query, Client, AgentOptions
│   ├── client.py        # query() and Client class
│   ├── types.py         # Message types and AgentOptions
│   └── utils.py         # OpenAI client helpers
├── examples/
│   ├── simple_lmstudio.py
│   ├── ollama_chat.py
│   └── copy_editor_port.py
├── tests/
├── pyproject.toml
└── README.md
```

## Development Status

**Currently in active development** - Phase 1 (MVP)

### Roadmap

- [x] Project planning and architecture
- [ ] **Phase 1**: Core query() and Client class (Week 1)
- [ ] **Phase 2**: Multi-turn support with tool monitoring (Week 2)
- [ ] **Phase 3**: Port copy_editor agent as validation (Week 3)
- [ ] **Phase 4**: Documentation and PyPI release

See [docs/implementation.md](docs/implementation.md) for detailed plan.

## Documentation

- [CLAUDE.md](CLAUDE.md) - Project overview and context
- [docs/technical-design.md](docs/technical-design.md) - Architecture details
- [docs/implementation.md](docs/implementation.md) - Implementation plan
- [examples/](examples/) - Usage examples

## Requirements

- Python 3.10+
- openai 1.0+ (for AsyncOpenAI client)
- pydantic 2.0+ (for types, optional)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- API design inspired by [claude-agent-sdk](https://github.com/anthropics/claude-agent-sdk-python)
- Built for local/open-source LLM enthusiasts

---

**Status**: Pre-alpha - API is subject to change

Star ⭐ this repo if you want claude-agent-sdk ergonomics for your local models!
