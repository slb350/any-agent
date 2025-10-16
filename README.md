# Any-Agent SDK

> A lightweight, opinionated agent framework that brings Claude Agent SDK-style capabilities to any OpenAI-compatible LLM provider.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Any-Agent SDK provides a production-ready agent framework for building AI agents that work with any OpenAI-compatible endpoint. Built on top of [LiteLLM](https://github.com/BerriAI/litellm), it adds the missing "agent layer" features that developers need:

- 🧠 **Automatic Context Management** - Never worry about token limits again
- 🛠️ **Tool Framework** - Easy function registration and execution
- 💬 **Session Management** - Stateful conversations with persistent context
- 💾 **Memory & Persistence** - SQLite-backed interaction history
- 🎯 **Claude SDK Ergonomics** - Familiar patterns for Claude developers

### Why Any-Agent?

| Feature | LiteLLM Alone | LangChain | Any-Agent |
|---------|---------------|-----------|-----------|
| Provider switching | ✅ | ✅ | ✅ |
| Automatic context management | ❌ | Manual | ✅ |
| Simple stateless queries | ✅ | Complex | ✅ |
| Stateful sessions | ❌ | Via memory | ✅ |
| Tool orchestration | ❌ | ✅ | ✅ |
| Learning curve | Low | High | Low |
| Typical agent LOC | ~50 | ~150 | **~30** |

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

### Basic Usage

```python
import asyncio
from any_agent import Agent

async def main():
    # Create an agent
    agent = Agent(model="gpt-4")

    # Simple query
    response = await agent.query("What is machine learning?")
    print(response)

asyncio.run(main())
```

### With Local Models

```python
from any_agent import Agent

# Use Ollama
agent = Agent(
    model="ollama/llama3.1",
    api_base="http://localhost:11434"
)

# Use LM Studio
agent = Agent(
    model="local-model",
    api_base="http://localhost:1234/v1"
)

response = await agent.query("Explain quantum computing")
```

### With Tools

```python
from any_agent import Agent

agent = Agent(model="gpt-4")

@agent.tool("get_weather")
def get_weather(city: str) -> str:
    """Get the current weather for a city"""
    # Your implementation
    return f"Sunny in {city}"

response = await agent.query("What's the weather in Paris?")
# Agent automatically calls get_weather tool
```

### Stateful Sessions

```python
from any_agent import Agent

agent = Agent(model="gpt-4")
session = await agent.create_session()

# Multi-turn conversation with context
await session.send("My name is Alice")
response1 = await session.receive()

await session.send("What's my name?")
response2 = await session.receive()  # Remembers "Alice"

# Save for later
await session.save()
```

### Custom Configuration

```python
from any_agent import Agent, AgentOptions, ContextStrategy

options = AgentOptions(
    system_prompt="You are a helpful coding assistant",
    temperature=0.7,
    max_tokens=4000,
    context_strategy=ContextStrategy.SLIDING_WINDOW,
    enable_memory=True
)

agent = Agent(model="gpt-4", options=options)
```

## Features

### 🧠 Automatic Context Management

No more manual token counting or message truncation. Any-Agent automatically:

- Counts tokens accurately per model
- Applies sliding window or truncation strategies
- Preserves system prompts and recent messages
- Handles conversations of any length

```python
# Have long conversations without thinking about tokens
session = await agent.create_session()
for i in range(100):  # Any length!
    await session.send(f"Message {i}")
    await session.receive()
```

### 🛠️ Tool Framework

Register Python functions as tools with a simple decorator:

```python
@agent.tool("search_docs")
async def search_docs(query: str, limit: int = 5) -> list:
    """Search documentation for relevant information"""
    results = await search_engine.search(query, limit)
    return results

# Agent automatically calls tools when needed
response = await agent.query("Find docs about async programming")
```

Features:
- Automatic parameter extraction from function signatures
- Support for both sync and async functions
- Timeout handling and retries
- Parallel tool execution

### 💬 Session Management

Maintain stateful conversations with automatic context preservation:

```python
session = await agent.create_session()

# Context is preserved across turns
await session.send("I'm working on a Python project")
await session.send("How do I use async/await?")  # Knows the context
await session.send("Show me an example")  # Still knows the context

# Summarize the conversation
summary = await session.summarize()

# Persist and restore
await session.save()
session = await agent.load_session(session.id)
```

### 💾 Memory & Persistence

SQLite-backed storage for interaction history:

```python
# Automatic interaction logging
agent = Agent(model="gpt-4", options=AgentOptions(enable_memory=True))

# Search past interactions
results = await agent.memory_store.search_interactions("machine learning")

# Get statistics
stats = await agent.memory_store.get_statistics()
print(f"Total interactions: {stats['total_interactions']}")
```

## Architecture

```
┌─────────────────────────────────────┐
│   Your Application / Agent Code     │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│      Any-Agent SDK (This Layer)     │
│  - Agent query orchestration        │
│  - Automatic context management     │
│  - Tool execution framework         │
│  - Session state management         │
│  - Memory persistence               │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│         LiteLLM (Dependency)        │
│  - Provider abstraction (100+)      │
│  - OpenAI-compatible interface      │
│  - Model routing & fallback         │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  Providers (OpenAI, Anthropic, etc) │
└─────────────────────────────────────┘
```

## Supported Providers

Thanks to LiteLLM, Any-Agent works with 100+ providers including:

**Cloud Providers:**
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude 3)
- Google (Gemini)
- Azure OpenAI
- AWS Bedrock
- Cohere
- Together AI
- Anyscale

**Local & Open Source:**
- Ollama
- LM Studio
- llama.cpp
- vLLM
- Text Generation Inference

See [LiteLLM's provider docs](https://docs.litellm.ai/docs/providers) for the complete list.

## Documentation

- **[Technical Design](docs/technical-design.md)** - Complete architecture and design decisions
- **[Implementation Plan](docs/implementation.md)** - Phase-by-phase development roadmap
- **[API Reference](#)** - Coming soon
- **[Examples](examples/)** - Sample agents and use cases

## Development Status

🚧 **Currently in active development** - Phase 1 (Foundation)

### Roadmap

- [x] Project planning and architecture
- [x] Technical design documentation
- [ ] **Phase 1**: Core Agent class with LiteLLM integration
- [ ] **Phase 2**: Context management with token handling
- [ ] **Phase 3**: Tool framework and execution engine
- [ ] **Phase 4**: Session management with persistence
- [ ] **Phase 5**: Memory store and interaction history
- [ ] **Phase 6**: Production polish and PyPI release

See [Implementation Plan](docs/implementation.md) for detailed milestones.

## Contributing

Contributions are welcome! This project is in early development, and we're open to:

- Feature suggestions and feedback
- Bug reports and fixes
- Documentation improvements
- Example agents and use cases

Please open an issue to discuss major changes before submitting a PR.

## Comparison with Alternatives

### vs LangChain

**LangChain** is a comprehensive framework with extensive integrations but can be complex for simple use cases.

**Any-Agent** focuses on simplicity and Claude SDK-style ergonomics, making it ideal for developers who want:
- Quick agent prototyping
- Automatic context management
- Minimal boilerplate
- Easy provider switching

### vs Building Your Own

**Any-Agent** provides production-ready patterns that would take weeks to build:
- Token counting across different model families
- Context management strategies
- Session persistence
- Tool orchestration
- Error handling and retries

Save time and use battle-tested components instead of reinventing the wheel.

## Requirements

- Python 3.10 or higher
- LiteLLM 1.51.0+
- aiosqlite 0.19.0+
- tiktoken 0.7.0+
- pydantic 2.0+

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on top of [LiteLLM](https://github.com/BerriAI/litellm) for provider abstraction
- Inspired by [Claude Agent SDK](https://github.com/anthropics/anthropic-sdk-python) patterns
- Token counting powered by [tiktoken](https://github.com/openai/tiktoken)

## Support

- 📖 [Documentation](docs/)
- 💬 [GitHub Discussions](#) - Coming soon
- 🐛 [Issue Tracker](https://github.com/yourusername/any-agent/issues)
- 📧 Email: your.email@example.com

---

**Status**: 🚧 Pre-alpha - API is subject to change

Star ⭐ this repo to follow development progress!
