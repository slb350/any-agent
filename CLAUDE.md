# Any-Agent SDK Project

## Project Overview

**Goal**: Claude Agent SDK-style API for local/self-hosted LLMs via OpenAI-compatible endpoints.

**Problem**: I love the `claude-agent-sdk` workflow for building agents, but want to use my local Qwen/Llama/Mistral models instead of paying for Claude API.

**Solution**: A lightweight SDK that provides the same ergonomics as `claude-agent-sdk`, but for OpenAI-compatible local model servers.

```
┌─────────────────────────────────────┐
│   Your Application / Agent Code     │
│   (Copy Editor, Market Analysis)    │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│      Any-Agent SDK (This Project)   │
│  - query() function                 │
│  - Client class (multi-turn)        │
│  - AgentOptions                     │
│  - Tool monitoring/streaming        │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│      OpenAI Python Client           │
│  (for async streaming)              │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  Local Model Servers                │
│  - LM Studio (localhost:1234)       │
│  - Ollama (localhost:11434)         │
│  - llama.cpp server                 │
│  - vLLM, Text Gen WebUI, etc.       │
└─────────────────────────────────────┘
```

## Scope: In vs Out

### Supported Providers (In Scope)
✅ **LM Studio** - http://localhost:1234/v1
✅ **Ollama** - http://localhost:11434/v1
✅ **llama.cpp server** - OpenAI-compatible mode
✅ **vLLM** - OpenAI-compatible API
✅ **Text Generation WebUI** - OpenAI extension
✅ **Any OpenAI-compatible local endpoint**
✅ **Local gateways proxying cloud models** (e.g., Ollama or other OpenAI-compatible gateways configured to call remote models)

### NOT Supported (Out of Scope)
❌ **Direct Claude/OpenAI SDK usage** - Use their official SDKs, unless you route through a local OpenAI-compatible gateway
❌ **Direct Cloud Provider SDKs** - Bedrock, Vertex, Azure, etc. (proxied via a local OpenAI-compatible gateway is fine)

This SDK is **specifically for local/self-hosted open-source models**.

### Core Features (In Scope)
🎯 **Simple query() function** - Single-turn requests
🎯 **Client class** - Multi-turn conversations with tool monitoring
🎯 **AgentOptions** - Configuration matching claude-agent-sdk
🎯 **Streaming support** - Async iteration over responses
🎯 **Tool use monitoring** - Track what tools the agent calls
🎯 **Message types** - Compatible with claude-agent-sdk patterns

### Advanced Features (Maybe Later)
⏳ Context window management
⏳ Automatic message history trimming
⏳ Optional storage helpers (as extras)

### Non-Features (Agent Responsibility)
🚫 **Persistent storage/memory** - Agents handle their own SQLite/DB (like your copy_editor.database)
🚫 **Tool execution** - Agents decide when/how to execute tools
🚫 **RAG/embeddings** - Domain-specific, agents implement as needed

## Core Architecture

### API Design (Matching claude-agent-sdk)

**Simple Query (like your copy_editor agent):**
```python
from any_agent import query, AgentOptions

# Configure for your local LM Studio instance
options = AgentOptions(
    system_prompt="You are a professional copy editor...",
    model="qwen2.5-32b-instruct",
    base_url="http://localhost:1234/v1",
    max_turns=1,
    temperature=0.1
)

# Query the model (streaming response)
result = query(prompt=user_prompt, options=options)

# Extract response
response_text = ""
async for msg in result:
    response_text += extract_response_text(msg)
```

**Multi-Turn Client (like your market_analysis agent):**
```python
from any_agent import Client, AgentOptions

options = AgentOptions(
    system_prompt="You are a market analyst...",
    model="llama-3.1-70b",
    base_url="http://localhost:1234/v1",
    max_turns=50
)

async with Client(options) as client:
    # Send initial query
    await client.query(user_prompt)

    # Process streaming responses
    async for msg in client.receive_messages():
        if isinstance(msg, TextBlock):
            print(f"Agent: {msg.text}")

        elif isinstance(msg, ToolUseBlock):
            # Log tool usage
            log_tool_use(msg.name, msg.input)
```

**Configuration Options:**
```python
class AgentOptions:
    system_prompt: str                # System prompt
    model: str                        # Model name (e.g., "qwen2.5-32b-instruct")
    base_url: str                     # Endpoint (e.g., "http://localhost:1234/v1")
    max_turns: int = 1                # Max conversation turns
    max_tokens: int | None = 4096     # Tokens to generate (None uses provider default)
    temperature: float = 0.7          # Sampling temperature
    api_key: str = "not-needed"       # Most local servers don't need this
```

### Key Components

1. **client.py** - Main query() function and Client class
   - `query()` - Single-turn requests
   - `Client` - Multi-turn conversations with context
   - Streaming response handling

2. **types.py** - Message types and options
   - `AgentOptions` - Configuration dataclass
   - `TextBlock` - Text content from model
   - `ToolUseBlock` - Tool calls from model
   - `ToolUseError` - Tool call parsing error
   - `AssistantMessage` - Full message wrapper

3. **utils.py** - OpenAI client helpers
   - Create AsyncOpenAI client
   - Format messages for API
   - Extract response blocks

## Storage & Memory Philosophy

**SDK does NOT provide**: Built-in database, session persistence, or memory management.

**Why**: Your agents have domain-specific storage needs:
- Copy Editor → Issues by chapter, severity trends, run history
- Style Analyzer → Voice patterns, motif tracking, drift detection
- Market Analysis → Comp titles, research sources, search history

**What SDK provides**: Conversation primitives for easy storage:
```python
# Client exposes history for agent to store
class Client:
    @property
    def history(self) -> list[dict]:
        """Full conversation history"""
        return self.message_history.copy()

    @property
    def turn_metadata(self) -> dict:
        """Metadata about conversation state"""
        return {
            "turn_count": self.turn_count,
            "started_at": self.started_at
        }
```

**Your agent's responsibility**:
```python
from any_agent import Client, AgentOptions
from copy_editor.database import CopyEditDatabase  # Your custom DB

class CopyEditorAgent:
    def __init__(self, novel, config):
        self.db = CopyEditDatabase(db_path)  # Domain-specific schema

    async def run_analysis(self):
        async with Client(options) as client:
            await client.query(prompt)

            async for block in client.receive_messages():
                # Process blocks
                pass

            # Save results with your custom logic
            self.db.add_issue(run_id, issue_data)
            # Or save conversation: self.db.save_conversation(client.history)
```

**Future possibility**: Optional `any_agent.extras.storage` helper for basic conversation storage:
```bash
pip install any-agent[storage]  # Adds aiosqlite
```

But most agents will have custom needs, so we don't force a choice.

## Project Structure

```
any-agent/
├── any_agent/
│   ├── __init__.py      # Main exports: query, Client, AgentOptions
│   ├── client.py        # query() and Client class
│   ├── types.py         # Message types and AgentOptions
│   └── utils.py         # OpenAI client helpers
├── examples/
│   ├── simple_lmstudio.py      # Simple query example
│   ├── ollama_chat.py          # Ollama multi-turn
│   └── copy_editor_port.py     # Port of stories copy_editor agent
├── tests/
│   ├── test_query.py
│   └── test_client.py
├── pyproject.toml
├── README.md
└── LICENSE
```

## Development Environment

**Virtual Environment**: This project uses a venv in the project root (`./venv`)

**Important**: Always use the project venv for Python/pip commands:
```bash
# Activate venv (if needed for manual commands)
source venv/bin/activate

# Install package in development mode
./venv/bin/pip install -e .

# Install with dev dependencies
./venv/bin/pip install -e ".[dev]"

# Run tests
./venv/bin/pytest

# Run examples
./venv/bin/python examples/simple_lmstudio.py
```

**For Claude Code**: All Python/pip tool uses should reference `./venv/bin/python` or `./venv/bin/pip` explicitly.

## Implementation Plan

See `docs/implementation.md` for detailed implementation steps.

### Phase 1: Core MVP (Week 1)
- ✅ Project setup (pyproject.toml, structure)
- ✅ `types.py` - AgentOptions, message types
- ✅ `utils.py` - OpenAI client wrapper with ToolCallAggregator
- ✅ `client.py` - Simple `query()` function
- ✅ Test with LM Studio - verified working with network server

### Phase 2: Multi-Turn Support (Week 2)
- ✅ `Client` class for multi-turn conversations
- ✅ Tool use monitoring (via ToolUseBlock yielding)
- ✅ Message history tracking with OpenAI-compatible format
- ✅ ollama_chat.py example
- ✅ Multi-turn context verified working with LM Studio

### Phase 3: Polish & Port (Week 3)
- 🔨 Port copy_editor agent as validation
- 🔨 Documentation and examples
- 🔨 PyPI package setup

## Recommended Local Models

**Fast & Efficient:**
- Qwen 2.5 7B/14B/32B (excellent instruction following)
- Llama 3.2 3B/8B (good for simple tasks)
- Mistral 7B v0.3 (solid all-around)

**Larger/Better:**
- Qwen 2.5 72B (near GPT-4 quality)
- Llama 3.1 70B (very capable)
- DeepSeek-V3 (via gateway/proxy)

**For Copy Editing:**
- Qwen 2.5 32B+ (excellent for detailed analysis)
- Command-R+ 104B (via gateway/proxy)

## Success Criteria

The SDK is successful when:
- ✅ Drop-in replacement for `claude-agent-sdk` imports
- ✅ Works with LM Studio, Ollama, llama.cpp out of the box
- ✅ Copy Editor agent ports with < 5 lines changed
- ✅ Market Analysis agent ports with minimal changes
- ✅ Clear documentation with local model examples
- ✅ Can build new agents using familiar patterns

## Why Not Just Use OpenAI Client Directly?

**Without any-agent:**
```python
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
response = await client.chat.completions.create(
    model="qwen2.5-32b-instruct",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    stream=True
)

async for chunk in response:
    # Complex parsing of chunks
    # Extract delta content
    # Handle tool calls manually
```

**With any-agent:**
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
