# Any-Agent Framework Project

## Project Overview

**Goal**: Build an Agent SDK layer that brings Claude Agent SDK-style capabilities to any LLM provider.

**Vision**: A lightweight, opinionated agent framework that sits on top of LiteLLM, adding the missing "agent" features:
- Context management (automatic token budget handling)
- Tool framework and registry
- Session management and conversation state
- Memory and persistence
- Claude SDK-style ergonomics

**Architecture Decision**: Use LiteLLM for provider abstraction (already solved), focus on agent-layer features.

```
┌─────────────────────────────────────┐
│   Your Application / Agent Code     │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│      Any-Agent SDK (This Project)   │
│  - Agent class with query()         │
│  - Context management               │
│  - Tool framework                   │
│  - Session management               │
│  - Memory/persistence               │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│         LiteLLM (Dependency)        │
│  - Provider abstraction (100+)      │
│  - OpenAI-compatible interface      │
│  - Model routing                    │
└─────────────────┬───────────────────┘
                  │
┌─────────────────▼───────────────────┐
│  Providers (OpenAI, Ollama, etc.)   │
└─────────────────────────────────────┘
```

## Scope: What We're Building vs Using

### Using LiteLLM For (Out of Scope)
✅ Provider routing and abstraction
✅ Model-specific API differences
✅ Rate limiting and retries (basic)
✅ OpenAI-compatible interface
✅ Support for 100+ providers

### Building in Any-Agent (In Scope)
🎯 **Agent Class**: Claude SDK-style API (`query()`, `AgentOptions`)
🎯 **Context Management**: Automatic token budget, sliding windows, message prioritization
🎯 **Tool Framework**: Tool registry, execution orchestration, result handling
🎯 **Session Management**: Stateful conversations, turn tracking, context restoration
🎯 **Memory Layer**: SQLite persistence, interaction history, retrieval
🎯 **Streaming Abstraction**: Unified streaming interface across providers

## Core Architecture

### API Design (Claude SDK-inspired)

```python
from any_agent import Agent, AgentOptions

# Simple stateless query
agent = Agent(model="gpt-4", provider="openai")
response = await agent.query("Analyze this text...")

# With options
agent = Agent(
    model="ollama/llama3.1",
    options=AgentOptions(
        system_prompt="You are a helpful assistant",
        max_tokens=4000,
        context_strategy="sliding",
        temperature=0.7
    )
)
response = await agent.query("Hello!")

# With tools
@agent.tool("search_docs")
async def search_docs(query: str) -> str:
    """Search documentation for relevant info"""
    return search_results

response = await agent.query("Find info about async in docs")

# Stateful session
session = await agent.create_session()
await session.send("What's the capital of France?")
response1 = await session.receive()

await session.send("What's its population?")  # Has context!
response2 = await session.receive()
```

### Key Components We're Building

1. **Agent Core** (`any_agent/agent.py`)
   - Query orchestration
   - Options management
   - LiteLLM integration wrapper

2. **Context Manager** (`any_agent/context.py`)
   - Token counting per model family
   - Message history management
   - Sliding window / truncation strategies

3. **Tool Framework** (`any_agent/tools.py`)
   - Tool decorator and registry
   - Execution engine
   - Result validation

4. **Session Manager** (`any_agent/session.py`)
   - Conversation state
   - Turn tracking
   - Context restoration

5. **Memory Store** (`any_agent/memory.py`)
   - SQLite persistence
   - Interaction logging
   - History retrieval

## Technical Considerations

### Token Counting
- Model-specific tokenizers (tiktoken for GPT, HuggingFace for open models)
- Context window management per model (4k-128k)

### Output Format
- JSON output with schema validation recommended for structured responses
- Streaming support for real-time output

### Local Model Support
- LM Studio (http://localhost:1234/v1)
- Ollama (http://localhost:11434)
- Recommended models: Llama 3.1, Qwen 2.5, Mistral/Mixtral

### Configuration
```yaml
analysis_defaults:
  provider: "openai"  # or "claude"
  model: "qwen2.5-32b-instruct"
  base_url: "http://localhost:1234/v1"
  temperature: 0.1
  max_tokens: 8000
```

## Implementation Roadmap

See `docs/implementation.md` for the complete phase-by-phase implementation plan with detailed steps, code examples, and success criteria.

### High-Level Phases

**Phase 1: Foundation**
- Core Agent class with LiteLLM integration
- Basic query functionality
- Provider detection and retry logic

**Phase 2: Context Management**
- Token counting per model
- Sliding window and truncation strategies
- Automatic context optimization

**Phase 3: Tool Framework**
- Tool registration and decorator
- Execution engine with timeout handling
- Parallel tool execution

**Phase 4: Session Management**
- Stateful conversation support
- Context preservation and restoration
- Session persistence

**Phase 5: Memory & Persistence**
- SQLite storage for interactions
- Search and analytics
- Interaction history

**Phase 6: Polish & Documentation**
- Production-ready error handling
- Comprehensive documentation
- Example agents and test suite

## Design Decisions to Make

### 1. Tool Invocation Pattern
**Option A**: Let LiteLLM handle function calling (if model supports it)
**Option B**: Build our own tool orchestration (emulate for models without native support)
**Decision needed**: Which approach for MVP?

### 2. Context Strategy Default
- Sliding window (keep recent N messages)
- Summarization (compress old messages)
- Truncation (drop oldest)
**Decision needed**: What's the sensible default?

### 3. Streaming API
```python
# Option A: Always async iterator
async for chunk in agent.query_stream(prompt):
    print(chunk)

# Option B: Flag on query()
response = await agent.query(prompt, stream=True)
```
**Decision needed**: Which feels more ergonomic?

### 4. Session Persistence
- Auto-save after every interaction (overhead)
- Manual save (requires `session.save()`)
- Lazy/background (eventual consistency)
**Decision needed**: Balance between safety and performance?

### 5. Project Structure
```
any-agent/
├── any_agent/
│   ├── __init__.py
│   ├── agent.py       # Core Agent class
│   ├── context.py     # ContextManager
│   ├── tools.py       # Tool framework
│   ├── session.py     # Session management
│   └── memory.py      # Persistence
├── examples/
│   ├── simple_query.py
│   ├── copy_editor.py
│   └── chatbot.py
├── tests/
├── pyproject.toml
└── README.md
```
**Decision needed**: Does this structure work?

## Success Criteria

The SDK is successful when:
- ✅ Simpler than LangChain for 80% of agent use cases
- ✅ Works with any LiteLLM-supported provider (100+)
- ✅ Automatic context management (developer doesn't think about tokens)
- ✅ Claude SDK-style ergonomics (familiar to Claude users)
- ✅ Copy Editor agent ports cleanly
- ✅ Can build production agents in < 100 lines of code
- ✅ Clear documentation and examples

## Value Proposition vs Alternatives

| Need | LiteLLM Alone | LangChain | Any-Agent |
|------|---------------|-----------|-----------|
| Switch providers | ✅ | ✅ | ✅ |
| Automatic context mgmt | ❌ | Manual | ✅ |
| Simple stateless queries | ✅ | Complex | ✅ |
| Stateful sessions | ❌ | Via memory | ✅ |
| Tool orchestration | ❌ | ✅ | ✅ |
| Learning curve | Low | High | Low |
| Lines of code (typical agent) | 50 | 150 | 30 |
