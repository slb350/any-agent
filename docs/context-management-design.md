# Context Management Design

## Overview

Unlike Claude SDK (which delegates to Claude Code CLI), we need to implement context management ourselves for local LLMs. This document outlines our approach.

## Problem Statement

**Challenge**: Long conversations will eventually exceed model context windows, causing API failures.

**Solution**: Automatically compact conversation history when approaching limits while preserving conversation coherence.

## Architecture Comparison

### Claude SDK (Reference)
```
Python SDK → Claude Code CLI → Anthropic API
              ↑ Handles compaction
```
- CLI does: history tracking, token counting, auto-compaction
- SDK provides: `PreCompact` hook for interception

### Our SDK (What We Need)
```
Python SDK → OpenAI Client → Local LLM
↑ Must handle everything
```
- We must: track history, count tokens, compact messages
- No CLI layer to delegate to

## Token Counting Strategies

### Option 1: tiktoken (OpenAI Tokenizer)
**Pros:**
- Fast, battle-tested
- Accurate for GPT-based models
- Simple integration

**Cons:**
- Not accurate for non-OpenAI models (Qwen, Llama, Mistral)
- Each model family has different tokenizers
- May under/over-estimate by 10-30%

**Use case**: Quick approximation, acceptable margin of error

### Option 2: Model-Specific Tokenizers
**Pros:**
- Accurate for specific model families
- Examples: `transformers` library, `sentencepiece`

**Cons:**
- Heavy dependencies (GB of model files)
- Different tokenizer per model family
- Slow to load

**Use case**: When accuracy is critical

### Option 3: Character-Based Approximation
**Pros:**
- No dependencies
- Fast, simple
- Works for all models

**Cons:**
- Inaccurate (typically 1 char ≈ 0.25-0.3 tokens)
- 25-40% error margin

**Use case**: Fallback when no tokenizer available

### **Recommendation: Hybrid Approach**
```python
class TokenCounter:
    def __init__(self, method: Literal["tiktoken", "transformers", "chars"] = "tiktoken"):
        # Default to tiktoken (good enough for most)
        # Allow override for specific needs
        pass

    def count(self, text: str) -> int:
        # Return approximate token count
        pass
```

**Default**: `tiktoken` (fast, good enough approximation)
**Advanced**: Allow custom `TokenCounter` injection for specific models

## Compaction Strategies

### 1. Simple Truncation (PRIORITY 1)
**How it works:**
- Remove oldest messages first
- Always preserve: system prompt + last N messages

**Pros:**
- Simple, fast, predictable
- No LLM calls needed

**Cons:**
- Loses context from early conversation
- May break tool call chains

**Configuration:**
```python
class TruncationStrategy:
    keep_recent: int = 10  # Keep last N messages
    preserve_system: bool = True
    preserve_tool_chains: bool = True  # Don't break tool use/result pairs
```

**Example:**
```
System: You are a helpful assistant
User: Tell me about Paris
Assistant: Paris is the capital of France...
User: What about Berlin?  ← Keep from here (last 10 messages)
Assistant: Berlin is...
[... 40 more messages ...]
User: Compare Paris and Berlin  ← Current message
```

### 2. Importance-Based (PRIORITY 2)
**How it works:**
- Assign importance scores to messages
- Remove low-importance messages first
- Always keep: system, recent messages, tool chains

**Scoring heuristics:**
- User messages: high importance (they asked for it)
- Tool use blocks: high (part of workflow)
- Long assistant messages: medium (detailed responses)
- Short assistant messages: low (acknowledgments)
- Messages referenced in recent context: high

**Pros:**
- Smarter than truncation
- Preserves important context

**Cons:**
- More complex
- Heuristics may not always be right

**Configuration:**
```python
class ImportanceStrategy:
    keep_recent: int = 10
    user_message_score: float = 1.0
    assistant_message_score: float = 0.6
    tool_use_score: float = 1.0
    min_score_threshold: float = 0.3
```

### 3. Summarization-Based (PRIORITY 3)
**How it works:**
- Use LLM to summarize old conversation chunks
- Replace N old messages with 1 summary message
- Keep recent messages verbatim

**Pros:**
- Preserves semantic content
- Most intelligent approach

**Cons:**
- Requires LLM call (slow, costs tokens)
- Summary quality depends on model
- Complex to implement

**Configuration:**
```python
class SummarizationStrategy:
    keep_recent: int = 10
    summarize_chunk_size: int = 20  # Summarize 20 messages at a time
    max_summary_tokens: int = 500
    # User provides their own Client for summarization
    summarize_with: Client | None = None
```

**Example:**
```
System: You are a helpful assistant
[SUMMARY]: User asked about Paris, Berlin, and Rome. I provided capital city information and cultural highlights for each.
User: Now tell me about London  ← Keep from here (last 10 messages)
Assistant: London is...
[... recent conversation ...]
User: Compare all four cities  ← Current message
```

## API Design

### AgentOptions Configuration
```python
@dataclass
class ContextConfig:
    """Context management configuration"""

    # Token limits
    max_context_tokens: int = 32000  # Model's max context
    target_tokens: int = 28000  # Trigger compaction at this point (safety margin)

    # Compaction strategy
    strategy: Literal["truncation", "importance", "summarization"] = "truncation"

    # Strategy-specific options
    truncation_options: TruncationStrategy | None = None
    importance_options: ImportanceStrategy | None = None
    summarization_options: SummarizationStrategy | None = None

    # Token counting
    token_counter: TokenCounter | None = None  # Default: tiktoken

    # PreCompact hook
    on_pre_compact: Callable[[CompactContext], Awaitable[CompactDecision]] | None = None


@dataclass
class AgentOptions:
    # ... existing fields ...
    context: ContextConfig | None = None
```

### Client Integration
```python
class Client:
    def __init__(self, options: AgentOptions):
        self.options = options
        self.message_history: list[dict] = []
        self.context_manager = ContextManager(options.context)

    async def query(self, prompt: str) -> None:
        # Add user message
        self.message_history.append({"role": "user", "content": prompt})

        # Check if compaction needed
        if await self.context_manager.should_compact(self.message_history):
            # Trigger PreCompact hook if configured
            if self.options.context and self.options.context.on_pre_compact:
                decision = await self.options.context.on_pre_compact(
                    CompactContext(
                        trigger="auto",
                        current_tokens=self.context_manager.count_tokens(self.message_history),
                        message_count=len(self.message_history)
                    )
                )
                if decision.cancel:
                    # User cancelled compaction
                    pass
                elif decision.custom_instructions:
                    # User provided custom instructions
                    pass

            # Compact messages
            self.message_history = await self.context_manager.compact(
                self.message_history
            )

        # Send to API with compacted history
        response = await self.openai_client.chat.completions.create(
            model=self.options.model,
            messages=self.message_history,
            stream=True
        )
        # ... rest of implementation
```

### Hook Support
```python
@dataclass
class CompactContext:
    """Context provided to PreCompact hook"""
    trigger: Literal["manual", "auto"]
    current_tokens: int
    target_tokens: int
    message_count: int
    custom_instructions: str | None = None


@dataclass
class CompactDecision:
    """Decision from PreCompact hook"""
    cancel: bool = False  # Cancel compaction
    custom_instructions: str | None = None  # Custom compaction instructions
    # For summarization: provide custom summary
    custom_summary: str | None = None
```

## Implementation Phases

### Phase 1: Token Counting + Simple Truncation
**Goal**: Prevent context overflow with basic compaction

**Tasks:**
1. Add `tiktoken` dependency for token counting
2. Implement `TokenCounter` class with tiktoken backend
3. Implement `TruncationStrategy`
4. Add `ContextManager` class to Client
5. Track token counts in message history
6. Test with long conversations

**Success criteria:**
```python
options = AgentOptions(
    model="qwen2.5-32b",
    base_url="http://localhost:1234/v1",
    context=ContextConfig(
        max_context_tokens=32000,
        target_tokens=28000,
        strategy="truncation",
        truncation_options=TruncationStrategy(keep_recent=10)
    )
)

async with Client(options) as client:
    # Send 100 messages - should auto-compact
    for i in range(100):
        await client.query(f"Message {i}")
        async for block in client.receive_messages():
            pass

    # History should be compacted, not all 100 messages
    assert len(client.history) < 100
```

### Phase 2: PreCompact Hook
**Goal**: Allow interception before compaction

**Tasks:**
1. Add `on_pre_compact` callback to `ContextConfig`
2. Implement hook invocation in `ContextManager`
3. Add `CompactContext` and `CompactDecision` types
4. Test hook cancellation and custom instructions

**Success criteria:**
```python
async def on_compact(ctx: CompactContext) -> CompactDecision:
    print(f"About to compact: {ctx.current_tokens} tokens, {ctx.message_count} messages")
    # Allow but log it
    return CompactDecision(cancel=False)

options = AgentOptions(
    model="qwen2.5-32b",
    base_url="http://localhost:1234/v1",
    context=ContextConfig(
        strategy="truncation",
        on_pre_compact=on_compact
    )
)
```

### Phase 3: Importance-Based Strategy
**Goal**: Smarter message selection

**Tasks:**
1. Implement `ImportanceStrategy` with scoring heuristics
2. Add message importance scoring logic
3. Implement importance-based message selection
4. Test with various conversation patterns

### Phase 4: Summarization Strategy (Optional)
**Goal**: Preserve semantic content through summarization

**Tasks:**
1. Implement `SummarizationStrategy`
2. Add summarization logic using model
3. Handle summary message injection
4. Test summary quality and performance

## Testing Strategy

### Unit Tests
- Token counting accuracy (compare with tiktoken output)
- Truncation strategy correctness
- Importance scoring logic
- PreCompact hook invocation

### Integration Tests
- Long conversation (100+ messages)
- Tool call chain preservation
- Hook cancellation behavior
- Multiple compaction cycles

### Example Use Cases
1. **Chat agent** - 50+ turn conversation
2. **Copy editor** - Processing 20 chapters with context
3. **Code reviewer** - Multi-file review with accumulated context

## Open Questions

1. **Token counting accuracy**: Is tiktoken "good enough" for local models?
   - **Decision**: Yes, with 10-20% safety margin

2. **Compaction trigger point**: When to trigger?
   - **Decision**: At 80-90% of max_context_tokens (configurable)

3. **Tool chain preservation**: How to handle mid-chain compaction?
   - **Decision**: Never split tool_use and tool_result pairs

4. **Summarization model**: Same model or separate?
   - **Decision**: Let user provide their own Client for summarization

5. **System prompt preservation**: Always keep?
   - **Decision**: Yes, always preserve system prompt

## References

- Claude SDK `PreCompactHookInput`: reference/claude-agent-sdk-python/src/claude_agent_sdk/types.py:208
- Our current Client: open_agent/client.py
- tiktoken: https://github.com/openai/tiktoken
- transformers tokenizers: https://huggingface.co/docs/transformers/main_classes/tokenizer
