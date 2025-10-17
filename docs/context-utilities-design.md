# Context Utilities Design (Lean Approach)

## Decision: Opt-In Utilities, Not Automatic Management

After evaluating the complexity vs. benefit tradeoff, we're taking a **lean approach**:

**What we're building:**
- Low-level helper utilities for manual history management
- `estimate_tokens(messages)` - Rough token counting
- `truncate_messages(messages, keep=N)` - Simple truncation helper
- Clear documentation on manual management patterns

**What we're NOT building:**
- Automatic context compaction
- ContextManager wired into Client
- Complex strategies (importance, summarization)
- PreCompact hooks

## Rationale

### Why NOT Automatic?

1. **Domain-specific needs**: Copy editor needs different compaction than market analyst
2. **Token accuracy issues**: Each model family has different tokenizers (Qwen, Llama, Mistral)
3. **Complexity vs. simplicity**: Our value prop is "claude-agent-sdk patterns, but simple"
4. **Risk of breaking agents**: Silently removing context could break tool chains or lose critical info
5. **Natural limits exist anyway**: Local models have 8k-32k context - compaction doesn't change that

### Why Provide Utilities?

1. **Real problem exists**: Long conversations do hit limits
2. **Avoid reinvention**: Better to provide tested primitives than have everyone write buggy versions
3. **Empowers power users**: Give tools, let them decide when/how to use
4. **Migration path**: If demand emerges, we can build on these primitives later

## Implementation

### 1. Token Estimation (`estimate_tokens`)

**Goal**: Provide rough token count approximation

**Approach**: Try tiktoken, fallback to character-based estimate

```python
def estimate_tokens(
    messages: list[dict[str, Any]],
    model: str = "gpt-3.5-turbo"
) -> int:
    """
    Estimate token count for message list.

    Uses tiktoken if available, otherwise falls back to character-based
    approximation (1 token ≈ 4 characters).

    Args:
        messages: List of message dicts (OpenAI format)
        model: Model name for tiktoken encoding (default: gpt-3.5-turbo)

    Returns:
        Estimated token count

    Note:
        This is an APPROXIMATION. Actual token counts vary by model family:
        - GPT models: ~90-95% accurate with tiktoken
        - Llama, Qwen, Mistral: ~70-85% accurate (different tokenizers)
        - Always include 10-20% safety margin when checking limits

    Examples:
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant"},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> estimate_tokens(messages)
        23
    """
    try:
        import tiktoken
        encoding = tiktoken.encoding_for_model(model)

        # Count tokens for each message
        # OpenAI's format: each message has 4 tokens overhead + content tokens
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # Message overhead
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
                    if key == "role":
                        num_tokens += 1  # Role token
        num_tokens += 2  # Conversation overhead
        return num_tokens

    except (ImportError, KeyError):
        # Fallback: character-based approximation
        # Rough estimate: 1 token ≈ 4 characters
        total_chars = 0
        for message in messages:
            for value in message.values():
                if isinstance(value, str):
                    total_chars += len(value)
        return total_chars // 4
```

### 2. Message Truncation (`truncate_messages`)

**Goal**: Simple helper to truncate old messages

**Approach**: Always preserve system prompt + keep last N messages

```python
def truncate_messages(
    messages: list[dict[str, Any]],
    keep: int = 10,
    preserve_system: bool = True
) -> list[dict[str, Any]]:
    """
    Truncate message history, keeping recent messages.

    Always preserves the system prompt (if present) and keeps the most
    recent N messages. This is a simple truncation - it does NOT attempt
    to preserve tool chains or important context.

    Args:
        messages: List of message dicts (OpenAI format)
        keep: Number of recent messages to keep (default: 10)
        preserve_system: Keep system message if present (default: True)

    Returns:
        Truncated message list

    Examples:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful"},
        ...     {"role": "user", "content": "Message 1"},
        ...     {"role": "assistant", "content": "Response 1"},
        ...     # ... 50 more messages ...
        ...     {"role": "user", "content": "Latest message"}
        ... ]
        >>> truncated = truncate_messages(messages, keep=5)
        >>> len(truncated)  # system + 5 recent = 6
        6

    Note:
        This is a SIMPLE truncation. For domain-specific needs (e.g.,
        preserving tool call chains, keeping important context), implement
        your own logic or use this as a starting point.
    """
    if len(messages) <= keep:
        return messages.copy()

    # Check if first message is system prompt
    has_system = (
        preserve_system and
        messages and
        messages[0].get("role") == "system"
    )

    if has_system:
        # Keep system + last N messages
        system_msg = [messages[0]]
        recent = messages[-keep:]
        return system_msg + recent
    else:
        # Just keep last N messages
        return messages[-keep:]
```

### 3. Utilities Module

**Location**: `open_agent/context.py`

**Exports**:
```python
# open_agent/context.py
"""Context management utilities for manual history management."""

__all__ = ["estimate_tokens", "truncate_messages"]

# Implementation here...
```

**Usage**:
```python
from open_agent import Client, AgentOptions
from open_agent.context import estimate_tokens, truncate_messages

async with Client(options) as client:
    # Long conversation...
    for i in range(50):
        await client.query(f"Question {i}")
        async for msg in client.receive_messages():
            pass

    # Check token count
    estimated = estimate_tokens(client.history)
    print(f"Current context: ~{estimated} tokens")

    # Manually truncate if needed
    if estimated > 28000:
        print("Approaching limit, truncating history...")
        client.message_history = truncate_messages(client.history, keep=10)
```

## Documentation Patterns

### Pattern 1: Stateless Agents (Recommended)

**Best for**: Single-task agents (copy editor, code formatter, etc.)

```python
# Process each chapter independently - no history accumulation
for chapter in chapters:
    async with Client(options) as client:
        await client.query(f"Edit chapter: {chapter}")
        result = await get_response(client)
        save_result(result)
    # Client disposed, fresh context for next chapter
```

### Pattern 2: Manual Truncation

**Best for**: Multi-turn conversations with natural breakpoints

```python
from open_agent.context import truncate_messages

async with Client(options) as client:
    for task in tasks:
        await client.query(task)
        async for msg in client.receive_messages():
            process(msg)

        # Truncate after each major task
        client.message_history = truncate_messages(client.history, keep=5)
```

### Pattern 3: Token Budget Monitoring

**Best for**: Long sessions with periodic checks

```python
from open_agent.context import estimate_tokens, truncate_messages

MAX_TOKENS = 28000  # Safety margin below 32k limit

async with Client(options) as client:
    while True:
        user_input = get_user_input()
        await client.query(user_input)

        # Check token budget periodically
        if estimate_tokens(client.history) > MAX_TOKENS:
            print("Context limit approaching, truncating...")
            client.message_history = truncate_messages(client.history, keep=10)
```

### Pattern 4: External Memory (RAG-lite)

**Best for**: Research agents, knowledge accumulation

```python
# Save important facts to database instead of keeping in context
database = {}

async with Client(options) as client:
    await client.query("Research topic X")
    async for msg in client.receive_messages():
        if is_important_fact(msg):
            database[extract_key(msg)] = msg

    # Clear history, query database when needed
    client.message_history = truncate_messages(client.history, keep=3)

    await client.query(f"Using these facts: {database}, analyze Y")
```

## Dependencies

**tiktoken**: Optional dependency for better token estimation

```toml
# pyproject.toml
[project.optional-dependencies]
context = ["tiktoken>=0.5.0"]
```

**Install**:
```bash
# Basic install (uses character-based estimation)
pip install open-agent-sdk

# With tiktoken for better estimation
pip install open-agent-sdk[context]
```

## Testing

**Unit tests**:
- `test_estimate_tokens_with_tiktoken` - When tiktoken available
- `test_estimate_tokens_fallback` - Character-based fallback
- `test_truncate_messages_basic` - Simple truncation
- `test_truncate_messages_preserve_system` - System prompt preservation
- `test_truncate_messages_no_system` - Without system prompt

**Integration tests**:
- Long conversation example (50+ turns)
- Token estimation accuracy (compare with actual API)
- Manual truncation workflow

## Future Extensions (If Needed)

If real demand emerges for automatic management, we can build on these primitives:

1. **Auto-truncate option in Client**:
   ```python
   options = AgentOptions(
       model="qwen2.5-32b",
       auto_truncate=AutoTruncate(
           max_tokens=28000,
           keep_recent=10,
           strategy=truncate_messages  # User-provided strategy
       )
   )
   ```

2. **Custom truncation strategies**:
   ```python
   def preserve_tool_chains(messages, keep):
       # Custom logic to not split tool use/result pairs
       pass

   options.auto_truncate.strategy = preserve_tool_chains
   ```

But we don't build this until users actually ask for it.

## Success Criteria

✅ Users can manually manage history with simple helpers
✅ Token estimation is "good enough" with clear accuracy warnings
✅ Documentation shows 4+ real-world patterns
✅ No silent mutations - users always in control
✅ Migration path exists for future automation if needed

## Summary

**Philosophy**: Provide primitives, document patterns, trust users to make domain-specific decisions.

**Result**: Lean SDK that acknowledges real limits without trying to magically solve them.
