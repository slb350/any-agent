"""
Context management utilities for manual conversation history management.

This module provides opt-in, low-level utilities for managing conversation
context and message history. The SDK deliberately does NOT automatically
manage context - you have full control over when and how to handle it.

Design Philosophy - Manual Context Management:
==============================================================================
The SDK follows an explicit, opt-in approach to context management:

1. NO AUTOMATIC TRUNCATION: The SDK never automatically removes messages
   from history. Your conversation history grows unbounded unless YOU
   explicitly manage it.

2. NO AUTOMATIC TOKEN COUNTING: The SDK doesn't track token usage or
   warn when approaching limits. YOU decide when to check.

3. NO MAGIC BEHAVIOR: All context management is explicit function calls.
   No background processes or hidden state mutations.

Why Manual Instead of Automatic?
- Different use cases need different strategies (sliding window, summarization,
  RAG-style retrieval, semantic search, etc.)
- Token estimation varies widely across model families (GPT vs Llama vs Qwen)
- Context limits differ by model (4k, 8k, 32k, 128k, unlimited)
- Some agents need full history (debugging, audit), others need minimal context
- Explicit > Implicit: predictable, debuggable, no surprises

This module provides BUILDING BLOCKS for implementing your own context
management strategy, not a one-size-fits-all solution.

Utilities Provided:
===================
1. estimate_tokens(messages, model)
   - Rough token count estimation using tiktoken (if available)
   - Falls back to character-based approximation (1 token ≈ 4 chars)
   - Use for: Checking if approaching context limits before API calls

2. truncate_messages(messages, keep, preserve_system)
   - Simple sliding window truncation (keep N most recent messages)
   - Always preserves system prompt (configurable)
   - Use for: Basic context management when history grows too large

Common Usage Patterns:
======================

Pattern 1: Periodic Truncation
>>> from open_agent import Client
>>> from open_agent.context import estimate_tokens, truncate_messages
>>>
>>> # Check token count periodically
>>> async with Client(options) as client:
...     await client.query("User prompt")
...     async for block in client.receive_messages():
...         print(block)
...
...     # Manually truncate when needed
...     if estimate_tokens(client.message_history) > 28000:
...         client.message_history = truncate_messages(client.message_history, keep=10)

Pattern 2: Stateless Agents (Minimal Context)
>>> # Only keep last user/assistant pair for stateless interactions
>>> async with Client(options) as client:
...     for user_input in inputs:
...         client.message_history = truncate_messages(client.message_history, keep=2)
...         await client.query(user_input)

Pattern 3: RAG-lite (Retrieve then Truncate)
>>> # Keep only system prompt + retrieved context + current query
>>> retrieved_context = search_knowledge_base(user_query)
>>> client.message_history = [
...     {"role": "system", "content": f"{system_prompt}\\n\\nContext: {retrieved_context}"},
... ]
>>> await client.query(user_query)

Pattern 4: Custom Truncation Strategy
>>> # Implement your own logic (preserve tool chains, important messages, etc.)
>>> def smart_truncate(messages, max_tokens=30000):
...     if estimate_tokens(messages) <= max_tokens:
...         return messages
...     # Custom logic: preserve system, recent tool chains, important context
...     # ... your implementation ...

For more examples, see: examples/context_management.py

Implementation Notes:
=====================
- estimate_tokens() uses tiktoken for GPT models if available
- For local LLMs (Llama, Qwen, etc.), it's only 70-85% accurate
- Always include 10-20% safety margin when checking limits
- truncate_messages() is simple - doesn't preserve tool call chains
- Helper functions (_iter_all_strings, _iter_string_values) recursively
  extract text from nested message structures for token estimation
"""

import math
from typing import Any, Iterable


def estimate_tokens(
    messages: list[dict[str, Any]], model: str = "gpt-3.5-turbo"
) -> int:
    """
    Estimate token count for a list of messages in OpenAI format.

    Provides rough token count estimation using two methods:
    1. tiktoken library (accurate for GPT models, if installed)
    2. Character-based fallback (1 token ≈ 4 characters)

    This is used to check if conversation history is approaching context limits
    before making API calls, allowing you to truncate or summarize proactively.

    Args:
        messages (list[dict[str, Any]]): List of message dictionaries in OpenAI
            chat format. Each message dict should have "role" and "content" keys.
            Example:
                [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there!"}
                ]

        model (str): Model name for tiktoken encoding selection. Default: "gpt-3.5-turbo"
            This parameter is only used when tiktoken is installed.
            Common values: "gpt-3.5-turbo", "gpt-4", "gpt-4-32k"
            For unrecognized models, falls back to "cl100k_base" encoding.

    Returns:
        int: Estimated token count including:
            - Content tokens (text encoded)
            - Message formatting overhead (~4 tokens per message)
            - Role tokens
            - Conversation-level overhead (~2 tokens)

    Accuracy by Model Family:
        - GPT models (with tiktoken): ~90-95% accurate
        - Llama models: ~70-85% accurate (different tokenizer)
        - Qwen models: ~70-85% accurate (different tokenizer)
        - Mistral models: ~70-85% accurate (different tokenizer)
        - Without tiktoken (char-based): ~60-80% accurate

    Safety Margin:
        Always add a 10-20% safety margin when checking limits:
        >>> limit = 32000
        >>> safe_limit = int(limit * 0.8)  # 80% of limit
        >>> if estimate_tokens(messages) > safe_limit:
        ...     # Truncate or summarize

    Examples:
        Basic usage:
        >>> messages = [
        ...     {"role": "system", "content": "You are a helpful assistant"},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> tokens = estimate_tokens(messages)
        >>> print(f"Estimated tokens: {tokens}")
        Estimated tokens: 23

        Check before API call:
        >>> if estimate_tokens(client.message_history) > 28000:
        ...     client.message_history = truncate_messages(
        ...         client.message_history, keep=10
        ...     )
        >>> await client.query("New message")

        Track token usage over conversation:
        >>> async with Client(options) as client:
        ...     for turn in range(100):
        ...         await client.query(f"Turn {turn}")
        ...         async for _ in client.receive_messages():
        ...             pass
        ...         tokens = estimate_tokens(client.message_history)
        ...         print(f"Turn {turn}: {tokens} tokens")

    Implementation Details:
        With tiktoken (preferred):
        - Loads encoding for specified model (or cl100k_base default)
        - Counts tokens for each message component
        - Adds 4 tokens overhead per message (OpenAI format)
        - Adds 1 token for role field
        - Adds 2 tokens for conversation-level overhead
        - Recursively handles nested content (tool calls, etc.)

        Without tiktoken (fallback):
        - Sums all character lengths in all messages
        - Divides by 4 (rough average: 1 token = 4 chars in English)
        - Uses math.ceil for conservative estimate (rounds up)

    Note on Local LLMs:
        Local models (Llama, Qwen, Mistral) use different tokenizers than GPT.
        tiktoken's estimate will be approximate but useful for rough limits.
        Consider these differences when setting context limits.

    Installing tiktoken:
        ```bash
        pip install open-agent-sdk[context]
        # or directly (see pyproject.toml for current version requirements)
        pip install tiktoken
        ```

    See Also:
        - truncate_messages(): Use this to reduce history when approaching limits
        - examples/context_management.py: Comprehensive usage patterns
    """
    try:
        # Try to use tiktoken for accurate token counting (GPT tokenizer)
        import tiktoken

        try:
            # Get encoding for specified model (e.g., "gpt-3.5-turbo" -> cl100k_base)
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Model not recognized by tiktoken (e.g., local model names)
            # Fall back to cl100k_base (used by GPT-3.5-turbo, GPT-4)
            encoding = tiktoken.get_encoding("cl100k_base")

        # Count tokens following OpenAI's message format overhead rules
        # See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        num_tokens = 0

        for message in messages:
            # Every message has formatting overhead (start/end markers)
            num_tokens += 4  # Message structure overhead tokens

            for key, value in message.items():
                # Special handling for role field
                if key == "role" and isinstance(value, str):
                    # Role field has 1 extra token + encoded role value
                    num_tokens += len(encoding.encode(value)) + 1
                    continue

                # For all other fields (content, name, etc.), encode text values
                # This recursively handles nested structures (tool calls with nested dicts/lists)
                for text_value in _iter_string_values(value):
                    num_tokens += len(encoding.encode(text_value))

        # Add conversation-level overhead (start/end of conversation markers)
        num_tokens += 2
        return num_tokens

    except ImportError:
        # tiktoken not installed - use character-based fallback
        # This is LESS accurate but still useful for rough estimates
        # Approximation: 1 token ≈ 4 characters (for English text)
        # WARNING: This ratio is calibrated for English. Non-English text
        # (especially Chinese/Japanese/Korean) may have significantly different
        # token-to-character ratios. For production multilingual apps, install
        # tiktoken for accurate counting: pip install open-agent-sdk[context]

        # Extract all string values from all messages (recursively)
        total_chars = sum(len(text) for text in _iter_all_strings(messages))

        if total_chars == 0:
            return 0

        # Divide by 4 and round up for conservative estimate
        # Using math.ceil ensures we err on the side of caution
        return math.ceil(total_chars / 4)


def truncate_messages(
    messages: list[dict[str, Any]], keep: int = 10, preserve_system: bool = True
) -> list[dict[str, Any]]:
    """Truncate message history, keeping recent messages.

    Always preserves the system prompt (if present) and keeps the most
    recent N messages. This is a simple truncation - it does NOT attempt
    to preserve tool chains or important context.

    Args:
        messages: List of message dicts in OpenAI format
        keep: Number of recent messages to keep (default: 10)
        preserve_system: Keep system message if present (default: True)

    Returns:
        Truncated message list (new list, original unchanged)

    Examples:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful"},
        ...     {"role": "user", "content": "Message 1"},
        ...     {"role": "assistant", "content": "Response 1"},
        ...     {"role": "user", "content": "Message 2"},
        ...     {"role": "assistant", "content": "Response 2"},
        ...     # ... many more messages ...
        ... ]
        >>> truncated = truncate_messages(messages, keep=2)
        >>> len(truncated)  # system + last 2 messages = 3
        3

        >>> # Manual truncation when needed
        >>> from open_agent.context import estimate_tokens, truncate_messages
        >>> if estimate_tokens(client.history) > 28000:
        ...     client.message_history = truncate_messages(client.history, keep=10)

    Note:
        This is a SIMPLE truncation. For domain-specific needs (e.g.,
        preserving tool call chains, keeping important context), implement
        your own logic or use this as a starting point.

        Warning: Truncating mid-conversation may remove context that the
        model needs to properly respond. Use judiciously at natural breakpoints.
    """
    if not messages:
        return []

    if len(messages) <= keep:
        return messages.copy()

    # Check if first message is system prompt
    has_system = (
        preserve_system and messages and messages[0].get("role") == "system"
    )

    if has_system:
        # Keep system + last N messages
        system_msg = [messages[0]]
        if keep > 0:
            recent = messages[-keep:]
            return system_msg + recent
        else:
            return system_msg
    else:
        # Just keep last N messages
        if keep > 0:
            return messages[-keep:]
        else:
            return []


# Public API exports for this module
# Only expose the two main utility functions to users
__all__ = ["estimate_tokens", "truncate_messages"]


# Private helper functions for token estimation
# These recursively extract all text from message structures

def _iter_all_strings(messages: list[dict[str, Any]]) -> Iterable[str]:
    """
    Yield all string values from a list of message dictionaries.

    Helper function for character-based token estimation fallback.
    Recursively extracts every string from the entire message list,
    regardless of nesting level or structure.

    Args:
        messages: List of message dicts (OpenAI format)

    Yields:
        str: Each string value found in the messages

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": [{"text": "Hi"}]}
        ... ]
        >>> list(_iter_all_strings(messages))
        ['user', 'Hello', 'assistant', 'Hi']
    """
    for message in messages:
        for value in message.values():
            # Recursively yield strings from this message's values
            yield from _iter_string_values(value)


def _iter_string_values(value: Any) -> Iterable[str]:
    """
    Recursively yield all string values from nested data structures.

    Helper function that handles arbitrarily nested combinations of dicts,
    lists, and strings. Used to extract all text content for token estimation
    when tiktoken is not available.

    Args:
        value: Any Python value (str, dict, list, or other)

    Yields:
        str: Each string found recursively in the structure

    Examples:
        >>> list(_iter_string_values("hello"))
        ['hello']

        >>> list(_iter_string_values({"a": "hello", "b": {"c": "world"}}))
        ['hello', 'world']

        >>> list(_iter_string_values(["hello", {"nested": "world"}]))
        ['hello', 'world']

    Implementation:
        - str: Yield immediately
        - dict: Recursively process all values (ignore keys)
        - list: Recursively process all items
        - other types: Ignored (numbers, None, etc.)
    """
    if isinstance(value, str):
        # Base case: found a string, yield it
        yield value
    elif isinstance(value, dict):
        # Recursive case: dict - process all values (keys are not tokenized)
        for nested in value.values():
            yield from _iter_string_values(nested)
    elif isinstance(value, list):
        # Recursive case: list - process all items
        for item in value:
            yield from _iter_string_values(item)
    # Other types (int, float, None, etc.) are ignored - not tokenized
