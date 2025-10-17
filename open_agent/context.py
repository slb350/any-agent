"""Context management utilities for manual history management.

This module provides low-level helpers for managing conversation history.
These are opt-in utilities - nothing is automatic. You decide when and how
to manage context.

Utilities:
    estimate_tokens: Rough token count estimation
    truncate_messages: Simple message history truncation

Patterns:
    See examples/context_management.py for usage patterns and examples.
"""

import math
from typing import Any, Iterable


def estimate_tokens(
    messages: list[dict[str, Any]], model: str = "gpt-3.5-turbo"
) -> int:
    """Estimate token count for message list.

    Uses tiktoken if available, otherwise falls back to character-based
    approximation (1 token ≈ 4 characters).

    Args:
        messages: List of message dicts in OpenAI format
            Example: [{"role": "user", "content": "Hello"}]
        model: Model name for tiktoken encoding (default: gpt-3.5-turbo)
            Only used if tiktoken is installed

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
        >>> tokens = estimate_tokens(messages)
        >>> print(f"Estimated tokens: {tokens}")
        Estimated tokens: 23

        >>> # Check if approaching context limit
        >>> if estimate_tokens(client.history) > 28000:
        ...     print("Need to truncate!")
    """
    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Model not recognized, use default encoding
            encoding = tiktoken.get_encoding("cl100k_base")

        # Count tokens for each message
        # OpenAI's format: each message has overhead + content tokens
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # Message formatting overhead
            for key, value in message.items():
                if key == "role" and isinstance(value, str):
                    num_tokens += len(encoding.encode(value)) + 1  # Role token
                    continue

                for text_value in _iter_string_values(value):
                    num_tokens += len(encoding.encode(text_value))
        num_tokens += 2  # Conversation-level overhead
        return num_tokens

    except ImportError:
        # Fallback: character-based approximation
        # Rough estimate: 1 token ≈ 4 characters
        total_chars = sum(len(text) for text in _iter_all_strings(messages))
        if total_chars == 0:
            return 0
        # Character-based approximation (ceil for conservative estimate)
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


__all__ = ["estimate_tokens", "truncate_messages"]


def _iter_all_strings(messages: list[dict[str, Any]]) -> Iterable[str]:
    """Yield all string values from a list of message dicts."""
    for message in messages:
        for value in message.values():
            yield from _iter_string_values(value)


def _iter_string_values(value: Any) -> Iterable[str]:
    """Recursively yield string values from nested structures."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_string_values(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_string_values(item)
