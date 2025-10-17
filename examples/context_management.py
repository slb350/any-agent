"""
Context Management Examples

This demonstrates manual history management patterns using the context utilities.

The SDK provides low-level helpers (estimate_tokens, truncate_messages) but does NOT
automatically manage context. You decide when and how to manage history based on your
domain-specific needs.

Patterns demonstrated:
1. Stateless agents (recommended for single-task agents)
2. Manual truncation at natural breakpoints
3. Token budget monitoring with periodic checks
4. External memory (RAG-lite pattern)
"""

import asyncio

from open_agent import AgentOptions, Client
from open_agent.context import estimate_tokens, truncate_messages


# ============================================================================
# Pattern 1: Stateless Agents (Recommended)
# ============================================================================
# Best for: Single-task agents (copy editor, code formatter, etc.)


async def pattern_1_stateless():
    """Process tasks independently with no history accumulation"""
    print("=== Pattern 1: Stateless Agents ===\n")

    options = AgentOptions(
        model="gpt-3.5-turbo",  # Change to your local model
        base_url="http://localhost:1234/v1",
        system_prompt="You are a helpful assistant",
    )

    # Process each task independently
    tasks = ["Explain Python", "Explain JavaScript", "Explain Rust"]

    for task in tasks:
        # Fresh client for each task - no history accumulation
        async with Client(options) as client:
            await client.query(task)

            response = ""
            async for msg in client.receive_messages():
                if hasattr(msg, "text"):
                    response += msg.text

            print(f"Task: {task}")
            print(f"Response length: {len(response)} chars")
            print(f"History size: {len(client.history)} messages\n")
        # Client disposed, fresh context for next task


# ============================================================================
# Pattern 2: Manual Truncation at Natural Breakpoints
# ============================================================================
# Best for: Multi-turn conversations with clear task boundaries


async def pattern_2_manual_truncation():
    """Manually truncate history after completing each major task"""
    print("=== Pattern 2: Manual Truncation ===\n")

    options = AgentOptions(
        model="gpt-3.5-turbo",
        base_url="http://localhost:1234/v1",
        system_prompt="You are a helpful coding assistant",
    )

    async with Client(options) as client:
        # First task - analyze code
        await client.query("Analyze this function: def add(a, b): return a + b")
        async for msg in client.receive_messages():
            pass  # Process messages

        print(f"After task 1: {len(client.history)} messages")

        # Second task - write tests
        await client.query("Write unit tests for the add function")
        async for msg in client.receive_messages():
            pass

        print(f"After task 2: {len(client.history)} messages")

        # Truncate after completing major milestone
        print("Truncating history (keeping last 3 messages)...")
        client.message_history = truncate_messages(client.history, keep=3)
        print(f"After truncation: {len(client.history)} messages\n")

        # Third task - continues with limited context
        await client.query("What was the function we discussed?")
        async for msg in client.receive_messages():
            if hasattr(msg, "text"):
                print(f"Response: {msg.text[:100]}...")


# ============================================================================
# Pattern 3: Token Budget Monitoring
# ============================================================================
# Best for: Long sessions that need to stay within model limits


async def pattern_3_token_monitoring():
    """Monitor token usage and truncate when approaching limit"""
    print("\n=== Pattern 3: Token Budget Monitoring ===\n")

    # Set conservative limit (leave safety margin)
    MODEL_MAX_TOKENS = 32000
    TRUNCATE_AT = int(MODEL_MAX_TOKENS * 0.85)  # Truncate at 85% capacity

    options = AgentOptions(
        model="gpt-3.5-turbo",
        base_url="http://localhost:1234/v1",
        system_prompt="You are a helpful assistant",
    )

    async with Client(options) as client:
        # Simulate long conversation
        for i in range(10):
            await client.query(f"Tell me about topic {i}")
            async for msg in client.receive_messages():
                pass  # Process messages

            # Check token budget periodically
            current_tokens = estimate_tokens(client.history)
            print(f"Turn {i+1}: ~{current_tokens} tokens")

            if current_tokens > TRUNCATE_AT:
                print(f"⚠️  Approaching limit ({TRUNCATE_AT} tokens), truncating...")
                old_size = len(client.history)
                client.message_history = truncate_messages(client.history, keep=5)
                new_tokens = estimate_tokens(client.history)
                print(
                    f"   Truncated {old_size} → {len(client.history)} messages "
                    f"(~{current_tokens} → ~{new_tokens} tokens)\n"
                )


# ============================================================================
# Pattern 4: External Memory (RAG-lite)
# ============================================================================
# Best for: Research agents, knowledge accumulation


async def pattern_4_external_memory():
    """Store important facts externally, keep conversation context small"""
    print("\n=== Pattern 4: External Memory (RAG-lite) ===\n")

    # Simple external "database" (in production, use SQLite, vector DB, etc.)
    knowledge_base = {}

    options = AgentOptions(
        model="gpt-3.5-turbo",
        base_url="http://localhost:1234/v1",
        system_prompt="You are a research assistant",
    )

    async with Client(options) as client:
        # Research phase - accumulate facts
        topics = ["Python features", "JavaScript async/await", "Rust ownership"]

        for topic in topics:
            await client.query(f"What are the key points about {topic}?")

            response_text = ""
            async for msg in client.receive_messages():
                if hasattr(msg, "text"):
                    response_text += msg.text

            # Save to external memory
            knowledge_base[topic] = response_text[:200]  # Save summary
            print(f"Stored: {topic} ({len(response_text)} chars)")

        # Clear history after research phase
        print(f"\nClearing history ({len(client.history)} messages)...")
        client.message_history = truncate_messages(client.history, keep=0)
        print(f"After clear: {len(client.history)} messages")

        # Analysis phase - query external memory
        print("\nKnowledge base contents:")
        for topic, summary in knowledge_base.items():
            print(f"  - {topic}: {summary[:50]}...")

        # Use stored knowledge for new task
        context = "\n".join(
            [f"- {topic}: {summary}" for topic, summary in knowledge_base.items()]
        )
        await client.query(f"Based on these facts:\n{context}\n\nCompare these languages")

        async for msg in client.receive_messages():
            if hasattr(msg, "text"):
                print(f"\nComparison: {msg.text[:150]}...")


# ============================================================================
# Utility: Display Context Stats
# ============================================================================


def display_context_stats(messages: list):
    """Display statistics about current context"""
    tokens = estimate_tokens(messages)
    system_count = sum(1 for m in messages if m.get("role") == "system")
    user_count = sum(1 for m in messages if m.get("role") == "user")
    assistant_count = sum(1 for m in messages if m.get("role") == "assistant")

    print(f"Context Statistics:")
    print(f"  Total messages: {len(messages)}")
    print(f"  System: {system_count}, User: {user_count}, Assistant: {assistant_count}")
    print(f"  Estimated tokens: ~{tokens}")
    print(f"  At 32k limit: {tokens / 32000 * 100:.1f}%")


# ============================================================================
# Main Demo Runner
# ============================================================================


async def main():
    """Run all pattern demonstrations"""
    print("Context Management Examples")
    print("=" * 60)
    print(
        "\nNOTE: These examples use 'gpt-3.5-turbo' as a placeholder."
        "\nChange to your local model endpoint before running."
        "\n"
    )

    # Uncomment to run specific patterns:

    # await pattern_1_stateless()
    # await pattern_2_manual_truncation()
    # await pattern_3_token_monitoring()
    # await pattern_4_external_memory()

    # Quick utility demo
    print("\n=== Utility Demo ===\n")
    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you today?"},
        {"role": "user", "content": "Tell me about Python"},
        {
            "role": "assistant",
            "content": "Python is a high-level programming language...",
        },
    ]

    print("Original messages:")
    display_context_stats(messages)

    print("\nAfter truncating to last 2 messages:")
    truncated = truncate_messages(messages, keep=2)
    display_context_stats(truncated)


if __name__ == "__main__":
    # Note: Update the base_url to point to your local LLM server
    # Examples:
    #   LM Studio: http://localhost:1234/v1
    #   Ollama: http://localhost:11434/v1
    #   llama.cpp: http://localhost:8080/v1

    asyncio.run(main())
