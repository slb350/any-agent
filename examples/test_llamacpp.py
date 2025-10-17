"""
Test suite for llama.cpp server provider.

Tests SDK features with llama.cpp's OpenAI-compatible API:
- Simple query
- Multi-turn conversation
- Streaming responses
- Configuration methods
"""
import asyncio
import os
from any_agent import query, Client, AgentOptions, TextBlock, ToolUseBlock, ToolUseError
from any_agent.config import get_model, get_base_url


# llama.cpp configuration
LLAMACPP_URL = "http://localhost:8080/v1"
LLAMACPP_MODEL = "tinyllama-1.1b.gguf"


async def test_simple_query():
    """Test 1: Basic single-turn query"""
    print("=" * 60)
    print("TEST 1: Simple Query with llama.cpp")
    print("=" * 60)

    options = AgentOptions(
        system_prompt="You are a helpful assistant. Be concise.",
        model=LLAMACPP_MODEL,
        base_url=LLAMACPP_URL,
        max_tokens=100,
        temperature=0.7
    )

    print(f"\nServer: {options.base_url}")
    print(f"Model: {options.model}")
    print("\nQuestion: What is the capital of Japan?")
    print("\nResponse: ", end="", flush=True)

    try:
        result = query(prompt="What is the capital of Japan?", options=options)

        response_text = ""
        async for msg in result:
            for block in msg.content:
                if isinstance(block, TextBlock):
                    response_text += block.text
                    print(block.text, end="", flush=True)

        print(f"\n\n✓ Success! Received {len(response_text)} characters")
        return True

    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_multi_turn():
    """Test 2: Multi-turn conversation with context"""
    print("\n" + "=" * 60)
    print("TEST 2: Multi-Turn Conversation")
    print("=" * 60)

    options = AgentOptions(
        system_prompt="You are a helpful assistant. Be concise.",
        model=LLAMACPP_MODEL,
        base_url=LLAMACPP_URL,
        max_turns=3,
        max_tokens=100,
        temperature=0.7
    )

    print(f"\nServer: {options.base_url}")
    print(f"Model: {options.model}")
    print(f"Max turns: {options.max_turns}\n")

    try:
        async with Client(options) as client:
            # Turn 1
            print("User: What's the capital of Japan?")
            await client.query("What's the capital of Japan?")

            print("Assistant: ", end="", flush=True)
            async for block in client.receive_messages():
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)

            print(f"\n[Turn {client.turn_count} complete]")

            # Turn 2 - tests context retention
            print("\nUser: What's its population?")
            await client.query("What's its population?")

            print("Assistant: ", end="", flush=True)
            async for block in client.receive_messages():
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)

            print(f"\n[Turn {client.turn_count} complete]")

            # Turn 3 - more context
            print("\nUser: Is it bigger than New York City?")
            await client.query("Is it bigger than New York City?")

            print("Assistant: ", end="", flush=True)
            async for block in client.receive_messages():
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)

            print(f"\n[Turn {client.turn_count} complete]")

            print(f"\n✓ Multi-turn conversation successful!")
            print(f"  Total turns: {client.turn_count}")
            print(f"  History length: {len(client.history)} messages")

            return True

    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_streaming():
    """Test 3: Verify streaming works properly"""
    print("\n" + "=" * 60)
    print("TEST 3: Streaming Response")
    print("=" * 60)

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model=LLAMACPP_MODEL,
        base_url=LLAMACPP_URL,
        max_tokens=150,
        temperature=0.7
    )

    print("\nQuestion: Count from 1 to 10 slowly")
    print("Response: ", end="", flush=True)

    try:
        result = query(
            prompt="Count from 1 to 10, putting each number on its own line",
            options=options
        )

        chunks_received = 0
        async for msg in result:
            for block in msg.content:
                if isinstance(block, TextBlock):
                    chunks_received += 1
                    print(block.text, end="", flush=True)

        print(f"\n\n✓ Streaming works! Received {chunks_received} chunks")
        return True

    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        return False


async def test_config_helpers():
    """Test 4: Config helpers with provider shortcut"""
    print("\n" + "=" * 60)
    print("TEST 4: Configuration Helpers")
    print("=" * 60)

    # Test provider shortcut
    base_url = get_base_url(provider="llamacpp")
    print(f"Provider shortcut 'llamacpp' resolves to: {base_url}")

    # Test with environment variables
    os.environ["ANY_AGENT_MODEL"] = LLAMACPP_MODEL
    os.environ["ANY_AGENT_BASE_URL"] = LLAMACPP_URL

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model=get_model(),
        base_url=get_base_url(),
        max_tokens=50
    )

    print(f"\nFrom env vars:")
    print(f"  Model: {options.model}")
    print(f"  Base URL: {options.base_url}")

    # Clean up
    del os.environ["ANY_AGENT_MODEL"]
    del os.environ["ANY_AGENT_BASE_URL"]

    print("\nTesting quick query...")
    print("Question: Say hello")
    print("Response: ", end="", flush=True)

    try:
        result = query(prompt="Say hello", options=options)

        async for msg in result:
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)

        print("\n\n✓ Config helpers work correctly!")
        return True

    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        return False


async def main():
    """Run all llama.cpp tests"""
    print("\n" + "=" * 60)
    print("LLAMA.CPP PROVIDER TEST SUITE")
    print(f"Server: {LLAMACPP_URL}")
    print(f"Model: {LLAMACPP_MODEL} (TinyLlama 1.1B)")
    print("=" * 60)

    results = []
    tests = [
        ("Simple Query", test_simple_query),
        ("Multi-Turn", test_multi_turn),
        ("Streaming", test_streaming),
        ("Config Helpers", test_config_helpers),
    ]

    for name, test_func in tests:
        print(f"\nRunning: {name}")
        result = await test_func()
        results.append((name, result))
        await asyncio.sleep(1)  # Brief pause between tests

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20} {status}")

    passed_count = sum(1 for _, passed in results if passed)
    print(f"\nOverall: {passed_count}/{len(results)} tests passed")

    if passed_count == len(results):
        print("\n🎉 All tests passed! llama.cpp provider is fully compatible.")
    else:
        print("\n⚠️ Some tests failed. Check errors above.")


if __name__ == "__main__":
    asyncio.run(main())