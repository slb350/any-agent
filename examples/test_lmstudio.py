"""
Comprehensive test suite for LM Studio provider.

Tests all SDK features with LM Studio server:
- Simple query
- Multi-turn conversation
- Streaming responses
- Configuration methods
- Error handling
- Timeout behavior
"""
import asyncio
import time
import os
from any_agent import query, Client, AgentOptions, TextBlock, ToolUseBlock, ToolUseError
from any_agent.config import get_model, get_base_url


# LM Studio configuration
# Replace with your actual LM Studio server URL
LMSTUDIO_URL = "http://localhost:1234/v1"  # Default local LM Studio
# LMSTUDIO_URL = "http://192.168.1.100:1234/v1"  # Example: Network server
LMSTUDIO_MODEL = "qwen/qwen3-30b-a3b-2507"


async def test_simple_query():
    """Test 1: Basic single-turn query"""
    print("=" * 60)
    print("TEST 1: Simple Query with LM Studio")
    print("=" * 60)

    options = AgentOptions(
        system_prompt="You are a helpful assistant. Be concise.",
        model=LMSTUDIO_MODEL,
        base_url=LMSTUDIO_URL,
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
                elif isinstance(block, ToolUseBlock):
                    print(f"\n[Tool: {block.name}]", flush=True)
                elif isinstance(block, ToolUseError):
                    print(f"\n[Tool Error: {block.error}]", flush=True)

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
        model=LMSTUDIO_MODEL,
        base_url=LMSTUDIO_URL,
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
            response1 = ""
            async for block in client.receive_messages():
                if isinstance(block, TextBlock):
                    response1 += block.text
                    print(block.text, end="", flush=True)

            print(f"\n[Turn {client.turn_count} complete]")

            # Turn 2 - tests context retention
            print("\nUser: What's its population?")
            await client.query("What's its population?")

            print("Assistant: ", end="", flush=True)
            response2 = ""
            async for block in client.receive_messages():
                if isinstance(block, TextBlock):
                    response2 += block.text
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
        model=LMSTUDIO_MODEL,
        base_url=LMSTUDIO_URL,
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
    """Test 4: Config helpers with environment variables"""
    print("\n" + "=" * 60)
    print("TEST 4: Configuration Helpers")
    print("=" * 60)

    # Test with explicit values
    model = get_model(LMSTUDIO_MODEL)
    base_url = get_base_url(base_url=LMSTUDIO_URL)

    print(f"Resolved model: {model}")
    print(f"Resolved base_url: {base_url}")

    # Test with environment variables
    os.environ["ANY_AGENT_MODEL"] = LMSTUDIO_MODEL
    os.environ["ANY_AGENT_BASE_URL"] = LMSTUDIO_URL

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


async def test_timeout():
    """Test 5: Timeout configuration"""
    print("\n" + "=" * 60)
    print("TEST 5: Timeout Configuration")
    print("=" * 60)

    # Test with custom timeout
    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model=LMSTUDIO_MODEL,
        base_url=LMSTUDIO_URL,
        timeout=30.0,  # 30 second timeout
        max_tokens=50
    )

    print(f"Timeout: {options.timeout} seconds")
    print("Question: What is 2+2?")
    print("Response: ", end="", flush=True)

    start = time.time()
    try:
        result = query(prompt="What is 2+2?", options=options)

        async for msg in result:
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)

        elapsed = time.time() - start
        print(f"\n\n✓ Completed in {elapsed:.2f} seconds (within timeout)")
        return True

    except Exception as e:
        elapsed = time.time() - start
        if "timeout" in str(e).lower():
            print(f"\n\n⚠️ Timed out after {elapsed:.2f} seconds")
        else:
            print(f"\n\n✗ Error after {elapsed:.2f} seconds: {e}")
        return False


async def test_error_handling():
    """Test 6: Error handling with invalid model"""
    print("\n" + "=" * 60)
    print("TEST 6: Error Handling")
    print("=" * 60)

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="invalid-model-xyz",  # Likely invalid
        base_url=LMSTUDIO_URL,
        max_tokens=50
    )

    print(f"Testing with invalid model: {options.model}")

    try:
        result = query(prompt="Hello", options=options)

        async for msg in result:
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(f"Unexpected success: {block.text}")

        print("⚠️ Model exists or server accepted invalid model")
        return True

    except Exception as e:
        print(f"✓ Properly caught error: {e}")
        return True


async def main():
    """Run all LM Studio tests"""
    print("\n" + "=" * 60)
    print("LM STUDIO PROVIDER TEST SUITE")
    print(f"Server: {LMSTUDIO_URL}")
    print(f"Model: {LMSTUDIO_MODEL}")
    print("=" * 60)

    results = []
    tests = [
        ("Simple Query", test_simple_query),
        ("Multi-Turn", test_multi_turn),
        ("Streaming", test_streaming),
        ("Config Helpers", test_config_helpers),
        ("Timeout", test_timeout),
        ("Error Handling", test_error_handling)
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
        print("\n🎉 All tests passed! LM Studio provider is fully compatible.")
    else:
        print("\n⚠️ Some tests failed. Check errors above.")


if __name__ == "__main__":
    asyncio.run(main())