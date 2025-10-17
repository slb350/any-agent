"""Test timeout parameter with Ollama"""
import asyncio
import time
from any_agent import query, AgentOptions, TextBlock
from any_agent.config import get_base_url


async def test_default_timeout():
    """Test with default 60 second timeout"""
    print("=== Test 1: Default Timeout (60 seconds) ===")

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="kimi-k2:1t-cloud",
        base_url=get_base_url(provider="ollama"),
        max_tokens=20  # Keep response short for quick test
    )

    print(f"Default timeout: {options.timeout} seconds")
    print("Asking a simple question...")

    start = time.time()
    try:
        result = query(prompt="What is 2+2?", options=options)
        async for msg in result:
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(f"Response: {block.text}")

        elapsed = time.time() - start
        print(f"✓ Completed in {elapsed:.2f} seconds")

    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ Failed after {elapsed:.2f} seconds: {e}")


async def test_custom_timeout():
    """Test with very short timeout (should fail)"""
    print("\n=== Test 2: Short Timeout (0.1 seconds) ===")

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="kimi-k2:1t-cloud",
        base_url=get_base_url(provider="ollama"),
        timeout=0.1,  # Very short, should timeout
        max_tokens=20
    )

    print(f"Custom timeout: {options.timeout} seconds")
    print("Asking a question (expecting timeout)...")

    start = time.time()
    try:
        result = query(prompt="What is 2+2?", options=options)
        async for msg in result:
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(f"Response: {block.text}")

        elapsed = time.time() - start
        print(f"✗ Unexpectedly succeeded in {elapsed:.2f} seconds")

    except Exception as e:
        elapsed = time.time() - start
        if "timeout" in str(e).lower() or "timed out" in str(e).lower():
            print(f"✓ Correctly timed out after {elapsed:.2f} seconds")
            print(f"  Error message: {e}")
        else:
            print(f"✗ Different error after {elapsed:.2f} seconds: {e}")


async def test_longer_timeout():
    """Test with longer timeout for slow models"""
    print("\n=== Test 3: Longer Timeout (120 seconds) ===")

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="kimi-k2:1t-cloud",
        base_url=get_base_url(provider="ollama"),
        timeout=120.0,  # Longer timeout for slow models
        max_tokens=20
    )

    print(f"Long timeout: {options.timeout} seconds")
    print("Asking a simple question...")

    start = time.time()
    try:
        result = query(prompt="What is the capital of France?", options=options)
        response = ""
        async for msg in result:
            for block in msg.content:
                if isinstance(block, TextBlock):
                    response += block.text

        elapsed = time.time() - start
        print(f"Response: {response}")
        print(f"✓ Completed in {elapsed:.2f} seconds")

    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ Failed after {elapsed:.2f} seconds: {e}")


async def main():
    print("TIMEOUT PARAMETER TEST")
    print("=" * 40)

    # Test 1: Default timeout
    await test_default_timeout()

    # Test 2: Very short timeout (should fail)
    await test_custom_timeout()

    # Test 3: Longer timeout
    await test_longer_timeout()

    print("\n" + "=" * 40)
    print("All timeout tests completed!")


if __name__ == "__main__":
    asyncio.run(main())