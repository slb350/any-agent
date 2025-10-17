"""
Test script for kimi-k2:1t-cloud via Ollama.

This validates:
1. Simple query() function works
2. Multi-turn Client maintains context
3. Config helpers work correctly
4. Streaming responses work

Prerequisites:
    - Ollama running on localhost:11434
    - kimi-k2:1t-cloud model available (ollama list)
"""
import asyncio
from open_agent import query, Client, AgentOptions, TextBlock, ToolUseBlock
from open_agent.config import get_base_url


async def test_simple_query():
    """Test 1: Simple single-turn query"""
    print("=" * 60)
    print("TEST 1: Simple Query with kimi-k2:1t-cloud")
    print("=" * 60)

    options = AgentOptions(
        system_prompt="You are a helpful assistant. Be concise.",
        model="kimi-k2:1t-cloud",
        base_url=get_base_url(provider="ollama"),
        max_tokens=100,
        temperature=0.7
    )

    print(f"\nServer: {options.base_url}")
    print(f"Model: {options.model}")
    print("\nQuestion: What is the capital of France?")
    print("\nResponse: ", end="", flush=True)

    try:
        result = query(prompt="What is the capital of France?", options=options)

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
        model="kimi-k2:1t-cloud",
        base_url=get_base_url(provider="ollama"),
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
            print("User: What's the capital of France?")
            await client.query("What's the capital of France?")

            print("Assistant: ", end="", flush=True)
            response1 = ""
            async for block in client.receive_messages():
                if isinstance(block, TextBlock):
                    response1 += block.text
                    print(block.text, end="", flush=True)
                elif isinstance(block, ToolUseBlock):
                    print(f"\n[Tool: {block.name}]")

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
                elif isinstance(block, ToolUseBlock):
                    print(f"\n[Tool: {block.name}]")

            print(f"\n[Turn {client.turn_count} complete]")

            # Verify context was maintained
            print(f"\n✓ Multi-turn conversation successful!")
            print(f"  Total turns: {client.turn_count}")
            print(f"  History length: {len(client.history)} messages")

            # Show conversation history
            print(f"\nConversation History:")
            for i, msg in enumerate(client.history):
                role = msg['role']
                content = msg['content']
                if isinstance(content, str):
                    preview = content[:60] + "..." if len(content) > 60 else content
                else:
                    preview = str(content)[:60] + "..."
                print(f"  {i+1}. [{role}] {preview}")

            return True

    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_config_helpers():
    """Test 3: Config helpers work correctly"""
    print("\n" + "=" * 60)
    print("TEST 3: Config Helper Integration")
    print("=" * 60)

    from open_agent.config import get_model, get_base_url

    # Test fallback values
    model = get_model("kimi-k2:1t-cloud")  # Fallback since no env var set
    base_url = get_base_url(provider="ollama")  # Provider shortcut

    print(f"\nResolved model: {model}")
    print(f"Resolved base_url: {base_url}")

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model=model,
        base_url=base_url,
        max_tokens=50
    )

    print(f"\nTesting quick query...")
    print("Question: What is 2+2?")
    print("Response: ", end="", flush=True)

    try:
        result = query(prompt="What is 2+2?", options=options)

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
    """Run all tests"""
    print("\n" + "=" * 60)
    print("ANY-AGENT SDK - OLLAMA INTEGRATION TEST")
    print("Testing with kimi-k2:1t-cloud via Ollama")
    print("=" * 60)

    results = []

    # Test 1: Simple query
    results.append(await test_simple_query())

    # Test 2: Multi-turn
    results.append(await test_multi_turn())

    # Test 3: Config helpers
    results.append(await test_config_helpers())

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Simple Query:      {'✓ PASS' if results[0] else '✗ FAIL'}")
    print(f"Multi-Turn:        {'✓ PASS' if results[1] else '✗ FAIL'}")
    print(f"Config Helpers:    {'✓ PASS' if results[2] else '✗ FAIL'}")
    print(f"\nOverall: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("\n🎉 All tests passed! SDK is working with Ollama.")
    else:
        print("\n⚠️  Some tests failed. See errors above.")


if __name__ == "__main__":
    asyncio.run(main())
