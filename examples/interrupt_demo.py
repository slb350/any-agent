"""
Interrupt Capability Demo

This example demonstrates how to use the interrupt() capability to cancel
long-running operations:

1. Timeout-based interruption
2. User-triggered interruption (simulated)
3. Interrupt with automatic tool execution

Note: This example uses LM Studio. Make sure LM Studio is running at
http://localhost:1234 with a model loaded before running.
"""
import asyncio
from open_agent import Client, AgentOptions, tool


# Example 1: Timeout-based Interruption
async def timeout_example():
    """
    Demonstrate interrupting a long-running query with a timeout.
    """
    print("=" * 60)
    print("Example 1: Timeout-based Interruption")
    print("=" * 60)

    options = AgentOptions(
        system_prompt="You are a helpful assistant. Be verbose in your responses.",
        model="qwen2.5-7b-instruct",  # or whatever model you have loaded
        base_url="http://localhost:1234/v1",
        temperature=0.7
    )

    async with Client(options) as client:
        await client.query("Write a detailed 1000-word essay about quantum computing")

        print("Starting to receive response (will timeout after 3 seconds)...\n")

        async def collect_messages():
            response = []
            async for block in client.receive_messages():
                print(block.text, end="", flush=True)
                response.append(block)
            return response

        try:
            # Set a timeout - will interrupt after 3 seconds
            result = await asyncio.wait_for(collect_messages(), timeout=3.0)
            print("\n\nResponse completed within timeout")
        except asyncio.TimeoutError:
            # Interrupt on timeout
            await client.interrupt()
            print("\n\n⚠️  Operation timed out and was interrupted!")
            print(f"Partial history preserved: {len(client.history)} messages\n")


# Example 2: Conditional Interruption
async def conditional_example():
    """
    Demonstrate interrupting based on response content.
    Interrupt if the response contains certain keywords.
    """
    print("=" * 60)
    print("Example 2: Conditional Interruption")
    print("=" * 60)

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="qwen2.5-7b-instruct",
        base_url="http://localhost:1234/v1",
        temperature=0.7
    )

    async with Client(options) as client:
        await client.query("Tell me about machine learning")

        print("Receiving response (will stop if we see 'neural network')...\n")

        full_text = ""
        async for block in client.receive_messages():
            print(block.text, end="", flush=True)
            full_text += block.text

            # Interrupt if we see a specific keyword
            if "neural network" in full_text.lower():
                await client.interrupt()
                print("\n\n⚠️  Found keyword 'neural network' - interrupting!")
                break

        print(f"\nReceived {len(full_text)} characters before interrupt\n")


# Example 3: Interrupt with Auto-Execution
async def auto_execution_example():
    """
    Demonstrate interrupting during automatic tool execution.
    """
    print("=" * 60)
    print("Example 3: Interrupt During Auto-Execution")
    print("=" * 60)

    # Define a slow tool
    @tool("slow_calculation", "Performs a slow calculation", {
        "type": "object",
        "properties": {
            "operation": {"type": "string"}
        }
    })
    async def slow_calculation(args):
        operation = args.get("operation", "unknown")
        print(f"  🔧 Executing slow tool: {operation}")
        await asyncio.sleep(2)  # Simulate slow operation
        return {"result": f"Completed {operation}"}

    @tool("fast_calculation", "Performs a fast calculation", {
        "type": "object",
        "properties": {
            "operation": {"type": "string"}
        }
    })
    async def fast_calculation(args):
        operation = args.get("operation", "unknown")
        print(f"  🔧 Executing fast tool: {operation}")
        return {"result": f"Completed {operation}"}

    options = AgentOptions(
        system_prompt="You are a calculator assistant. Use tools to perform calculations.",
        model="qwen2.5-7b-instruct",
        base_url="http://localhost:1234/v1",
        tools=[slow_calculation, fast_calculation],
        auto_execute_tools=True,
        max_tool_iterations=10,
        temperature=0.7
    )

    async with Client(options) as client:
        await client.query("Perform slow_calculation with operation 'compute', then fast_calculation with operation 'add', then another slow_calculation")

        print("Starting auto-execution (will interrupt after first tool)...\n")

        tool_count = 0
        async for block in client.receive_messages():
            if hasattr(block, 'name'):  # ToolUseBlock
                tool_count += 1
                print(f"\n📦 Tool call #{tool_count}: {block.name}")

                if tool_count == 1:
                    # Interrupt after first tool
                    await asyncio.sleep(0.1)  # Let it start
                    await client.interrupt()
                    print("\n⚠️  Interrupted after first tool!\n")
                    break
            elif hasattr(block, 'text'):  # TextBlock
                print(f"💬 Assistant: {block.text}")

        print(f"Total tools seen before interrupt: {tool_count}\n")


# Example 4: Concurrent Interruption
async def concurrent_example():
    """
    Demonstrate interrupting from a separate asyncio task.
    This simulates a user clicking a "Cancel" button.
    """
    print("=" * 60)
    print("Example 4: Concurrent Interruption (Simulated User Cancel)")
    print("=" * 60)

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="qwen2.5-7b-instruct",
        base_url="http://localhost:1234/v1",
        temperature=0.7
    )

    client = Client(options)

    async def stream_task():
        """Simulate the main streaming task"""
        await client.query("Explain artificial intelligence in detail")
        print("Receiving response...\n")

        full_text = ""
        async for block in client.receive_messages():
            print(block.text, end="", flush=True)
            full_text += block.text
            await asyncio.sleep(0.1)  # Simulate processing

        return full_text

    async def cancel_button_task():
        """Simulate a user clicking cancel after 2 seconds"""
        await asyncio.sleep(2.0)
        print("\n\n🛑 User clicked cancel button!")
        await client.interrupt()

    # Run both tasks concurrently
    try:
        result, _ = await asyncio.gather(
            stream_task(),
            cancel_button_task(),
            return_exceptions=True
        )

        if isinstance(result, str):
            print(f"\n\nReceived {len(result)} characters before interrupt")
    finally:
        await client.close()

    print()


# Example 5: Interrupt and Retry
async def retry_example():
    """
    Demonstrate interrupting and then retrying with a different query.
    """
    print("=" * 60)
    print("Example 5: Interrupt and Retry")
    print("=" * 60)

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="qwen2.5-7b-instruct",
        base_url="http://localhost:1234/v1",
        temperature=0.7
    )

    async with Client(options) as client:
        # First query - will be interrupted
        print("First query (will be interrupted)...")
        await client.query("Tell me everything about the history of computing")

        count = 0
        async for block in client.receive_messages():
            count += 1
            if count == 3:
                print("\n⚠️  Oops, that was too broad. Interrupting...\n")
                await client.interrupt()
                break

        # Retry with more specific query
        print("Retrying with more specific query...\n")
        await client.query("Tell me about Alan Turing in 2 sentences")

        async for block in client.receive_messages():
            print(block.text, end="", flush=True)

        print("\n\nSuccess! Query completed after retry.\n")


async def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("INTERRUPT CAPABILITY DEMO")
    print("=" * 60 + "\n")

    print("This demo requires LM Studio running at http://localhost:1234")
    print("with a model loaded (e.g., qwen2.5-7b-instruct)\n")

    # Check if user wants to run all examples
    print("Running examples...")
    print()

    try:
        await timeout_example()
        await asyncio.sleep(1)

        await conditional_example()
        await asyncio.sleep(1)

        await auto_execution_example()
        await asyncio.sleep(1)

        await concurrent_example()
        await asyncio.sleep(1)

        await retry_example()

        print("=" * 60)
        print("All examples completed!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("\nMake sure LM Studio is running at http://localhost:1234")


if __name__ == "__main__":
    asyncio.run(main())
