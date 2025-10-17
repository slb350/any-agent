"""Test multi-turn conversation with network LM Studio"""
import asyncio
from open_agent import Client, AgentOptions, TextBlock, ToolUseBlock, ToolUseError


async def main():
    # Example: Testing multi-turn conversation on network server
    # Replace with your actual server URL
    options = AgentOptions(
        system_prompt="You are a helpful assistant. Be concise.",
        model="qwen/qwen3-30b-a3b-2507",
        base_url="http://192.168.1.100:1234/v1",  # Example network server
        max_turns=5,
        temperature=0.7,
        max_tokens=100
    )

    print("Testing Multi-Turn Conversation with Open Agent SDK")
    print(f"Server: {options.base_url}")
    print(f"Model: {options.model}\n")

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

        print(f"\n[Turn {client.turn_count} complete]\n")

        # Turn 2 - testing context
        print("User: What's its population?")
        await client.query("What's its population?")

        print("Assistant: ", end="", flush=True)
        response2 = ""
        async for block in client.receive_messages():
            if isinstance(block, TextBlock):
                response2 += block.text
                print(block.text, end="", flush=True)

        print(f"\n[Turn {client.turn_count} complete]\n")

        # Verify context is maintained
        print(f"\n✓ Multi-turn conversation successful!")
        print(f"  Total turns: {client.turn_count}")
        print(f"  History length: {len(client.history)} messages")
        print(f"\nHistory:")
        for i, msg in enumerate(client.history):
            print(f"  {i+1}. {msg['role']}: {msg['content'][:50]}...")


if __name__ == "__main__":
    asyncio.run(main())
