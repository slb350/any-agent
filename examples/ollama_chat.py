"""Multi-turn chat example with Ollama"""
import asyncio
from any_agent import Client, AgentOptions, TextBlock, ToolUseBlock, ToolUseError


async def main():
    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="llama3.1:70b",
        base_url="http://localhost:11434/v1",
        max_turns=5,
        temperature=0.7
    )

    async with Client(options) as client:
        # Turn 1
        print("User: What's the capital of France?")
        await client.query("What's the capital of France?")

        print("Assistant: ", end="", flush=True)
        async for block in client.receive_messages():
            if isinstance(block, TextBlock):
                print(block.text, end="", flush=True)
            elif isinstance(block, ToolUseBlock):
                print(f"\n[Tool: {block.name}]")
                print(f"  Input: {block.input}")
            elif isinstance(block, ToolUseError):
                print(f"\n[Tool Error: {block.error}]")

        print("\n")

        # Turn 2
        print("User: What's its population?")
        await client.query("What's its population?")

        print("Assistant: ", end="", flush=True)
        async for block in client.receive_messages():
            if isinstance(block, TextBlock):
                print(block.text, end="", flush=True)
            elif isinstance(block, ToolUseBlock):
                print(f"\n[Tool: {block.name}]")
                print(f"  Input: {block.input}")

        print(f"\n\nTotal turns: {client.turn_count}")
        print(f"Max turns: {client.turn_metadata['max_turns']}")


if __name__ == "__main__":
    asyncio.run(main())
