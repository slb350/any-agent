"""Multi-turn chat example with Ollama

Prerequisites:
    - Ollama running on localhost:11434
    - Model available (e.g., kimi-k2:1t-cloud, deepseek-v3.1:671b-cloud)
    - Check available models: ollama list
"""
import asyncio
from open_agent import Client, AgentOptions, TextBlock, ToolUseBlock, ToolUseError
from open_agent.config import get_base_url


async def main():
    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="kimi-k2:1t-cloud",  # Change to your available model
        base_url=get_base_url(provider="ollama"),
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
