"""
Simple example using environment variable for configuration.

To use your network LM Studio server:
    export ANY_AGENT_BASE_URL="https://lmstudio.localbrandonfamily.com/v1"
    python examples/simple_with_env.py

Or use provider shorthand for localhost servers:
    python examples/simple_with_env.py  # Defaults to LM Studio
"""
import asyncio
from any_agent import query, AgentOptions, TextBlock


async def main():
    # No base_url needed - uses environment variable or defaults to localhost
    options = AgentOptions(
        system_prompt="You are a helpful assistant. Be concise.",
        model="qwen/qwen3-30b-a3b-2507",  # Your model name
        max_tokens=100
    )

    print(f"Using server: {options.base_url}")
    print("Question: What is the capital of France?\n")
    print("Response: ", end="", flush=True)

    result = query(prompt="What is the capital of France?", options=options)

    async for msg in result:
        for block in msg.content:
            if isinstance(block, TextBlock):
                print(block.text, end="", flush=True)

    print("\n\n✓ Complete!")


if __name__ == "__main__":
    asyncio.run(main())
