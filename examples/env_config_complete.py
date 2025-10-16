"""
Complete configuration using environment variables only.

This example shows how to configure EVERYTHING via environment variables,
making it easy to switch between development and production without
changing any code.

Usage:
    export ANY_AGENT_BASE_URL="https://lmstudio.localbrandonfamily.com/v1"
    export ANY_AGENT_MODEL="qwen/qwen3-30b-a3b-2507"
    python examples/env_config_complete.py
"""
import asyncio
from any_agent import query, AgentOptions, TextBlock


async def main():
    # No hardcoded configuration - everything from environment!
    options = AgentOptions(
        system_prompt="You are a helpful assistant. Be concise.",
        # model and base_url are automatically resolved from env vars
    )

    print(f"Server: {options.base_url}")
    print(f"Model: {options.model}")
    print("\nQuestion: What is Python?\n")
    print("Response: ", end="", flush=True)

    result = query(prompt="What is Python? Answer in one sentence.", options=options)

    async for msg in result:
        for block in msg.content:
            if isinstance(block, TextBlock):
                print(block.text, end="", flush=True)

    print("\n\n✓ Complete - all config from environment variables!")


if __name__ == "__main__":
    # This will fail with a clear error if ANY_AGENT_MODEL is not set
    # Try running without the env var to see the error message
    asyncio.run(main())
