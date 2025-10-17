"""
Complete configuration using environment variables with config helpers.

This example shows how to use config helpers to pull from environment variables,
making it easy to switch between development and production.

Usage:
    export ANY_AGENT_BASE_URL="https://lmstudio.localbrandonfamily.com/v1"
    export ANY_AGENT_MODEL="qwen/qwen3-30b-a3b-2507"
    python examples/env_config_complete.py

Without env vars, get_model() returns None and raises a helpful error.
"""
import asyncio
from any_agent import query, AgentOptions, TextBlock
from any_agent.config import get_base_url, get_model


async def main():
    # Use config helpers - they check env vars and provide defaults
    model = get_model()  # Checks ANY_AGENT_MODEL env var
    if not model:
        raise ValueError(
            "Model not configured. Set ANY_AGENT_MODEL environment variable "
            "or pass model explicitly"
        )

    options = AgentOptions(
        system_prompt="You are a helpful assistant. Be concise.",
        model=model,
        base_url=get_base_url()  # Checks ANY_AGENT_BASE_URL env var, defaults to LM Studio
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
