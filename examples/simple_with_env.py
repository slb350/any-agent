"""
Simple example using config helpers for environment variable support.

To use your network LM Studio server:
    export ANY_AGENT_BASE_URL="https://lmstudio.localbrandonfamily.com/v1"
    export ANY_AGENT_MODEL="qwen/qwen3-30b-a3b-2507"
    python examples/simple_with_env.py

Or use provider shorthand:
    python examples/simple_with_env.py  # Uses ollama default
"""
import asyncio
from any_agent import query, AgentOptions, TextBlock
from any_agent.config import get_base_url, get_model


async def main():
    # Use config helpers to resolve from env vars with fallbacks
    options = AgentOptions(
        system_prompt="You are a helpful assistant. Be concise.",
        model=get_model("qwen/qwen3-30b-a3b-2507"),  # Uses ANY_AGENT_MODEL env or fallback
        base_url=get_base_url(provider="lmstudio"),  # Uses ANY_AGENT_BASE_URL env or provider default
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
