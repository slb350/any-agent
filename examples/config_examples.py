"""
Examples of different configuration methods for Any-Agent SDK.

The SDK supports multiple ways to configure the base_url:
1. Explicit base_url parameter
2. Environment variable ANY_AGENT_BASE_URL
3. Provider shorthand (lmstudio, ollama, llamacpp, vllm)
4. Default to LM Studio (http://localhost:1234/v1)
"""
import asyncio
import os
from any_agent import query, AgentOptions, TextBlock


async def example_explicit_url():
    """Example 1: Explicit base_url (highest priority)"""
    print("=== Example 1: Explicit URL ===")

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="qwen/qwen3-30b-a3b-2507",
        base_url="https://lmstudio.localbrandonfamily.com/v1",
        max_tokens=50
    )

    print(f"Using: {options.base_url}")
    # Result: Uses https://lmstudio.localbrandonfamily.com/v1


async def example_env_var():
    """Example 2: Environment variable"""
    print("\n=== Example 2: Environment Variable ===")

    # Set environment variable
    os.environ["ANY_AGENT_BASE_URL"] = "https://lmstudio.localbrandonfamily.com/v1"

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="qwen/qwen3-30b-a3b-2507",
        max_tokens=50
    )

    print(f"Using: {options.base_url}")
    # Result: Uses https://lmstudio.localbrandonfamily.com/v1 from env var

    # Clean up
    del os.environ["ANY_AGENT_BASE_URL"]


async def example_provider_ollama():
    """Example 3: Provider shorthand"""
    print("\n=== Example 3: Provider Shorthand (Ollama) ===")

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="llama3.1:70b",
        provider="ollama",  # Automatically uses http://localhost:11434/v1
        max_tokens=50
    )

    print(f"Using: {options.base_url}")
    # Result: Uses http://localhost:11434/v1


async def example_provider_lmstudio():
    """Example 4: Provider shorthand (LM Studio)"""
    print("\n=== Example 4: Provider Shorthand (LM Studio) ===")

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="qwen2.5-32b-instruct",
        provider="lmstudio",  # Automatically uses http://localhost:1234/v1
        max_tokens=50
    )

    print(f"Using: {options.base_url}")
    # Result: Uses http://localhost:1234/v1


async def example_default():
    """Example 5: Default (no configuration)"""
    print("\n=== Example 5: Default (LM Studio on localhost) ===")

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="qwen2.5-32b-instruct",
        max_tokens=50
    )

    print(f"Using: {options.base_url}")
    # Result: Defaults to http://localhost:1234/v1


async def example_priority():
    """Example 6: Priority order demonstration"""
    print("\n=== Example 6: Priority Order ===")

    # Set env var
    os.environ["ANY_AGENT_BASE_URL"] = "http://env-server:1234/v1"

    # Explicit URL overrides everything
    options1 = AgentOptions(
        system_prompt="Test",
        model="test",
        base_url="http://explicit:8080/v1",
        provider="ollama"
    )
    print(f"Explicit + Env + Provider → {options1.base_url}")

    # Env overrides provider
    options2 = AgentOptions(
        system_prompt="Test",
        model="test",
        provider="ollama"
    )
    print(f"Env + Provider → {options2.base_url}")

    # Clean up
    del os.environ["ANY_AGENT_BASE_URL"]

    # Provider overrides default
    options3 = AgentOptions(
        system_prompt="Test",
        model="test",
        provider="ollama"
    )
    print(f"Provider only → {options3.base_url}")

    # Default
    options4 = AgentOptions(
        system_prompt="Test",
        model="test"
    )
    print(f"Default → {options4.base_url}")


async def main():
    """Run all examples"""
    await example_explicit_url()
    await example_env_var()
    await example_provider_ollama()
    await example_provider_lmstudio()
    await example_default()
    await example_priority()

    print("\n=== All Providers ===")
    print("lmstudio  → http://localhost:1234/v1")
    print("ollama    → http://localhost:11434/v1")
    print("llamacpp  → http://localhost:8080/v1")
    print("vllm      → http://localhost:8000/v1")


if __name__ == "__main__":
    asyncio.run(main())
