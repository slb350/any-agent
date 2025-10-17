"""
Examples of different configuration methods for Open Agent SDK.

The SDK uses config helpers to resolve model and base_url:
1. Explicit parameters (highest priority)
2. Environment variables ANY_AGENT_MODEL and ANY_AGENT_BASE_URL
3. Fallback values passed to config helpers
4. Provider shortcuts via get_base_url(provider="...")

AgentOptions requires model and base_url - use config helpers to resolve them.
"""
import asyncio
import os
from open_agent import query, AgentOptions, TextBlock
from open_agent.config import get_model, get_base_url


async def example_explicit():
    """Example 1: Explicit configuration (simplest)"""
    print("=== Example 1: Explicit Configuration ===")

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="qwen/qwen3-30b-a3b-2507",
        base_url="http://192.168.1.100:1234/v1",  # Example: LM Studio on local network
        max_tokens=50
    )

    print(f"Model: {options.model}")
    print(f"Base URL: {options.base_url}")


async def example_env_vars():
    """Example 2: Environment variables with config helpers"""
    print("\n=== Example 2: Environment Variables ===")

    # Set environment variables
    os.environ["ANY_AGENT_MODEL"] = "qwen/qwen3-30b-a3b-2507"
    os.environ["ANY_AGENT_BASE_URL"] = "http://192.168.1.100:1234/v1"  # Example network server

    # Config helpers read from environment
    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model=get_model(),
        base_url=get_base_url(),
        max_tokens=50
    )

    print(f"Model: {options.model}")
    print(f"Base URL: {options.base_url}")

    # Clean up
    del os.environ["ANY_AGENT_MODEL"]
    del os.environ["ANY_AGENT_BASE_URL"]


async def example_provider_ollama():
    """Example 3: Provider shorthand for base_url"""
    print("\n=== Example 3: Provider Shorthand (Ollama) ===")

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="llama3.1:70b",
        base_url=get_base_url(provider="ollama"),  # Returns http://localhost:11434/v1
        max_tokens=50
    )

    print(f"Model: {options.model}")
    print(f"Base URL: {options.base_url}")


async def example_provider_lmstudio():
    """Example 4: Provider shorthand (LM Studio)"""
    print("\n=== Example 4: Provider Shorthand (LM Studio) ===")

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="qwen2.5-32b-instruct",
        base_url=get_base_url(provider="lmstudio"),  # Returns http://localhost:1234/v1
        max_tokens=50
    )

    print(f"Model: {options.model}")
    print(f"Base URL: {options.base_url}")


async def example_fallbacks():
    """Example 5: Fallback values when env vars not set"""
    print("\n=== Example 5: Fallback Values ===")

    # Ensure env vars are not set
    os.environ.pop("ANY_AGENT_MODEL", None)
    os.environ.pop("ANY_AGENT_BASE_URL", None)

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model=get_model("qwen2.5-32b-instruct"),  # Fallback to this model
        base_url=get_base_url(provider="lmstudio"),  # Fallback to LM Studio
        max_tokens=50
    )

    print(f"Model: {options.model}")
    print(f"Base URL: {options.base_url}")


async def example_priority():
    """Example 6: Priority order demonstration"""
    print("\n=== Example 6: Priority Order ===")

    # Set env vars
    os.environ["ANY_AGENT_MODEL"] = "env-model"
    os.environ["ANY_AGENT_BASE_URL"] = "http://env-server:1234/v1"

    # Explicit parameters override everything
    options1 = AgentOptions(
        system_prompt="Test",
        model="explicit-model",
        base_url="http://explicit:8080/v1"
    )
    print(f"Explicit params → Model: {options1.model}, URL: {options1.base_url}")

    # Env vars override fallbacks
    options2 = AgentOptions(
        system_prompt="Test",
        model=get_model("fallback-model"),
        base_url=get_base_url(provider="ollama")
    )
    print(f"Env + fallbacks → Model: {options2.model}, URL: {options2.base_url}")

    # Clean up
    del os.environ["ANY_AGENT_MODEL"]
    del os.environ["ANY_AGENT_BASE_URL"]

    # Fallbacks when no env vars
    options3 = AgentOptions(
        system_prompt="Test",
        model=get_model("fallback-model"),
        base_url=get_base_url(provider="ollama")
    )
    print(f"Fallbacks only → Model: {options3.model}, URL: {options3.base_url}")


async def main():
    """Run all examples"""
    await example_explicit()
    await example_env_vars()
    await example_provider_ollama()
    await example_provider_lmstudio()
    await example_fallbacks()
    await example_priority()

    print("\n=== Available Provider Shortcuts ===")
    print("lmstudio  → http://localhost:1234/v1")
    print("ollama    → http://localhost:11434/v1")
    print("llamacpp  → http://localhost:8080/v1")
    print("vllm      → http://localhost:8000/v1")

    print("\n=== Recommended Pattern ===")
    print("Use env vars + fallbacks for flexibility:")
    print("  model=get_model('default-model'),")
    print("  base_url=get_base_url(provider='lmstudio')")


if __name__ == "__main__":
    asyncio.run(main())
