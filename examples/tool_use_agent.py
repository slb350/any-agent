"""
Complete tool use example showing the full execution loop.

This example demonstrates:
1. Detecting tool calls in the stream
2. Executing tools locally
3. Using add_tool_result() to feed results back
4. Continuing conversation with tool context
5. Helper pattern for automatic tool execution

Prerequisites:
    - Ollama running with a model that supports tools
    - Or any OpenAI-compatible endpoint with tool support
"""
import ast
import asyncio
from typing import Any, Awaitable, Callable

from any_agent import Client, AgentOptions, TextBlock, ToolUseBlock, ToolUseError
from any_agent.config import get_base_url


# Example tools that an agent might use
async def get_weather(location: str) -> dict[str, Any]:
    """Mock weather tool"""
    # In real agent, this would call a weather API
    return {
        "location": location,
        "temperature": 72,
        "conditions": "Partly cloudy",
        "forecast": "Clear skies expected tomorrow"
    }


def _safe_eval(expression: str) -> Any:
    """
    Evaluate a simple arithmetic expression safely.

    Supports literals and +, -, *, /, **, parentheses. Rejects everything else.
    """
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.FloorDiv,
        ast.Call,
    )

    tree = ast.parse(expression, mode="eval")

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported expression element: {type(node).__name__}")
        if isinstance(node, ast.Call):
            raise ValueError("Function calls are not allowed")

    return eval(compile(tree, "<string>", "eval"), {"__builtins__": {}}, {})


async def calculate(expression: str) -> dict[str, Any]:
    """Mock calculator tool using a restricted arithmetic parser."""
    try:
        result = _safe_eval(expression)
        return {"result": result, "expression": expression}
    except Exception as exc:
        return {"error": str(exc), "expression": expression}


async def search_web(query: str, max_results: int = 5) -> dict[str, Any]:
    """Mock web search tool"""
    # In real agent, this would call a search API
    return {
        "query": query,
        "results": [
            {"title": f"Result {i+1} for '{query}'", "url": f"https://example.com/{i}"}
            for i in range(max_results)
        ]
    }


# Tool registry - maps tool names to functions
TOOL_REGISTRY: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_web": search_web
}


async def example_manual_tool_handling():
    """Example 1: Manual tool detection and execution"""
    print("=" * 60)
    print("EXAMPLE 1: Manual Tool Handling")
    print("=" * 60)

    options = AgentOptions(
        system_prompt="""You are a helpful assistant with access to tools.

Available tools:
- get_weather(location: str): Get current weather
- calculate(expression: str): Perform calculations
- search_web(query: str): Search the web

Use tools when appropriate to answer questions.""",
        model="kimi-k2:1t-cloud",  # Use a model that supports tools
        base_url=get_base_url(provider="ollama"),
        max_turns=3,
        temperature=0.7,
        timeout=60.0
    )

    async with Client(options) as client:
        # Ask something that requires tools
        user_query = "What's the weather in San Francisco and what is 42 * 17?"
        print(f"\nUser: {user_query}")

        await client.query(user_query)

        print("Assistant: ", end="", flush=True)
        tool_calls_made = []

        # Receive and process the response
        async for block in client.receive_messages():
            if isinstance(block, TextBlock):
                print(block.text, end="", flush=True)

            elif isinstance(block, ToolUseBlock):
                print(f"\n[Tool Call: {block.name}]")
                print(f"  Args: {block.input}")
                tool_calls_made.append(block)

            elif isinstance(block, ToolUseError):
                print(f"\n[Tool Error: {block.error}]")

        # Execute tools if any were called
        if tool_calls_made:
            print(f"\n\nExecuting {len(tool_calls_made)} tool(s)...")

            for tool_call in tool_calls_made:
                if tool_call.name in TOOL_REGISTRY:
                    tool_func = TOOL_REGISTRY[tool_call.name]

                    # Execute the tool
                    try:
                        result = await tool_func(**tool_call.input)
                        print(f"✓ {tool_call.name} returned: {result}")

                        # Feed result back to model
                        client.add_tool_result(
                            tool_call.id,
                            result,
                            name=tool_call.name  # Optional but helpful
                        )
                    except Exception as e:
                        error_result = {"error": str(e)}
                        print(f"✗ {tool_call.name} failed: {e}")
                        client.add_tool_result(tool_call.id, error_result)
                else:
                    print(f"✗ Unknown tool: {tool_call.name}")
                    client.add_tool_result(
                        tool_call.id,
                        {"error": f"Unknown tool: {tool_call.name}"}
                    )

            # Continue conversation with tool results
            print("\nContinuing with tool results...")
            await client.query("Please summarize the results.")

            print("Assistant: ", end="", flush=True)
            async for block in client.receive_messages():
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)

        print(f"\n\n✓ Conversation complete. Turns: {client.turn_count}")


async def execute_tools_helper(
    client: Client,
    tool_registry: dict[str, Callable[..., Awaitable[dict[str, Any]]]],
    tool_calls: list[ToolUseBlock]
) -> list[dict[str, Any]]:
    """
    Helper function to execute tools and add results to conversation.

    This is the pattern you might extract into a base Agent class.
    """
    results = []

    for tool_call in tool_calls:
        if tool_call.name not in tool_registry:
            error = {"error": f"Unknown tool: {tool_call.name}"}
            client.add_tool_result(tool_call.id, error)
            results.append(error)
            continue

        tool_func = tool_registry[tool_call.name]

        try:
            # Execute the tool
            result = await tool_func(**tool_call.input)

            # Add to conversation history
            client.add_tool_result(
                tool_call.id,
                result,
                name=tool_call.name
            )

            results.append(result)

        except Exception as e:
            error = {"error": str(e), "tool": tool_call.name}
            client.add_tool_result(tool_call.id, error)
            results.append(error)

    return results


async def example_with_helper():
    """Example 2: Using helper pattern for cleaner code"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Helper Pattern")
    print("=" * 60)

    options = AgentOptions(
        system_prompt="""You are a helpful assistant with access to tools.

Available tools:
- get_weather(location: str): Get current weather
- calculate(expression: str): Perform calculations
- search_web(query: str, max_results: int = 5): Search the web

Always use tools when they would help answer the question.""",
        model="kimi-k2:1t-cloud",
        base_url=get_base_url(provider="ollama"),
        max_turns=5,
        temperature=0.7
    )

    async with Client(options) as client:
        # More complex query requiring multiple tools
        user_query = "Search for 'Python async programming' and calculate 2**10"
        print(f"\nUser: {user_query}")

        await client.query(user_query)

        # Collect response and tool calls
        response_text = ""
        tool_calls = []

        async for block in client.receive_messages():
            if isinstance(block, TextBlock):
                response_text += block.text
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(block)

        print(f"Assistant: {response_text}")

        if tool_calls:
            print(f"\n[Detected {len(tool_calls)} tool call(s)]")

            # Use helper to execute all tools
            results = await execute_tools_helper(client, TOOL_REGISTRY, tool_calls)

            for tool_call, result in zip(tool_calls, results):
                print(f"  {tool_call.name}: {result}")

            # Get final response with tool results
            await client.query("Now provide a complete answer based on the tool results.")

            print("\nAssistant: ", end="", flush=True)
            async for block in client.receive_messages():
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)

        print(f"\n\n✓ Helper pattern complete. History: {len(client.history)} messages")


class ToolAgent:
    """
    Example 3: Agent class with automatic tool execution.

    This pattern encapsulates tool management for reusable agents.
    """

    def __init__(self, options: AgentOptions, tools: dict[str, Callable]):
        self.options = options
        self.tools = tools
        self.client = None

    async def __aenter__(self):
        client = Client(self.options)
        await client.__aenter__()
        self.client = client
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)

    async def chat(self, message: str, auto_execute: bool = True) -> str:
        """
        Send message and optionally auto-execute tools.

        Returns the complete response text.
        """
        if not self.client:
            raise RuntimeError("ToolAgent is not active. Use 'async with ToolAgent(...)'.")

        await self.client.query(message)

        response_text = ""
        tool_calls = []

        # Process initial response
        async for block in self.client.receive_messages():
            if isinstance(block, TextBlock):
                response_text += block.text
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(block)

        # Auto-execute tools if enabled and tools were called
        if auto_execute and tool_calls:
            print(f"[Auto-executing {len(tool_calls)} tools]")

            await execute_tools_helper(self.client, self.tools, tool_calls)

            # Get response with tool results
            await self.client.query("Continue with the tool results.")

            async for block in self.client.receive_messages():
                if isinstance(block, TextBlock):
                    response_text += block.text

        return response_text


async def example_agent_class():
    """Example 3: Using an Agent class for cleaner API"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Agent Class Pattern")
    print("=" * 60)

    options = AgentOptions(
        system_prompt="You are a helpful assistant with tool access.",
        model="kimi-k2:1t-cloud",
        base_url=get_base_url(provider="ollama"),
        max_turns=10
    )

    async with ToolAgent(options, TOOL_REGISTRY) as agent:
        # Simple API - tools execute automatically
        response = await agent.chat(
            "What's the weather in Tokyo and what's 100 factorial?"
        )
        print(f"Response: {response}")

        # Continue conversation
        response = await agent.chat(
            "Is that temperature in Celsius or Fahrenheit?"
        )
        print(f"Follow-up: {response}")


async def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("TOOL USE EXAMPLES")
    print("=" * 60)
    print("\nThese examples show different patterns for handling tool calls.")
    print("Pick the pattern that fits your agent's complexity.\n")

    # Example 1: Manual handling (most control)
    await example_manual_tool_handling()

    # Example 2: Helper function (balance of control and convenience)
    await example_with_helper()

    # Example 3: Agent class (most convenient)
    await example_agent_class()

    print("\n" + "=" * 60)
    print("All examples complete!")
    print("\nKey takeaways:")
    print("1. ToolUseBlock gives you tool calls from the model")
    print("2. Execute tools locally with your own logic")
    print("3. Use add_tool_result() to feed results back")
    print("4. Continue conversation to get final answer")
    print("5. Consider helper functions or classes for production")


if __name__ == "__main__":
    # Note: This example assumes you have a model that supports tool use.
    # Not all local models support tools - check your model's capabilities.
    asyncio.run(main())
