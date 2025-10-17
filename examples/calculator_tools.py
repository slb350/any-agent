#!/usr/bin/env python3
"""
Example: Calculator with Tools

Demonstrates using the @tool decorator to give local LLMs function calling abilities.
Shows both automatic tool execution (recommended) and manual execution (advanced).

Usage:
    python examples/calculator_tools.py

Requirements:
    - LM Studio running on http://localhost:1234 (or modify base_url)
    - Model loaded that supports tool/function calling (e.g., qwen2.5-32b-instruct)
"""

import asyncio
from open_agent import tool, Client, AgentOptions, TextBlock, ToolUseBlock


# Define calculator tools
@tool("add", "Add two numbers", {"a": float, "b": float})
async def add(args):
    """Add two numbers together"""
    result = args["a"] + args["b"]
    return {"result": result}


@tool("subtract", "Subtract two numbers", {"a": float, "b": float})
async def subtract(args):
    """Subtract b from a"""
    result = args["a"] - args["b"]
    return {"result": result}


@tool("multiply", "Multiply two numbers", {"a": float, "b": float})
async def multiply(args):
    """Multiply two numbers"""
    result = args["a"] * args["b"]
    return {"result": result}


@tool("divide", "Divide two numbers", {"a": float, "b": float})
async def divide(args):
    """Divide a by b"""
    if args["b"] == 0:
        return {"error": "Cannot divide by zero"}
    result = args["a"] / args["b"]
    return {"result": result}


async def main():
    """Run calculator examples - both automatic and manual modes"""

    # Example calculations
    queries = [
        "What is 25 plus 17?",
        "Calculate 144 divided by 12",
        "What's 7 times 8, then add 5?",
    ]

    # ========================================
    # AUTOMATIC TOOL EXECUTION (Recommended)
    # ========================================
    print("=" * 70)
    print("AUTOMATIC TOOL EXECUTION (Recommended)")
    print("=" * 70)
    print("Tools execute automatically - simple and clean code!\n")

    auto_options = AgentOptions(
        system_prompt=(
            "You are a helpful calculator assistant. "
            "Use the provided tools to perform calculations. "
            "Always show your work and explain the result."
        ),
        model="qwen2.5-32b-instruct",
        base_url="http://localhost:1234/v1",
        tools=[add, subtract, multiply, divide],
        auto_execute_tools=True,  # 🔥 Enable automatic execution
        max_tool_iterations=10,
        max_turns=5,
        temperature=0.1,
    )

    for query in queries:
        print(f"User: {query}")
        print("-" * 50)

        async with Client(auto_options) as client:
            await client.query(query)

            # Simply iterate - tools execute automatically!
            async for block in client.receive_messages():
                if isinstance(block, ToolUseBlock):
                    print(f"🔧 Tool: {block.name}({block.input})")
                elif isinstance(block, TextBlock):
                    print(f"Assistant: {block.text}")

        print()

    # ========================================
    # MANUAL TOOL EXECUTION (Advanced)
    # ========================================
    print("\n" + "=" * 70)
    print("MANUAL TOOL EXECUTION (Advanced)")
    print("=" * 70)
    print("For when you need custom execution logic or result handling.\n")

    # Create tool registry for manual lookup
    tool_registry = {
        "add": add,
        "subtract": subtract,
        "multiply": multiply,
        "divide": divide,
    }

    manual_options = AgentOptions(
        system_prompt=(
            "You are a helpful calculator assistant. "
            "Use the provided tools to perform calculations. "
            "Always show your work and explain the result."
        ),
        model="qwen2.5-32b-instruct",
        base_url="http://localhost:1234/v1",
        tools=[add, subtract, multiply, divide],
        auto_execute_tools=False,  # Manual mode
        max_turns=5,
        temperature=0.1,
    )

    # Just show one example in manual mode
    query = queries[0]
    print(f"User: {query}")
    print("-" * 50)

    async with Client(manual_options) as client:
        await client.query(query)

        async for block in client.receive_messages():
            if isinstance(block, ToolUseBlock):
                # You manually execute the tool
                print(f"🔧 Tool call: {block.name}")
                print(f"   Arguments: {block.input}")

                tool_func = tool_registry.get(block.name)
                if tool_func:
                    result = await tool_func.execute(block.input)
                    print(f"   Result: {result}")

                    # Send result back to agent
                    await client.add_tool_result(
                        tool_call_id=block.id,
                        content=result,
                        name=block.name
                    )

                    # Continue conversation to get agent's response
                    await client.query("")
                    async for response in client.receive_messages():
                        if isinstance(response, TextBlock):
                            print(f"Assistant: {response.text}")

            elif isinstance(block, TextBlock):
                print(f"Assistant: {block.text}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
