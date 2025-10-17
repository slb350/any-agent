#!/usr/bin/env python3
"""
Example: Calculator with Tools

Demonstrates using the @tool decorator to give local LLMs function calling abilities.
The agent can perform calculations by calling tools we define.

Usage:
    python examples/calculator_tools.py

Requirements:
    - LM Studio running on http://localhost:1234 (or modify base_url)
    - Model loaded that supports tool/function calling (e.g., qwen2.5-32b-instruct)
"""

import asyncio
import json
from open_agent import tool, Client, AgentOptions, ToolUseBlock


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
    """Run calculator example"""
    print("Calculator Agent with Tools")
    print("=" * 50)
    print()

    # Create tool registry for easy lookup
    tool_registry = {
        "add": add,
        "subtract": subtract,
        "multiply": multiply,
        "divide": divide,
    }

    # Configure agent with tools
    options = AgentOptions(
        system_prompt=(
            "You are a helpful calculator assistant. "
            "Use the provided tools to perform calculations. "
            "Always show your work and explain the result."
        ),
        model="qwen2.5-32b-instruct",  # or your preferred model
        base_url="http://localhost:1234/v1",
        tools=[add, subtract, multiply, divide],
        max_turns=5,
        temperature=0.1,  # Low temperature for precise calculations
    )

    # Example calculations
    queries = [
        "What is 25 plus 17?",
        "Calculate 144 divided by 12",
        "What's 7 times 8, then add 5?",
    ]

    for query in queries:
        print(f"User: {query}")
        print("-" * 50)

        async with Client(options) as client:
            await client.query(query)

            async for block in client.receive_messages():
                if isinstance(block, ToolUseBlock):
                    # Agent wants to use a tool
                    print(f"🔧 Tool call: {block.name}")
                    print(f"   Arguments: {json.dumps(block.input, indent=2)}")

                    # Execute the tool
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
                        await client.query("")  # Empty prompt to continue
                        async for response in client.receive_messages():
                            if hasattr(response, 'text'):
                                print(f"\nAssistant: {response.text}")

                elif hasattr(block, 'text'):
                    # Text response from agent
                    print(f"Assistant: {block.text}")

        print()
        print("=" * 50)
        print()


if __name__ == "__main__":
    asyncio.run(main())
