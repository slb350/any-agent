#!/usr/bin/env python3
"""
Example: Simple Tool Usage

Minimal example showing how to define and use a tool with open-agent-sdk.

Usage:
    python examples/simple_tool.py

Requirements:
    - LM Studio running on http://localhost:1234
    - Model that supports function calling (e.g., qwen2.5-32b-instruct)
"""

import asyncio
from open_agent import tool, query, AgentOptions, ToolUseBlock


# Define a simple tool
@tool(
    name="get_weather",
    description="Get the current weather for a location",
    input_schema={"location": str, "units": str}
)
async def get_weather(args):
    """Mock weather tool - returns fake data"""
    location = args["location"]
    units = args["units"]

    # In a real tool, you'd call a weather API here
    weather_data = {
        "location": location,
        "temperature": 72,
        "conditions": "sunny",
        "units": units,
    }

    return weather_data


async def main():
    """Run simple tool example"""
    print("Simple Tool Example")
    print("=" * 50)

    # Configure agent with the tool
    options = AgentOptions(
        system_prompt="You are a helpful assistant with access to weather data.",
        model="qwen2.5-32b-instruct",
        base_url="http://localhost:1234/v1",
        tools=[get_weather],  # Register our tool
    )

    # Query the agent
    user_prompt = "What's the weather like in Paris? Use Celsius."

    print(f"User: {user_prompt}\n")

    result = query(user_prompt, options)

    async for message in result:
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                # The model wants to call our tool
                print(f"🔧 Model called tool: {block.name}")
                print(f"   Arguments: {block.input}")

                # Execute the tool
                tool_result = await get_weather.execute(block.input)
                print(f"   Result: {tool_result}\n")

                # In a real app, you'd send this back to continue the conversation
                # For this simple example, we just show what happened

            elif hasattr(block, 'text'):
                # Text response from the model
                print(f"Assistant: {block.text}")


if __name__ == "__main__":
    asyncio.run(main())
