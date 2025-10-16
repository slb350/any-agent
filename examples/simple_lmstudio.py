"""Simple query example with LM Studio"""
import asyncio
from any_agent import query, AgentOptions, TextBlock, ToolUseBlock, ToolUseError


async def main():
    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="qwen2.5-32b-instruct",
        base_url="http://localhost:1234/v1",
        max_turns=1,
        temperature=0.7
    )

    print("Querying LM Studio...")
    print("Question: What is 2+2?")
    print("\nResponse: ", end="", flush=True)

    result = query(prompt="What is 2+2?", options=options)

    response_text = ""
    async for msg in result:
        for block in msg.content:
            if isinstance(block, TextBlock):
                response_text += block.text
                print(block.text, end="", flush=True)
            elif isinstance(block, ToolUseBlock):
                print(f"\n[Tool: {block.name}]", flush=True)
                print(f"  Input: {block.input}")
            elif isinstance(block, ToolUseError):
                print(f"\n[Tool Error: {block.error}]", flush=True)

    print(f"\n\nFull response: {response_text}")


if __name__ == "__main__":
    asyncio.run(main())
