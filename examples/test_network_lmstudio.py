"""Test with network LM Studio server"""
import asyncio
from any_agent import query, AgentOptions, TextBlock, ToolUseBlock, ToolUseError


async def main():
    # Example: Testing with LM Studio on a network server
    # Replace with your actual server URL
    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="qwen/qwen3-30b-a3b-2507",
        base_url="http://192.168.1.100:1234/v1",  # Example network server
        max_turns=1,
        temperature=0.7,
        max_tokens=200
    )

    print("Testing Any-Agent SDK with network LM Studio...")
    print(f"Server: {options.base_url}")
    print(f"Model: {options.model}")
    print("\nQuestion: What is the capital of France?")
    print("\nResponse: ", end="", flush=True)

    try:
        result = query(prompt="What is the capital of France?", options=options)

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

        print(f"\n\n✓ Success! Received {len(response_text)} characters")

    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
