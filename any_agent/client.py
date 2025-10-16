"""Main client implementation"""
import logging
from typing import AsyncGenerator
from .types import AgentOptions, AssistantMessage, TextBlock, ToolUseBlock, ToolUseError
from .utils import create_client, format_messages, ToolCallAggregator

logger = logging.getLogger(__name__)


async def query(
    prompt: str,
    options: AgentOptions
) -> AsyncGenerator[AssistantMessage, None]:
    """
    Simple single-turn query.

    Usage:
        options = AgentOptions(...)
        result = query("Hello", options)
        async for msg in result:
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(block.text)
                elif isinstance(block, ToolUseBlock):
                    print(f"Tool: {block.name}")
                elif isinstance(block, ToolUseError):
                    print(f"Tool error: {block.error}")
    """
    client = create_client(options)
    messages = format_messages(options.system_prompt, prompt)

    try:
        response = await client.chat.completions.create(
            model=options.model,
            messages=messages,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            stream=True
        )

        current_message = AssistantMessage(content=[])
        aggregator = ToolCallAggregator()

        async for chunk in response:
            # Process text blocks immediately
            text_block = aggregator.process_chunk(chunk)
            if text_block:
                current_message.content.append(text_block)
                yield current_message

        # Finalize any pending tool calls
        tool_blocks = aggregator.finalize_tools()
        if tool_blocks:
            current_message.content.extend(tool_blocks)
            yield current_message

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise
