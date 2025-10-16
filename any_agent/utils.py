"""OpenAI client utilities"""
import json
import logging
from typing import Any
from openai import AsyncOpenAI
from .types import AgentOptions, TextBlock, ToolUseBlock, ToolUseError

logger = logging.getLogger(__name__)


def create_client(options: AgentOptions) -> AsyncOpenAI:
    """Create configured AsyncOpenAI client"""
    return AsyncOpenAI(
        base_url=options.base_url,
        api_key=options.api_key,
        timeout=60.0
    )


def format_messages(
    system_prompt: str,
    user_prompt: str,
    history: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Format messages for OpenAI API"""
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_prompt})
    return messages


class ToolCallAggregator:
    """
    Stateful aggregator for streaming tool calls.

    OpenAI streams tool calls incrementally:
    - Arguments arrive as partial JSON strings across multiple chunks
    - id and name may appear in separate chunks
    - Multiple tools use 'index' to track which tool is being updated
    """

    def __init__(self):
        self.pending_tools: dict[int, dict[str, Any]] = {}

    def process_chunk(self, chunk) -> TextBlock | None:
        """
        Process a streaming chunk, accumulating tool calls.
        Returns TextBlock if there's text content, None otherwise.
        Tool calls are accumulated internally until finalized.
        """
        try:
            if not chunk.choices:
                return None

            delta = chunk.choices[0].delta

            # Handle text content
            if hasattr(delta, 'content') and delta.content:
                return TextBlock(text=delta.content)

            # Handle tool call deltas
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                for tc in delta.tool_calls:
                    index = tc.index

                    # Initialize tool slot if needed
                    if index not in self.pending_tools:
                        self.pending_tools[index] = {
                            "id": None,
                            "name": None,
                            "arguments_buffer": ""
                        }

                    tool = self.pending_tools[index]

                    # Update id if present
                    if hasattr(tc, 'id') and tc.id:
                        tool["id"] = tc.id

                    # Update name if present
                    if hasattr(tc, 'function') and hasattr(tc.function, 'name') and tc.function.name:
                        tool["name"] = tc.function.name

                    # Accumulate arguments
                    if hasattr(tc, 'function') and hasattr(tc.function, 'arguments') and tc.function.arguments:
                        tool["arguments_buffer"] += tc.function.arguments

            return None

        except Exception as e:
            logger.warning(f"Failed to process chunk: {e}")
            return None

    def finalize_tools(self) -> list[ToolUseBlock | ToolUseError]:
        """
        Finalize all pending tool calls, parsing accumulated JSON arguments.
        Returns list of completed ToolUseBlocks or ToolUseErrors.
        """
        results: list[ToolUseBlock | ToolUseError] = []

        for index, tool in sorted(self.pending_tools.items()):
            # Validate required fields
            if not tool["id"] or not tool["name"]:
                logger.error(f"Tool at index {index} missing id or name: {tool}")
                results.append(ToolUseError(
                    error=f"Tool call missing required fields (id={tool['id']}, name={tool['name']})",
                    raw_data=str(tool)
                ))
                continue

            # Parse arguments JSON
            try:
                if tool["arguments_buffer"]:
                    input_dict = json.loads(tool["arguments_buffer"])
                else:
                    input_dict = {}

                results.append(ToolUseBlock(
                    id=tool["id"],
                    name=tool["name"],
                    input=input_dict
                ))

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse tool arguments JSON: {e}")
                logger.error(f"Raw buffer: {tool['arguments_buffer']}")
                results.append(ToolUseError(
                    error=f"Invalid JSON in tool arguments: {e}",
                    raw_data=tool["arguments_buffer"]
                ))

        # Clear state for next turn
        self.pending_tools.clear()

        return results
