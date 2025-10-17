"""Main client implementation"""
import logging
import json
from typing import AsyncGenerator, Any
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
    aggregator = ToolCallAggregator()
    current_message = AssistantMessage(content=[])

    try:
        response = await client.chat.completions.create(
            model=options.model,
            messages=messages,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            stream=True
        )

        try:
            async for chunk in response:
                # Process text blocks immediately
                text_block = aggregator.process_chunk(chunk)
                if text_block:
                    current_message.content.append(text_block)
                    yield current_message
        finally:
            # Finalize any pending tool calls
            tool_blocks = aggregator.finalize_tools()
            if tool_blocks:
                current_message.content.extend(tool_blocks)
                yield current_message

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise
    finally:
        await client.close()


class Client:
    """
    Multi-turn conversation client.

    Usage:
        async with Client(options) as client:
            await client.query("Hello")
            async for msg in client.receive_messages():
                # Process messages
    """

    def __init__(self, options: AgentOptions):
        self.options = options
        self.client = create_client(options)
        self.message_history: list[dict[str, Any]] = []
        self.turn_count = 0
        self.response_stream = None
        self._aggregator: ToolCallAggregator | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the underlying OpenAI client."""
        await self.client.close()

    async def query(self, prompt: str):
        """Send query and prepare to receive messages"""
        messages = format_messages(
            self.options.system_prompt,
            prompt,
            self.message_history
        )

        user_entry = {
            "role": "user",
            "content": prompt
        }

        try:
            response_stream = await self.client.chat.completions.create(
                model=self.options.model,
                messages=messages,
                max_tokens=self.options.max_tokens,
                temperature=self.options.temperature,
                stream=True
            )
        except Exception:
            self.response_stream = None
            self._aggregator = None
            raise

        self.response_stream = response_stream
        # Initialize aggregator for this turn
        self._aggregator = ToolCallAggregator()
        # Add user message to history only after successful request setup
        self.message_history.append(user_entry)

    async def receive_messages(self) -> AsyncGenerator[TextBlock | ToolUseBlock | ToolUseError, None]:
        """Stream individual blocks from response"""
        if not self.response_stream:
            raise RuntimeError("Call query() first")
        if not self._aggregator:
            raise RuntimeError("Aggregator not initialized")

        assistant_blocks: list[TextBlock | ToolUseBlock | ToolUseError] = []

        # Stream text blocks
        async for chunk in self.response_stream:
            text_block = self._aggregator.process_chunk(chunk)
            if text_block:
                assistant_blocks.append(text_block)
                yield text_block

        # Finalize tool calls
        tool_blocks = self._aggregator.finalize_tools()
        if tool_blocks:
            assistant_blocks.extend(tool_blocks)
            for tool_block in tool_blocks:
                yield tool_block

        # Add assistant response to history with proper structure
        # Preserve both text and tool calls for OpenAI API compatibility
        history_entry = self._format_history_entry(assistant_blocks)
        self.message_history.append(history_entry)

        self.turn_count += 1

        # Check max turns
        if self.turn_count >= self.options.max_turns:
            logger.info(f"Reached max_turns ({self.options.max_turns})")

        # Reset streaming state
        self.response_stream = None
        self._aggregator = None

    def _format_history_entry(
        self,
        blocks: list[TextBlock | ToolUseBlock | ToolUseError]
    ) -> dict[str, Any]:
        """
        Format assistant response for message history.
        Handles mixed text + tool calls per OpenAI API format.
        """
        # Separate text and tools
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in blocks:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input)  # OpenAI expects JSON string
                    }
                })
            # Skip ToolUseError - don't preserve malformed calls in history

        # Build history entry per OpenAI format
        if tool_calls:
            # Assistant used tools
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(text_parts) if text_parts else None,
                "tool_calls": tool_calls
            }
        else:
            # Text-only response
            entry = {
                "role": "assistant",
                "content": "".join(text_parts)
            }

        return entry

    def add_tool_result(
        self,
        tool_call_id: str,
        content: str | dict | list[Any],
        *,
        name: str | None = None
    ) -> None:
        """
        Append a tool execution result to the conversation history.

        This mirrors OpenAI's required `{"role": "tool"}` message so the model
        can see what the tool returned before the next assistant turn.
        """
        if not tool_call_id:
            raise ValueError("tool_call_id cannot be empty")

        if not self._tool_call_known(tool_call_id):
            raise ValueError(f"Unknown tool_call_id: {tool_call_id}")

        if isinstance(content, str):
            message_content = content
        elif isinstance(content, (dict, list)):
            message_content = json.dumps(content)
        else:
            raise TypeError("content must be a str, dict, or list")

        entry: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": message_content
        }

        if name:
            entry["name"] = name

        self.message_history.append(entry)

    def _tool_call_known(self, tool_call_id: str) -> bool:
        """Return True if the tool call id exists in the stored history."""
        for message in reversed(self.message_history):
            if message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls", []):
                if tool_call.get("id") == tool_call_id:
                    return True
        return False

    @property
    def history(self) -> list[dict[str, Any]]:
        """Get full conversation history for agent storage"""
        return self.message_history.copy()

    @property
    def turn_metadata(self) -> dict[str, Any]:
        """Get conversation metadata for agent tracking"""
        return {
            "turn_count": self.turn_count,
            "max_turns": self.options.max_turns
        }
