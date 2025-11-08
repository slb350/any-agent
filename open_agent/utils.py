"""
OpenAI client utilities for local LLM integration

This module provides low-level utilities for working with OpenAI-compatible APIs.
These functions handle client initialization, message formatting, and streaming
response parsing - the foundational plumbing that powers the SDK's client.py.

Key Components:
===============
1. create_client(): Initialize AsyncOpenAI client with AgentOptions
2. format_messages(): Convert prompts to OpenAI chat completion format
3. format_tools(): Convert Tool instances to OpenAI function calling format
4. ToolCallAggregator: Stateful parser for streaming tool call chunks

The ToolCallAggregator is the most complex component, handling the challenge
of parsing tool calls that arrive incrementally across multiple streaming chunks.

Design Notes:
=============
- All functions are pure/stateless except ToolCallAggregator (stateful by design)
- Uses OpenAI's official AsyncOpenAI client for HTTP communication
- Handles provider quirks (cumulative vs incremental text streaming)
- Robust error handling for malformed tool call JSON
- Logging for debugging streaming issues
"""

import json
import logging
from typing import Any, TYPE_CHECKING
from openai import AsyncOpenAI
from .types import AgentOptions, TextBlock, ToolUseBlock, ToolUseError

# TYPE_CHECKING is only True during static type checking (mypy, pyright, etc.)
# This prevents circular import at runtime (tools.py imports from types.py,
# and we need Tool type here)
if TYPE_CHECKING:
    from .tools import Tool

# Module-level logger for debugging streaming issues and tool call parsing
logger = logging.getLogger(__name__)


def create_client(options: AgentOptions) -> AsyncOpenAI:
    """
    Create and configure AsyncOpenAI client from AgentOptions.

    This is a simple wrapper that initializes the official OpenAI Python client
    with settings from AgentOptions. The AsyncOpenAI client handles all HTTP
    communication, retry logic, and connection pooling.

    Args:
        options: AgentOptions instance with base_url, api_key, timeout configured

    Returns:
        AsyncOpenAI: Configured async client ready for chat completions

    Example:
        >>> opts = AgentOptions(
        ...     system_prompt="...",
        ...     model="qwen2.5-32b",
        ...     base_url="http://localhost:1234/v1",
        ...     api_key="not-needed",
        ...     timeout=60.0
        ... )
        >>> client = create_client(opts)
        >>> # client is ready for client.chat.completions.create(...)

    Note:
        Most local LLM servers don't require authentication, so api_key can
        be any non-empty string (AgentOptions defaults to "not-needed").
        Some servers may require a specific key for multi-user setups.
    """
    return AsyncOpenAI(
        base_url=options.base_url,  # OpenAI-compatible endpoint (e.g., http://localhost:1234/v1)
        api_key=options.api_key,  # API key (most local servers don't check this)
        timeout=options.timeout  # HTTP request timeout in seconds
    )


def format_messages(
    system_prompt: str,
    user_prompt: str,
    history: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """
    Format system prompt, history, and user prompt into OpenAI chat completion format.

    Constructs the messages array for OpenAI's chat completion API following
    the standard format:
    1. System message (always first, defines agent behavior)
    2. Conversation history (optional, user/assistant/tool messages)
    3. Current user message (what the user just said)

    Args:
        system_prompt: System-level instructions defining agent role/behavior
            Example: "You are a helpful coding assistant."

        user_prompt: Current user input to respond to
            Example: "What's the weather in NYC?"

        history: Optional conversation history as list of message dicts
            Each dict has {"role": "user"/"assistant"/"tool", "content": ...}
            Default: None (no history, first turn)

    Returns:
        list[dict[str, Any]]: Messages array ready for OpenAI API
            Format: [
                {"role": "system", "content": system_prompt},
                ...history messages...,
                {"role": "user", "content": user_prompt}
            ]

    Examples:
        First turn (no history):
        >>> format_messages("You are helpful", "Hello")
        [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]

        Multi-turn with history:
        >>> history = [
        ...     {"role": "user", "content": "Hi"},
        ...     {"role": "assistant", "content": "Hello!"}
        ... ]
        >>> format_messages("You are helpful", "How are you?", history)
        [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"}
        ]

    Note:
        This function does NOT validate history contents. The caller (client.py)
        is responsible for ensuring history has valid message structure.
    """
    # Always start with system message (defines agent behavior)
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    # Add conversation history if provided (maintains context)
    if history:
        messages.extend(history)

    # Add current user prompt (what we're responding to)
    messages.append({"role": "user", "content": user_prompt})

    return messages


def format_tools(tools: list["Tool"]) -> list[dict[str, Any]]:
    """
    Convert Tool instances to OpenAI function calling format.

    Args:
        tools: List of Tool instances from @tool decorator

    Returns:
        List of tool definitions in OpenAI format

    Example:
        >>> tools = [add_tool, multiply_tool]
        >>> format_tools(tools)
        [
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add two numbers",
                    "parameters": {...}
                }
            },
            ...
        ]
    """
    return [tool.to_openai_format() for tool in tools]


class ToolCallAggregator:
    """
    Stateful aggregator for parsing streaming tool calls from OpenAI-compatible APIs.

    The Challenge:
    ==============
    When LLMs generate function calls during streaming responses, the tool call
    data arrives incrementally across multiple chunks, NOT as a single complete
    JSON object. This creates several parsing challenges:

    1. PARTIAL JSON: Arguments arrive as fragments: '{"ci' -> 'ty": "NY' -> 'C"}'
       - Can't parse JSON until all chunks received
       - Must buffer and accumulate

    2. MULTIPLE FIELDS IN SEPARATE CHUNKS:
       - Chunk 1: tool_call.id = "call_abc123"
       - Chunk 2: tool_call.function.name = "get_weather"
       - Chunk 3-5: tool_call.function.arguments = '{"city":' + '"NYC"' + '}'

    3. MULTIPLE TOOL CALLS IN PARALLEL:
       - LLM can request multiple tools simultaneously
       - Each has an 'index' field to track which tool is being updated
       - Index 0: get_weather chunks, Index 1: get_time chunks (interleaved)

    4. TEXT INTERLEAVED WITH TOOL CALLS:
       - Some providers send text before/after tool calls
       - Must extract text separately from tool call data

    How It Works:
    =============
    1. process_chunk(chunk):
       - Called for each streaming chunk from OpenAI API
       - Extracts text content (returns TextBlock immediately)
       - Accumulates tool call data in self.pending_tools dict
       - Returns None for tool call chunks (accumulated internally)

    2. finalize_tools():
       - Called when streaming completes
       - Parses accumulated JSON arguments for each tool
       - Returns list of ToolUseBlock or ToolUseError
       - Clears internal state for next turn

    State Management:
    =================
    - pending_tools: Dict[index -> tool_data]
      Stores partially-received tool calls indexed by their position
      Structure: {
          index: {
              "id": "call_abc123" or None (until received),
              "name": "get_weather" or None (until received),
              "arguments_buffer": '{"city":"NYC"}' (accumulated string)
          }
      }

    - _text_accumulator: str
      Handles providers that send cumulative text vs incremental deltas
      Some providers re-send all text so far, others send only new chars

    Provider Compatibility:
    =======================
    Different providers have different streaming behaviors:
    - OpenAI: Incremental arguments, one chunk at a time
    - Ollama: May send cumulative text (full message each time)
    - llama.cpp: Varies by configuration
    - vLLM: Generally incremental

    This class handles both cumulative and incremental streaming patterns.

    Example Usage:
    ==============
    >>> aggregator = ToolCallAggregator()
    >>>
    >>> # Process streaming chunks
    >>> async for chunk in stream:
    ...     text_block = aggregator.process_chunk(chunk)
    ...     if text_block:
    ...         yield text_block  # Emit text immediately
    ...
    >>> # When stream completes, finalize tool calls
    >>> tool_blocks = aggregator.finalize_tools()
    >>> for block in tool_blocks:
    ...     yield block  # Emit tool calls

    Error Handling:
    ===============
    - Missing id/name: Returns ToolUseError with descriptive message
    - Invalid JSON in arguments: Returns ToolUseError with raw data
    - Malformed chunks: Logs warning, continues processing
    - Exceptions during parsing: Caught and logged, doesn't crash

    Thread Safety:
    ==============
    NOT thread-safe. Each stream should have its own aggregator instance.
    """

    def __init__(self):
        """
        Initialize empty aggregator for a new streaming response.

        Creates empty state dictionaries for accumulating tool calls and text.
        This should be called once per streaming request.
        """
        # Maps tool index (from chunk) to partially-accumulated tool data
        # index: int -> {"id": str|None, "name": str|None, "arguments_buffer": str}
        self.pending_tools: dict[int, dict[str, Any]] = {}

        # Accumulates text content to detect cumulative vs incremental streaming
        # Some providers resend all text so far, others send only deltas
        self._text_accumulator: str = ""

    def process_chunk(self, chunk) -> TextBlock | None:
        """
        Process a single streaming chunk from the OpenAI API.

        Extracts text content (if present) and accumulates tool call data (if present).
        Text is returned immediately as a TextBlock for streaming to the user.
        Tool calls are buffered internally and returned later by finalize_tools().

        Args:
            chunk: Streaming chunk from AsyncOpenAI chat completion stream
                Expected structure: chunk.choices[0].delta
                Delta may contain: .content (text) or .tool_calls (function calls)

        Returns:
            TextBlock if text content found, None otherwise
            - Text is returned immediately for real-time streaming
            - Tool calls return None (accumulated for later finalization)

        Error Handling:
            Catches all exceptions to prevent single bad chunk from killing stream.
            Logs warnings for debugging but continues processing.

        Example:
            >>> chunk1.choices[0].delta.content = "I'll check"
            >>> aggregator.process_chunk(chunk1)
            TextBlock(text="I'll check")

            >>> chunk2.choices[0].delta.tool_calls[0].function.name = "get_weather"
            >>> aggregator.process_chunk(chunk2)
            None  # Tool call buffered internally
        """
        try:
            # Chunk may be empty (end marker) - skip it
            if not chunk.choices:
                return None

            # Extract delta (incremental update) from chunk
            delta = chunk.choices[0].delta

            # Handle text content (if present in this chunk)
            # _extract_new_text handles cumulative vs incremental streaming
            new_text = self._extract_new_text(delta)
            if new_text:
                # Return text immediately for real-time streaming to user
                return TextBlock(text=new_text)

            # Handle tool call deltas (if present in this chunk)
            # Tool calls may be interleaved with text or arrive separately
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                # Loop through tool call updates in this chunk
                # Multiple tool calls can be updated in one chunk (different indices)
                for tc in delta.tool_calls:
                    # Index identifies which tool call this update belongs to
                    # Index 0 = first tool, index 1 = second tool, etc.
                    index = tc.index

                    # Initialize storage for this tool index if first time seeing it
                    if index not in self.pending_tools:
                        self.pending_tools[index] = {
                            "id": None,  # Tool call ID (e.g., "call_abc123")
                            "name": None,  # Tool name (e.g., "get_weather")
                            "arguments_buffer": ""  # JSON arguments (accumulated string)
                        }

                    # Get reference to this tool's data
                    tool = self.pending_tools[index]

                    # Update tool ID if present in this chunk
                    # ID usually arrives in first chunk for this tool
                    if hasattr(tc, 'id') and tc.id:
                        tool["id"] = tc.id

                    # Update tool name if present in this chunk
                    # Name usually arrives early (same chunk as ID or shortly after)
                    if hasattr(tc, 'function') and hasattr(tc.function, 'name') and tc.function.name:
                        tool["name"] = tc.function.name

                    # Accumulate arguments chunk (partial JSON string)
                    # Arguments arrive incrementally: '{"ci' + 'ty":"NY' + 'C"}'
                    # We concatenate all chunks and parse JSON only when finalized
                    if hasattr(tc, 'function') and hasattr(tc.function, 'arguments') and tc.function.arguments:
                        tool["arguments_buffer"] += tc.function.arguments

            # Tool call chunks don't return anything - accumulated internally
            return None

        except Exception as e:
            # Catch all exceptions to prevent stream failures from bad chunks
            # Log for debugging but don't crash - continue processing remaining chunks
            logger.warning(f"Failed to process chunk: {e}")
            return None

    def finalize_tools(self) -> list[ToolUseBlock | ToolUseError]:
        """
        Finalize all pending tool calls by parsing accumulated JSON arguments.

        Called when streaming completes to convert buffered tool call data into
        completed ToolUseBlock instances (or ToolUseError if parsing fails).

        Process:
        1. Iterate through all accumulated tools (sorted by index for consistency)
        2. Validate that id and name were received
        3. Parse arguments_buffer as JSON
        4. Create ToolUseBlock (success) or ToolUseError (failure)
        5. Clear internal state for next streaming request

        Returns:
            list[ToolUseBlock | ToolUseError]: Completed tool calls
            - ToolUseBlock: Successfully parsed tool call ready for execution
            - ToolUseError: Malformed tool call with error message

        Validation Errors (returns ToolUseError):
            - Missing id or name (incomplete tool call)
            - Invalid JSON in arguments_buffer (parse error)

        Example:
            >>> aggregator.pending_tools = {
            ...     0: {"id": "call_1", "name": "add", "arguments_buffer": '{"a":1,"b":2}'}
            ... }
            >>> aggregator.finalize_tools()
            [ToolUseBlock(id="call_1", name="add", input={"a": 1, "b": 2})]

        State Management:
            Clears self.pending_tools and self._text_accumulator after finalization.
            Aggregator is ready for a new streaming request after this call.

        Note:
            Always call this after streaming completes, even if no tool calls expected.
            It's safe to call when pending_tools is empty (returns empty list).
        """
        results: list[ToolUseBlock | ToolUseError] = []

        # Process tools in sorted order (by index) for deterministic output
        for index, tool in sorted(self.pending_tools.items()):
            # Validate that we received both id and name during streaming
            # If either is missing, the tool call is incomplete/malformed
            if not tool["id"] or not tool["name"]:
                logger.error(f"Tool at index {index} missing id or name: {tool}")
                results.append(ToolUseError(
                    error=f"Tool call missing required fields (id={tool['id']}, name={tool['name']})",
                    raw_data=str(tool)  # Include partial data for debugging
                ))
                continue  # Skip this tool, continue with others

            # Parse the accumulated JSON arguments buffer
            try:
                if tool["arguments_buffer"]:
                    # Parse the complete JSON string (accumulated from all chunks)
                    input_dict = json.loads(tool["arguments_buffer"])
                else:
                    # No arguments provided (empty buffer) - use empty dict
                    # Some tools have no parameters, which is valid
                    input_dict = {}

                # Successfully parsed - create ToolUseBlock
                results.append(ToolUseBlock(
                    id=tool["id"],  # Tool call ID for correlation
                    name=tool["name"],  # Tool function name
                    input=input_dict  # Parsed arguments as dict
                ))

            except json.JSONDecodeError as e:
                # JSON parsing failed - buffer contains invalid/incomplete JSON
                # This can happen if streaming was interrupted or LLM generated bad JSON
                logger.error(f"Failed to parse tool arguments JSON: {e}")
                logger.error(f"Raw buffer: {tool['arguments_buffer']}")
                results.append(ToolUseError(
                    error=f"Invalid JSON in tool arguments: {e}",
                    raw_data=tool["arguments_buffer"]  # Include raw JSON for debugging
                ))

        # Clear all state for next streaming request
        # After finalization, aggregator is ready to be reused
        self.pending_tools.clear()
        self._text_accumulator = ""

        return results

    def _extract_new_text(self, delta) -> str | None:
        """
        Extract new text content from a streaming delta, handling provider differences.

        Handles two different streaming patterns used by various providers:
        1. Incremental: Each chunk contains only NEW text ("Hello" + " world")
        2. Cumulative: Each chunk contains ALL text so far ("Hello" + "Hello world")

        Args:
            delta: Streaming delta from chunk.choices[0].delta
                May have .content field with text

        Returns:
            str | None: New text to emit (None if no text or empty)

        Provider Behavior:
            - OpenAI: Incremental (each chunk is new text)
            - Ollama: May be cumulative (resends all text)
            - llama.cpp: Configurable (depends on server settings)

        Algorithm:
            1. Extract content from delta
            2. Normalize content format (handle string/list/dict variations)
            3. Check if new text starts with previously accumulated text:
               - Yes: Cumulative provider, emit only the delta
               - No: Incremental provider, emit the chunk directly
            4. Update accumulator for next comparison

        Example (Incremental):
            >>> delta1.content = "Hello"
            >>> self._extract_new_text(delta1)
            "Hello"  # Accumulator: "Hello"
            >>> delta2.content = " world"
            >>> self._extract_new_text(delta2)
            " world"  # Accumulator: "Hello world"

        Example (Cumulative):
            >>> delta1.content = "Hello"
            >>> self._extract_new_text(delta1)
            "Hello"  # Accumulator: "Hello"
            >>> delta2.content = "Hello world"  # Resends "Hello"!
            >>> self._extract_new_text(delta2)
            " world"  # Detected cumulative, emit only delta
        """
        # Extract content field from delta (may be None)
        content = getattr(delta, "content", None)
        if not content:
            return None  # No text in this chunk

        # Normalize content to plain string (handles various formats)
        text = self._normalise_content(content)
        if text is None:
            return None  # Content was empty or malformed

        # Check if this is cumulative streaming (text starts with what we've seen)
        if self._text_accumulator and text.startswith(self._text_accumulator):
            # Cumulative: Provider re-sent previous text + new text
            # Extract only the new portion by slicing off the prefix
            new_text = text[len(self._text_accumulator):]
            # Update accumulator to full text (for next comparison)
            self._text_accumulator = text
            return new_text or None  # Return None if delta was empty

        # Incremental streaming: This chunk is entirely new text
        new_text = text
        # Append to accumulator (builds up full text over time)
        self._text_accumulator += new_text
        return new_text or None

    @staticmethod
    def _normalise_content(content) -> str | None:
        """
        Normalize various delta.content formats into a plain string.

        Different LLM providers return content in different formats:
        - Simple string: "Hello world"
        - List of strings: ["Hello", " world"]
        - List of dicts: [{"text": "Hello"}, {"text": " world"}]
        - List of objects: [TextChunk(text="Hello"), TextChunk(text=" world")]

        This function handles all these variations and returns a single string.

        Args:
            content: Delta content in any supported format

        Returns:
            str | None: Normalized plain text string, or None if empty/invalid

        Supported Formats:
            1. String: content = "Hello world"
               Returns: "Hello world"

            2. List of strings: content = ["Hello", " world"]
               Returns: "Hello world"

            3. List of dicts: content = [{"text": "Hello"}, {"text": " world"}]
               Returns: "Hello world"

            4. List of objects: content = [TextChunk(text="Hello"), ...]
               Returns: "Hello world"

        Edge Cases:
            - Empty string/list: Returns None
            - Mixed list (strings + dicts + objects): Extracts text from all
            - Missing "text" key in dict: Skips that item
            - Non-text content: Ignored (no error)

        Example:
            >>> _normalise_content("Hello")
            "Hello"

            >>> _normalise_content(["Hello", " world"])
            "Hello world"

            >>> _normalise_content([{"text": "Hello"}, {"text": " world"}])
            "Hello world"
        """
        # Case 1: Already a plain string (most common case)
        if isinstance(content, str):
            return content

        # Case 2: List of mixed content (strings, dicts, objects)
        if isinstance(content, list):
            parts: list[str] = []  # Accumulate text parts

            for item in content:
                # Subcase 2a: Item is a plain string
                if isinstance(item, str):
                    parts.append(item)

                # Subcase 2b: Item is a dict with "text" key
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:  # Only append if text exists and is non-empty
                        parts.append(text)

                # Subcase 2c: Item is an object with .text attribute
                else:
                    text = getattr(item, "text", None)
                    if text:  # Only append if text exists and is non-empty
                        parts.append(text)

            # Join all extracted text parts into single string
            return "".join(parts) if parts else None

        # Case 3: Unknown format - return None
        # This handles None, numbers, booleans, etc. (not text content)
        return None
