"""Main client implementation"""
import asyncio
import logging
import json
from typing import AsyncGenerator, Any
from .types import AgentOptions, AssistantMessage, TextBlock, ToolUseBlock, ToolUseError
from .utils import create_client, format_messages, format_tools, ToolCallAggregator
from .hooks import (
    PreToolUseEvent,
    PostToolUseEvent,
    UserPromptSubmitEvent,
    HookEvent,
    HookDecision,
    HOOK_PRE_TOOL_USE,
    HOOK_POST_TOOL_USE,
    HOOK_USER_PROMPT_SUBMIT,
)

logger = logging.getLogger(__name__)


async def _run_hooks_standalone(
    hooks: dict[str, list] | None,
    hook_name: str,
    event: HookEvent
) -> HookDecision | None:
    """
    Run hooks for standalone query() function.
    Same logic as Client._run_hooks but without instance dependency.
    """
    if not hooks:
        return None

    handlers = hooks.get(hook_name, [])
    for handler in handlers:
        try:
            decision = await handler(event)
            if decision is not None:
                if decision.reason:
                    logger.info(f"Hook {hook_name} decision: {decision.reason}")
                return decision
        except Exception as e:
            logger.error(f"Hook {hook_name} failed: {e}")
            raise

    return None


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
    # Fire UserPromptSubmit hook with system prompt for context
    history_snapshot = [{"role": "system", "content": options.system_prompt}]
    event = UserPromptSubmitEvent(prompt=prompt, history=history_snapshot)
    decision = await _run_hooks_standalone(options.hooks, HOOK_USER_PROMPT_SUBMIT, event)

    if decision and not decision.continue_:
        error_msg = decision.reason or "Query blocked by hook"
        logger.warning(f"UserPromptSubmit hook blocked query: {error_msg}")
        raise RuntimeError(f"Query blocked by hook: {error_msg}")

    # Use modified prompt if provided
    final_prompt = decision.modified_prompt if (decision and decision.modified_prompt) else prompt

    client = create_client(options)
    messages = format_messages(options.system_prompt, final_prompt)
    aggregator = ToolCallAggregator()

    # Prepare API request parameters
    request_params: dict[str, Any] = {
        "model": options.model,
        "messages": messages,
        "max_tokens": options.max_tokens,
        "temperature": options.temperature,
        "stream": True,
    }

    # Add tools if configured
    if options.tools:
        request_params["tools"] = format_tools(options.tools)

    try:
        response = await client.chat.completions.create(**request_params)

        try:
            async for chunk in response:
                # Process text blocks immediately
                text_block = aggregator.process_chunk(chunk)
                if text_block:
                    yield AssistantMessage(content=[text_block])
        finally:
            # Finalize any pending tool calls
            tool_blocks = aggregator.finalize_tools()
            if tool_blocks:
                for block in tool_blocks:
                    # Fire PreToolUse hook for each tool block
                    if isinstance(block, ToolUseBlock):
                        event = PreToolUseEvent(
                            tool_name=block.name,
                            tool_input=block.input,
                            tool_use_id=block.id,
                            history=messages  # Include full request context (system + user prompt)
                        )
                        decision = await _run_hooks_standalone(options.hooks, HOOK_PRE_TOOL_USE, event)

                        if decision and not decision.continue_:
                            error_msg = decision.reason or "Tool use blocked by hook"
                            logger.warning(f"PreToolUse hook blocked {block.name}: {error_msg}")
                            yield AssistantMessage(content=[ToolUseError(error=error_msg, raw_data=str(block.input))])
                            continue

                        if decision and decision.modified_input is not None:
                            logger.info(f"PreToolUse hook modified {block.name} input")
                            block.input = decision.modified_input

                    yield AssistantMessage(content=[block])

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

        # Interrupt state management
        self._interrupted = False
        self._stream_task: asyncio.Task | None = None

        # Build tool registry with duplicate validation
        self._tool_registry: dict[str, Any] = {}  # Maps tool name -> Tool instance
        if options.tools:
            seen_names: set[str] = set()
            for tool in options.tools:
                if tool.name in seen_names:
                    raise ValueError(
                        f"Duplicate tool name '{tool.name}' detected. "
                        f"Each tool must have a unique name."
                    )
                seen_names.add(tool.name)
                self._tool_registry[tool.name] = tool

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the underlying OpenAI client."""
        await self.client.close()

    async def interrupt(self) -> None:
        """
        Interrupt the current operation and cancel any in-progress streaming.

        This method:
        - Cancels the current response stream
        - Stops auto-execution loop if running
        - Cleans up network resources
        - Leaves client in valid state for reuse

        Safe to call multiple times (idempotent).
        Safe to call when no operation is in progress (no-op).

        Examples:
            # Cancel current streaming response
            await client.query("Long task...")
            await asyncio.sleep(1)  # Let it start
            await client.interrupt()  # Cancel it

            # Use with timeout
            try:
                await asyncio.wait_for(
                    process_messages(client),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                await client.interrupt()
        """
        if self._interrupted:
            # Already interrupted or nothing running
            logger.debug("Interrupt called but already interrupted")
            return

        logger.info("Interrupting current operation")
        self._interrupted = True

        # Cancel the streaming task if it exists
        if self._stream_task and not self._stream_task.done():
            stream_task = self._stream_task
            current_task = asyncio.current_task()
            if stream_task is not current_task:
                stream_task.cancel()
                try:
                    await stream_task
                except asyncio.CancelledError:
                    pass  # Expected
            self._stream_task = None

        # Best-effort close on the iterator itself
        if self.response_stream and hasattr(self.response_stream, "aclose"):
            try:
                await self.response_stream.aclose()
            except Exception as exc:
                logger.warning(f"Error closing response stream: {exc}")

        self.response_stream = None

        # Clear aggregator
        self._aggregator = None

        logger.info("Operation interrupted successfully")

    async def query(self, prompt: str):
        """Send query and prepare to receive messages"""
        # Reset interrupt flag for new query
        self._interrupted = False

        # Fire UserPromptSubmit hook before formatting/sending
        event = UserPromptSubmitEvent(
            prompt=prompt,
            history=self.message_history.copy()
        )
        decision = await self._run_hooks(HOOK_USER_PROMPT_SUBMIT, event)

        # Handle hook decision
        if decision and not decision.continue_:
            # Hook blocked the query
            error_msg = decision.reason or "Query blocked by hook"
            logger.warning(f"UserPromptSubmit hook blocked query: {error_msg}")
            raise RuntimeError(f"Query blocked by hook: {error_msg}")

        # If hook modified prompt, use the modified version
        final_prompt = prompt
        if decision and decision.modified_prompt is not None:
            logger.info("UserPromptSubmit hook modified prompt")
            final_prompt = decision.modified_prompt

        messages = format_messages(
            self.options.system_prompt,
            final_prompt,
            self.message_history
        )

        user_entry = {
            "role": "user",
            "content": final_prompt
        }

        # Prepare API request parameters
        request_params: dict[str, Any] = {
            "model": self.options.model,
            "messages": messages,
            "max_tokens": self.options.max_tokens,
            "temperature": self.options.temperature,
            "stream": True,
        }

        # Add tools if configured
        if self.options.tools:
            request_params["tools"] = format_tools(self.options.tools)

        try:
            response_stream = await self.client.chat.completions.create(**request_params)
        except Exception:
            self.response_stream = None
            self._aggregator = None
            raise

        self.response_stream = response_stream
        # Initialize aggregator for this turn
        self._aggregator = ToolCallAggregator()
        # Add user message to history only after successful request setup
        self.message_history.append(user_entry)

    async def _continue_turn(self):
        """
        Continue conversation without new user message or hook re-firing.
        Used by auto-execution loop after tool results are added to history.

        Unlike query(), this:
        - Does NOT fire UserPromptSubmit hooks
        - Does NOT add new user message to history
        - Reuses existing message_history (which includes tool results)
        - Uses empty prompt (like manual query("")) to avoid repeating user's question
        """
        # Format messages with current history (includes tool results)
        # Use empty prompt - history already contains all context
        messages = format_messages(
            self.options.system_prompt,
            "",  # Empty prompt for continuation (mirrors manual query("") pattern)
            self.message_history
        )

        # Prepare API request parameters
        request_params: dict[str, Any] = {
            "model": self.options.model,
            "messages": messages,
            "max_tokens": self.options.max_tokens,
            "temperature": self.options.temperature,
            "stream": True,
        }

        # Add tools if configured
        if self.options.tools:
            request_params["tools"] = format_tools(self.options.tools)

        try:
            response_stream = await self.client.chat.completions.create(**request_params)
        except Exception:
            self.response_stream = None
            self._aggregator = None
            raise

        self.response_stream = response_stream
        # Initialize aggregator for this continuation
        self._aggregator = ToolCallAggregator()
        # Note: We do NOT add a user message - history already has tool results

    async def _auto_execute_loop(self) -> AsyncGenerator[TextBlock | ToolUseBlock | ToolUseError, None]:
        """
        Automatic tool execution loop.
        Executes tools automatically and continues conversation until text-only response
        or max_tool_iterations reached.

        This method:
        1. Calls _receive_once() to get all blocks (fully consumes stream before execution)
        2. Yields all blocks to caller
        3. If text-only, updates history and returns
        4. If tool blocks present:
           - Executes each tool automatically
           - Adds results to history (fires PostToolUse hooks)
           - Continues conversation via _continue_turn()
        5. Repeats until text-only or max_tool_iterations reached
        """
        for iteration in range(self.options.max_tool_iterations):
            # Check at loop start
            if self._interrupted:
                logger.info(f"Auto-execution interrupted at iteration {iteration}")
                return

            # Get all blocks from this turn (ensures stream fully consumed before tool execution)
            try:
                assistant_blocks = await self._receive_once()
            except Exception as e:
                if self._interrupted:
                    logger.info("Auto-execution interrupted during streaming")
                    return
                raise

            # Check for interrupt after receiving blocks
            if self._interrupted:
                logger.info("Auto-execution interrupted after receiving blocks")
                return

            # Yield all blocks to caller
            for block in assistant_blocks:
                if self._interrupted:
                    logger.info("Auto-execution interrupted during block yielding")
                    return
                yield block

            # Check if we have any tool calls
            tool_blocks = [b for b in assistant_blocks if isinstance(b, ToolUseBlock)]

            # Add assistant response to history BEFORE checking if text-only
            # (needed so add_tool_result can find the tool_call_id)
            history_entry = self._format_history_entry(assistant_blocks)
            self.message_history.append(history_entry)

            if not tool_blocks:
                # Text-only response - we're done
                self.turn_count += 1

                if self.turn_count >= self.options.max_turns:
                    logger.info(f"Reached max_turns ({self.options.max_turns})")

                # Reset streaming state
                self.response_stream = None
                self._aggregator = None
                return

            # We have tool calls - execute them automatically
            for tool_block in tool_blocks:
                # Check BEFORE starting tool
                if self._interrupted:
                    logger.info("Auto-execution interrupted before tool execution")
                    return

                tool_name = tool_block.name
                tool_input = tool_block.input
                tool_call_id = tool_block.id

                # Look up tool in registry
                tool = self._tool_registry.get(tool_name)
                if not tool:
                    # Unknown tool - add error to history and yield ToolUseError
                    error_payload = {
                        "error": f"Tool '{tool_name}' not found in registry",
                        "tool": tool_name
                    }
                    logger.warning(f"Unknown tool '{tool_name}' requested")

                    # Yield ToolUseError for monitoring
                    yield ToolUseError(
                        error=f"Tool '{tool_name}' not found in registry",
                        raw_data=str(tool_input)
                    )

                    # Add to history so model sees the error
                    await self.add_tool_result(
                        tool_call_id=tool_call_id,
                        content=error_payload,  # Dict, not JSON string
                        name=tool_name
                    )
                    continue

                # Execute tool in try/except to catch CancelledError
                try:
                    # Wrap in create_task so cancellation propagates
                    tool_task = asyncio.create_task(tool.execute(tool_input))
                    result = await tool_task

                    # Check AFTER tool execution
                    if self._interrupted:
                        logger.info(f"Auto-execution interrupted after {tool_name}")
                        return

                    logger.debug(f"Tool {tool_name} executed successfully")
                    await self.add_tool_result(
                        tool_call_id=tool_call_id,
                        content=result,
                        name=tool_name
                    )

                except asyncio.CancelledError:
                    logger.info(f"Tool {tool_name} cancelled")
                    if self._interrupted:
                        return
                    raise

                except Exception as e:
                    # Tool execution failed - add error to history and yield ToolUseError
                    error_payload = {
                        "error": f"Tool execution failed: {str(e)}",
                        "tool": tool_name
                    }
                    logger.error(f"Tool {tool_name} execution failed: {e}")

                    # Yield ToolUseError for monitoring
                    yield ToolUseError(
                        error=f"Tool execution failed: {str(e)}",
                        raw_data=str(tool_input)
                    )

                    # Add to history so model sees the error
                    await self.add_tool_result(
                        tool_call_id=tool_call_id,
                        content=error_payload,  # Dict, not JSON string
                        name=tool_name
                    )

                    # Check for interrupt after error handling
                    if self._interrupted:
                        return

            # Check BEFORE continuing conversation
            if self._interrupted:
                logger.info("Auto-execution interrupted before _continue_turn")
                return

            # Continue conversation (get model's response to tool results)
            # Note: assistant response with tool calls already added to history above
            await self._continue_turn()
            # Loop again to process next response

        # Hit max_tool_iterations - get final response and return
        logger.warning(
            f"Reached max_tool_iterations ({self.options.max_tool_iterations}). "
            f"Stopping auto-execution."
        )

        # Get final response
        assistant_blocks = await self._receive_once()
        for block in assistant_blocks:
            yield block

        # Update history and state
        history_entry = self._format_history_entry(assistant_blocks)
        self.message_history.append(history_entry)
        self.turn_count += 1

        if self.turn_count >= self.options.max_turns:
            logger.info(f"Reached max_turns ({self.options.max_turns})")

        # Reset streaming state
        self.response_stream = None
        self._aggregator = None

    async def _receive_once(self) -> list[TextBlock | ToolUseBlock | ToolUseError]:
        """
        Internal helper to consume one complete response from the model.
        Returns list of blocks after fully consuming stream and running hooks.
        Does NOT update history or turn_count - caller's responsibility.

        If interrupted mid-stream, flushes partial text blocks to history
        and skips tool finalization.
        """
        if not self.response_stream:
            raise RuntimeError("Call query() or _continue_turn() first")
        if not self._aggregator:
            raise RuntimeError("Aggregator not initialized")

        assistant_blocks: list[TextBlock | ToolUseBlock | ToolUseError] = []

        current_task = asyncio.current_task()
        previous_task = self._stream_task
        self._stream_task = current_task

        try:
            # Stream text blocks - check for interruption
            try:
                async for chunk in self.response_stream:
                    if self._interrupted:
                        logger.info("Streaming interrupted during chunk processing")
                        break

                    text_block = self._aggregator.process_chunk(chunk)
                    if text_block:
                        assistant_blocks.append(text_block)
            except asyncio.CancelledError:
                logger.info("Streaming cancelled")
                self._interrupted = True
                raise
            except Exception as exc:
                # Stream closed or other error
                if self._interrupted:
                    logger.info(f"Stream closed during interrupt: {exc}")
                else:
                    raise

            # If interrupted mid-stream, flush partial text and clear pending tool buffers
            if self._interrupted:
                if assistant_blocks:
                    # Flush partial text blocks to history (no tool data)
                    text_parts = [block.text for block in assistant_blocks if isinstance(block, TextBlock)]
                    if text_parts:
                        history_entry = {"role": "assistant", "content": "".join(text_parts)}
                        self.message_history.append(history_entry)
                        logger.info("Flushed partial output to history due to interrupt")

                # Clear pending tool buffers - they won't appear in history
                if self._aggregator and hasattr(self._aggregator, 'pending_tools'):
                    self._aggregator.pending_tools.clear()

                return assistant_blocks

            # Normal path: finalize tools when not interrupted
            tool_blocks = self._aggregator.finalize_tools()
            if tool_blocks:
                for tool_block in tool_blocks:
                    # Fire PreToolUse hook before adding tool block
                    if isinstance(tool_block, ToolUseBlock):
                        event = PreToolUseEvent(
                            tool_name=tool_block.name,
                            tool_input=tool_block.input,
                            tool_use_id=tool_block.id,
                            history=self.message_history.copy()
                        )
                        decision = await self._run_hooks(HOOK_PRE_TOOL_USE, event)

                        # Handle hook decision
                        if decision and not decision.continue_:
                            # Hook blocked the tool - add error instead
                            error_msg = decision.reason or "Tool use blocked by hook"
                            logger.warning(f"PreToolUse hook blocked {tool_block.name}: {error_msg}")
                            error_block = ToolUseError(error=error_msg, raw_data=str(tool_block.input))
                            assistant_blocks.append(error_block)
                            continue

                        # If hook modified input, update the block
                        if decision and decision.modified_input is not None:
                            logger.info(f"PreToolUse hook modified {tool_block.name} input")
                            tool_block.input = decision.modified_input

                    # Add to blocks list
                    assistant_blocks.append(tool_block)

            return assistant_blocks

        finally:
            # Restore previous stream task handle
            if self._stream_task is current_task:
                self._stream_task = previous_task

    async def receive_messages(self) -> AsyncGenerator[TextBlock | ToolUseBlock | ToolUseError, None]:
        """
        Stream individual blocks from response.

        Dispatches to either:
        - _auto_execute_loop() if auto_execute_tools=True (automatic tool execution)
        - Manual mode if auto_execute_tools=False (caller handles tool execution)

        In manual mode, caller must:
        1. Receive ToolUseBlock instances
        2. Execute tools themselves
        3. Call add_tool_result() for each tool
        4. Call query("") to continue conversation

        In auto mode, tools are executed automatically and the loop continues
        until a text-only response is received.
        """
        if self._interrupted and not self.response_stream:
            logger.info("No active response stream (already interrupted); nothing to receive")
            return

        try:
            if self.options.auto_execute_tools:
                # Auto mode - execute tools automatically until text-only response
                async for block in self._auto_execute_loop():
                    if self._interrupted:
                        logger.info("Auto-execution loop interrupted")
                        return
                    yield block
            else:
                # Manual mode - caller handles tool execution
                # Get all blocks from the response
                assistant_blocks = await self._receive_once()

                # Check for interrupt after receiving blocks
                if self._interrupted:
                    logger.info("Message receive interrupted")
                    return

                # Add assistant response to history with proper structure
                history_entry = self._format_history_entry(assistant_blocks)
                self.message_history.append(history_entry)

                # Yield each block to caller
                for block in assistant_blocks:
                    if self._interrupted:
                        logger.info("Message receive interrupted during yielding")
                        return
                    yield block

                self.turn_count += 1

                # Check max turns
                if self.turn_count >= self.options.max_turns:
                    logger.info(f"Reached max_turns ({self.options.max_turns})")

                # Reset streaming state
                self.response_stream = None
                self._aggregator = None
        except asyncio.CancelledError:
            logger.info("Streaming cancelled via task cancellation")
            self._interrupted = True
            raise

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

    async def add_tool_result(
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

        Note: This method is now async to support PostToolUse hooks.
        """
        if not tool_call_id:
            raise ValueError("tool_call_id cannot be empty")

        if not self._tool_call_known(tool_call_id):
            raise ValueError(f"Unknown tool_call_id: {tool_call_id}")

        # Find the tool name and input from history
        tool_name, tool_input = self._get_tool_info(tool_call_id)

        if isinstance(content, str):
            message_content = content
        elif isinstance(content, (dict, list)):
            message_content = json.dumps(content)
        else:
            raise TypeError("content must be a str, dict, or list")

        # Fire PostToolUse hook before adding result to history
        event = PostToolUseEvent(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=tool_call_id,
            tool_result=content,
            history=self.message_history.copy()
        )
        decision = await self._run_hooks(HOOK_POST_TOOL_USE, event)

        # PostToolUse hooks can observe but not block
        # (result already executed, just logging/monitoring)
        if decision and decision.reason:
            logger.info(f"PostToolUse hook for {tool_name}: {decision.reason}")

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

    def _get_tool_info(self, tool_call_id: str) -> tuple[str, dict[str, Any]]:
        """Get tool name and input for a given tool call ID."""
        for message in reversed(self.message_history):
            if message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls", []):
                if tool_call.get("id") == tool_call_id:
                    function = tool_call.get("function", {})
                    name = function.get("name", "unknown")
                    # Parse arguments JSON string
                    try:
                        args_str = function.get("arguments", "{}")
                        tool_input = json.loads(args_str)
                    except json.JSONDecodeError:
                        tool_input = {}
                    return name, tool_input
        # Fallback if not found (shouldn't happen due to _tool_call_known check)
        return "unknown", {}

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

    async def _run_hooks(
        self,
        hook_name: str,
        event: HookEvent
    ) -> HookDecision | None:
        """
        Run hooks for a given event. Returns first non-None decision.

        Hooks run sequentially in order registered. If any hook returns a
        HookDecision, that decision is returned immediately (short-circuit).
        If a hook raises an exception, it propagates to the caller.

        Args:
            hook_name: Hook event name (e.g., HOOK_PRE_TOOL_USE)
            event: Event data for the hook

        Returns:
            HookDecision if any hook returned one, otherwise None
        """
        if not self.options.hooks:
            return None

        handlers = self.options.hooks.get(hook_name, [])
        for handler in handlers:
            try:
                decision = await handler(event)
                if decision is not None:
                    # First non-None decision wins
                    if decision.reason:
                        logger.info(f"Hook {hook_name} decision: {decision.reason}")
                    return decision
            except Exception as e:
                logger.error(f"Hook {hook_name} failed: {e}")
                raise

        return None
