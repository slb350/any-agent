"""Hooks system for intercepting and controlling agent execution.

This module provides a simple hooks system for monitoring and controlling agent
behavior at key lifecycle points. It follows a Pythonic, local-first design without
the complexity of CLI control protocols.

Usage:
    ```python
    from open_agent import Client, AgentOptions
    from open_agent.hooks import PreToolUseEvent, HookDecision

    async def approve_tool(event: PreToolUseEvent) -> HookDecision | None:
        if event.tool_name == "delete_file":
            return HookDecision(continue_=False, reason="Dangerous operation blocked")
        return None  # Continue normally

    options = AgentOptions(
        system_prompt="...",
        model="...",
        base_url="...",
        hooks={
            "pre_tool_use": [approve_tool],
        }
    )

    async with Client(options) as client:
        await client.query("Delete all files")
        # approve_tool will block the deletion
    ```
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class PreToolUseEvent:
    """Event fired before tool execution.

    Attributes:
        tool_name: Name of the tool about to be executed
        tool_input: Input parameters for the tool
        tool_use_id: Unique identifier for this tool use
        history: Snapshot of conversation history (read-only)
    """

    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str
    history: list[dict[str, Any]]


@dataclass
class PostToolUseEvent:
    """Event fired after tool execution.

    Attributes:
        tool_name: Name of the tool that was executed
        tool_input: Input parameters that were used
        tool_use_id: Unique identifier for this tool use
        tool_result: Result returned by the tool (success or error)
        history: Snapshot of conversation history (read-only)
    """

    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str
    tool_result: Any
    history: list[dict[str, Any]]


@dataclass
class UserPromptSubmitEvent:
    """Event fired before processing user input.

    Attributes:
        prompt: The user's input prompt
        history: Snapshot of conversation history (read-only)
    """

    prompt: str
    history: list[dict[str, Any]]


# Union type for all hook events
HookEvent = PreToolUseEvent | PostToolUseEvent | UserPromptSubmitEvent


@dataclass
class HookDecision:
    """Decision returned by hook handler to control execution.

    Attributes:
        continue_: Whether to continue execution (default: True)
        modified_input: For PreToolUse - modified tool input (overrides original)
        modified_prompt: For UserPromptSubmit - modified prompt (overrides original)
        reason: Optional explanation for logging/debugging

    Examples:
        Block execution:
            ```python
            return HookDecision(continue_=False, reason="Operation not allowed")
            ```

        Modify tool input:
            ```python
            return HookDecision(
                modified_input={"path": "/safe/path"},
                reason="Redirected to safe directory"
            )
            ```

        Continue normally (equivalent to returning None):
            ```python
            return HookDecision()
            ```
    """

    continue_: bool = True
    modified_input: dict[str, Any] | None = None
    modified_prompt: str | None = None
    reason: str | None = None


# Hook handler callable type
# Returns:
#   - None: Continue normally with no modifications
#   - HookDecision: Control execution (continue/skip/modify)
# Raises:
#   - Exception: Abort execution entirely
HookHandler = Callable[[HookEvent], Awaitable[HookDecision | None]]


# Hook event names (for AgentOptions.hooks dict keys)
HOOK_PRE_TOOL_USE = "pre_tool_use"
HOOK_POST_TOOL_USE = "post_tool_use"
HOOK_USER_PROMPT_SUBMIT = "user_prompt_submit"
