"""
Hooks system for intercepting and controlling agent execution.

This module provides a flexible lifecycle hooks system for monitoring, controlling,
and modifying agent behavior at critical execution points. The hooks system enables:
- Security gates (block dangerous operations)
- Audit logging (track all tool calls and inputs)
- Input sanitization (modify user prompts before processing)
- Result filtering (modify tool outputs before returning to LLM)

Design Philosophy:
- Simple, Pythonic async functions (not complex callback classes)
- Local-first (runs in your process, no CLI/MCP protocol overhead)
- Opt-in (no hooks run unless explicitly configured)
- Composable (chain multiple hooks for same event)
- Type-safe (proper event types with IDE autocomplete)

Architecture:
The hooks system follows an event-driven pattern with three lifecycle points:

1. HOOK_USER_PROMPT_SUBMIT: Fired before processing user input
   - Use case: Sanitize input, block inappropriate prompts
   - Can modify or block the prompt before it reaches the LLM

2. HOOK_PRE_TOOL_USE: Fired before executing a tool
   - Use case: Security gates, parameter validation, audit logging
   - Can modify tool inputs or block execution entirely

3. HOOK_POST_TOOL_USE: Fired after a tool executes
   - Use case: Result sanitization, error handling, logging
   - Can modify the result before it's sent back to the LLM

Hook Execution Flow:
1. Event occurs (user prompt, tool call, etc.)
2. SDK creates event object with context (tool name, inputs, history)
3. SDK calls all registered handlers for that event in order
4. Handlers return HookDecision or None
   - None: Continue normally
   - HookDecision(continue_=False): Block the operation
   - HookDecision(modified_input={...}): Modify parameters
5. SDK processes decisions and proceeds accordingly

Example Use Cases:

Security Gate:
    ```python
    async def security_gate(event: PreToolUseEvent) -> HookDecision | None:
        # Block file system operations outside safe directory
        if event.tool_name == "read_file":
            path = event.tool_input.get("path", "")
            if not path.startswith("/safe/"):
                return HookDecision(
                    continue_=False,
                    reason="File access outside /safe/ directory blocked"
                )
        return None  # Allow other operations
    ```

Audit Logger:
    ```python
    async def audit_logger(event: PreToolUseEvent) -> HookDecision | None:
        # Log all tool calls for compliance
        print(f"[AUDIT] Tool: {event.tool_name}, Input: {event.tool_input}")
        return None  # Don't block, just log
    ```

Input Sanitization:
    ```python
    async def sanitize_prompt(event: UserPromptSubmitEvent) -> HookDecision | None:
        # Remove PII from user prompts
        sanitized = remove_email_addresses(event.prompt)
        if sanitized != event.prompt:
            return HookDecision(
                modified_prompt=sanitized,
                reason="Removed email addresses"
            )
        return None
    ```

Chaining Multiple Hooks:
    ```python
    options = AgentOptions(
        system_prompt="...",
        model="...",
        base_url="...",
        hooks={
            HOOK_PRE_TOOL_USE: [
                security_gate,      # Check security first
                audit_logger,       # Log after security check passes
                parameter_validator # Validate params last
            ],
        }
    )
    ```

Hook Execution Order:
- Hooks execute in registration order (list order in AgentOptions.hooks)
- First hook that returns HookDecision stops the chain (subsequent hooks don't run)
- Return None to pass control to the next hook
- If all hooks return None, the operation proceeds normally
- Example: If security_gate returns HookDecision(continue_=False), neither
  audit_logger nor parameter_validator will run

Error Handling:
- If a hook raises an exception, execution is aborted immediately
- Use exceptions for unrecoverable errors (validation failures, etc.)
- Use HookDecision(continue_=False) for expected blocking cases

Thread Safety:
- Hook handlers are async functions called sequentially (not parallel)
- Handlers receive a snapshot of history (read-only, no mutations)
- Safe to use in async/await contexts
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class PreToolUseEvent:
    """
    Event object passed to pre-tool-use hooks before a tool is executed.

    This event is fired immediately before the SDK calls a tool's execute() method.
    Handlers can inspect the tool call details and decide whether to:
    - Allow execution to proceed (return None or HookDecision(continue_=True))
    - Block execution (return HookDecision(continue_=False))
    - Modify tool inputs (return HookDecision(modified_input={...}))

    Timing:
        Fired AFTER the LLM generates a ToolUseBlock but BEFORE Tool.execute() is called.
        In auto-execution mode, this runs automatically. In manual mode, this runs when
        the user explicitly calls Client.add_tool_result() or auto-execute is enabled.

    Attributes:
        tool_name (str): Name of the tool about to be executed. Matches a tool
            defined in AgentOptions.tools. Example: "get_weather", "read_file"

        tool_input (dict[str, Any]): Dictionary of arguments the LLM wants to pass
            to the tool. Keys are parameter names from the tool's schema.
            Example: {"city": "San Francisco", "units": "celsius"}

        tool_use_id (str): Unique identifier for this specific tool call, generated
            by the LLM. Format is provider-dependent (e.g., "call_abc123").
            Use this to correlate with ToolResultBlock if needed.

        history (list[dict[str, Any]]): Read-only snapshot of the conversation
            history at the time of this event. Each dict is a message in OpenAI
            format ({"role": "user"/"assistant", "content": ...}).
            Use this for context-aware security decisions.

    Example Handler:
        ```python
        async def block_dangerous_tools(event: PreToolUseEvent) -> HookDecision | None:
            # Block file deletion
            if event.tool_name == "delete_file":
                return HookDecision(
                    continue_=False,
                    reason="File deletion blocked by security policy"
                )

            # Sanitize file paths
            if event.tool_name == "read_file":
                path = event.tool_input.get("path", "")
                if not path.startswith("/safe/"):
                    return HookDecision(
                        modified_input={"path": "/safe/default.txt"},
                        reason="Redirected to safe directory"
                    )

            # Allow other tools
            return None
        ```

    Note:
        The history snapshot is immutable - modifications won't affect the actual
        conversation history. To modify state, use the HookDecision return value.
    """
    tool_name: str  # Name of tool to execute (e.g., "get_weather")
    tool_input: dict[str, Any]  # Arguments dict (e.g., {"city": "NYC"})
    tool_use_id: str  # Unique tool call ID (e.g., "call_abc123")
    history: list[dict[str, Any]]  # Read-only conversation history snapshot


@dataclass
class PostToolUseEvent:
    """
    Event object passed to post-tool-use hooks after a tool has executed.

    This event is fired immediately after a tool's execute() method completes,
    whether it succeeded or raised an exception. Handlers can inspect the result
    and decide whether to:
    - Allow the result to pass through (return None)
    - Modify the result before sending to LLM (return HookDecision(modified_input={...}))
    - Block the result (return HookDecision(continue_=False))

    Timing:
        Fired AFTER Tool.execute() completes but BEFORE the result is converted
        to a ToolResultBlock and sent back to the LLM. This gives you a chance to
        sanitize outputs, handle errors, or add logging.

    Attributes:
        tool_name (str): Name of the tool that was executed.
            Example: "get_weather", "search_database"

        tool_input (dict[str, Any]): The input parameters that were used for execution.
            This is the FINAL input after any PreToolUseEvent modifications.
            Example: {"city": "San Francisco", "units": "celsius"}

        tool_use_id (str): Unique identifier for this tool call, matching the ID
            from the corresponding PreToolUseEvent.

        tool_result (Any): The raw result returned by Tool.execute(). Can be:
            - Any Python object if execution succeeded
            - Exception instance if execution failed (checked in is_error flag)
            Note: This is NOT yet converted to ToolResultBlock format.

        history (list[dict[str, Any]]): Read-only snapshot of conversation history
            at the time of this event (same format as PreToolUseEvent).

    Example Handler:
        ```python
        async def sanitize_results(event: PostToolUseEvent) -> HookDecision | None:
            # Remove sensitive data from API responses
            if event.tool_name == "get_user_data":
                result = event.tool_result
                if isinstance(result, dict):
                    sanitized = {k: v for k, v in result.items()
                                 if k not in ["ssn", "password"]}
                    return HookDecision(
                        modified_input=sanitized,  # Note: reuses modified_input field
                        reason="Removed PII fields"
                    )

            # Log errors
            if isinstance(event.tool_result, Exception):
                print(f"Tool {event.tool_name} failed: {event.tool_result}")

            return None
        ```

    Note:
        - modified_input in HookDecision is used to modify the tool_result
        - The is_error flag in ToolResultBlock is set automatically based on
          whether tool_result is an Exception instance
    """
    tool_name: str  # Name of executed tool
    tool_input: dict[str, Any]  # Input parameters that were used
    tool_use_id: str  # Unique tool call ID
    tool_result: Any  # Raw result from Tool.execute() (any type or Exception)
    history: list[dict[str, Any]]  # Read-only conversation history snapshot


@dataclass
class UserPromptSubmitEvent:
    """
    Event object passed to user-prompt-submit hooks before processing user input.

    This event is fired when a user submits a new prompt (via Client.query() or
    query() function), BEFORE the prompt is sent to the LLM. Handlers can:
    - Sanitize or modify the prompt (return HookDecision(modified_prompt="..."))
    - Block inappropriate prompts (return HookDecision(continue_=False))
    - Log user inputs for audit trails (return None after logging)

    Timing:
        Fired BEFORE the user message is added to history and BEFORE the API
        call to the LLM is made. This is the earliest interception point in
        the request lifecycle.

    Attributes:
        prompt (str): The user's input text that is about to be sent to the LLM.
            This is the raw string from Client.query(prompt).

        history (list[dict[str, Any]]): Read-only snapshot of the conversation
            history BEFORE this prompt is added. For the first turn, this will
            be empty. Each dict is in OpenAI message format.

    Example Handler:
        ```python
        async def content_filter(event: UserPromptSubmitEvent) -> HookDecision | None:
            # Block prompts with banned words
            banned_words = ["spam", "harmful"]
            if any(word in event.prompt.lower() for word in banned_words):
                return HookDecision(
                    continue_=False,
                    reason="Prompt contains banned content"
                )

            # Remove PII (email addresses)
            import re
            sanitized = re.sub(r'\b[\w.-]+@[\w.-]+\.\w+\b', r'[EMAIL]', event.prompt)
            if sanitized != event.prompt:
                return HookDecision(
                    modified_prompt=sanitized,
                    reason="Removed email addresses"
                )

            return None  # Allow prompt to proceed
        ```

    Use Cases:
        - Content filtering (block inappropriate requests)
        - PII removal (strip emails, phone numbers, etc.)
        - Prompt augmentation (add context or instructions)
        - Rate limiting (check user quota before processing)
        - Audit logging (record all user inputs for compliance)

    Note:
        If multiple handlers modify the prompt, modifications are applied sequentially
        in the order handlers are registered in AgentOptions.hooks[HOOK_USER_PROMPT_SUBMIT].
    """
    prompt: str  # User's input text
    history: list[dict[str, Any]]  # Read-only conversation history (before this prompt)


# Union type for all hook event types
# This allows type-safe pattern matching in handlers that handle multiple event types
# Example:
#   async def universal_logger(event: HookEvent) -> HookDecision | None:
#       match event:
#           case PreToolUseEvent():
#               print(f"Tool call: {event.tool_name}")
#           case PostToolUseEvent():
#               print(f"Tool result: {event.tool_result}")
#           case UserPromptSubmitEvent():
#               print(f"User prompt: {event.prompt}")
#       return None
HookEvent = PreToolUseEvent | PostToolUseEvent | UserPromptSubmitEvent


@dataclass
class HookDecision:
    """
    Decision object returned by hook handlers to control execution flow.

    Hook handlers return either None (continue normally) or a HookDecision instance
    to modify behavior. This dataclass encodes the handler's decision about whether
    and how to proceed with the operation.

    Attributes:
        continue_ (bool): Whether to proceed with the operation. If False, the
            operation is aborted and (in auto-execute mode) an error is sent to
            the LLM explaining the block. Default: True.
            Note: Named continue_ (with underscore) because "continue" is a keyword.

        modified_input (dict[str, Any] | None): For PreToolUseEvent hooks, this
            replaces the tool_input that will be passed to Tool.execute().
            For PostToolUseEvent hooks, this replaces the tool_result that will
            be sent back to the LLM.
            If None, no modification occurs. Default: None.

        modified_prompt (str | None): For UserPromptSubmitEvent hooks, this replaces
            the user's prompt that will be sent to the LLM. If None, no modification
            occurs. Default: None.

        reason (str | None): Optional human-readable explanation for logging/debugging.
            When continue_=False, this reason may be sent to the LLM (in auto-execute
            mode) so it understands why the operation was blocked.
            Example: "File access outside /safe/ directory blocked". Default: None.

    Return Semantics:
        - None: Equivalent to HookDecision(continue_=True) - continue normally
        - HookDecision(): Continue with defaults (continue_=True, no modifications)
        - HookDecision(continue_=False): Block the operation entirely
        - HookDecision(modified_input={...}): Modify parameters, then continue
        - HookDecision(continue_=False, reason="..."): Block with explanation

    Examples:
        Block execution:
            ```python
            return HookDecision(
                continue_=False,
                reason="Operation not allowed by security policy"
            )
            ```

        Modify tool input:
            ```python
            return HookDecision(
                modified_input={"path": "/safe/path", "mode": "read"},
                reason="Sanitized file path to safe directory"
            )
            ```

        Modify user prompt:
            ```python
            return HookDecision(
                modified_prompt="[Sanitized] " + cleaned_prompt,
                reason="Removed PII from prompt"
            )
            ```

        Continue normally (these are equivalent):
            ```python
            return None
            return HookDecision()
            return HookDecision(continue_=True)
            ```

    Implementation Notes:
        - Only one modification field should be set at a time:
          * modified_input for Pre/PostToolUseEvent
          * modified_prompt for UserPromptSubmitEvent
        - Setting the wrong modification field for an event type is ignored
        - Multiple handlers can chain modifications (output of handler N becomes
          input to handler N+1)
    """
    continue_: bool = True  # Whether to proceed with operation (False = block)
    modified_input: dict[str, Any] | None = None  # Modified tool input/result
    modified_prompt: str | None = None  # Modified user prompt
    reason: str | None = None  # Optional explanation for logging/debugging


# Type alias for hook handler functions
# All hook handlers must match this signature:
# - Accept a single HookEvent parameter (PreToolUseEvent | PostToolUseEvent | UserPromptSubmitEvent)
# - Return either None (continue normally) or HookDecision (control execution)
# - Be async (return Awaitable)
#
# Return value semantics:
#   - None: Continue normally with no modifications
#   - HookDecision: Control execution flow (continue/block/modify)
#
# Error handling:
#   - If handler raises an exception, execution is aborted immediately
#   - Use exceptions for unrecoverable errors
#   - Use HookDecision(continue_=False) for expected blocking conditions
#
# Example:
#   async def my_hook(event: PreToolUseEvent) -> HookDecision | None:
#       if event.tool_name == "dangerous":
#           raise ValueError("Dangerous tool not allowed")  # Abort with error
#       if event.tool_name == "risky":
#           return HookDecision(continue_=False)  # Graceful block
#       return None  # Allow
HookHandler = Callable[[HookEvent], Awaitable[HookDecision | None]]


# Hook event name constants
# These string constants are used as keys in AgentOptions.hooks dictionary to
# register handlers for specific lifecycle events. Using constants prevents typos
# and enables IDE autocomplete.
#
# Usage:
#   options = AgentOptions(
#       ...,
#       hooks={
#           HOOK_PRE_TOOL_USE: [security_gate, audit_logger],
#           HOOK_POST_TOOL_USE: [result_sanitizer],
#           HOOK_USER_PROMPT_SUBMIT: [content_filter],
#       }
#   )

HOOK_PRE_TOOL_USE = "pre_tool_use"  # Fired before Tool.execute() is called
HOOK_POST_TOOL_USE = "post_tool_use"  # Fired after Tool.execute() completes
HOOK_USER_PROMPT_SUBMIT = "user_prompt_submit"  # Fired before user prompt is sent to LLM
