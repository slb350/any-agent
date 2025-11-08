"""
Open Agent SDK - Claude Agent SDK-style API for local LLMs

This is the main package entry point that exports the public API for the Open Agent SDK.
The SDK provides a Claude-inspired interface for building AI agents that run on local
or self-hosted LLMs via OpenAI-compatible endpoints (LM Studio, Ollama, llama.cpp, etc.).

Public API Structure:
- query(): Simple single-turn query function for quick interactions
- Client: Multi-turn conversation client with message history management
- AgentOptions: Configuration object for customizing agent behavior
- Message types (TextBlock, ToolUseBlock, etc.): Data structures for agent responses
- @tool decorator & Tool class: Define callable tools for the agent
- Hooks system: Lifecycle hooks for monitoring and controlling agent behavior
"""

# Core client functionality for interacting with LLMs
# - query(): Simplified single-turn API for quick queries without state management
# - Client: Full-featured client class for multi-turn conversations with history
from .client import query, Client

# Type definitions for agent configuration and message structures
# - AgentOptions: Configuration dataclass with validation (model, base_url, hooks, etc.)
# - TextBlock: Represents text content in agent responses
# - ToolUseBlock: Represents a tool call request from the agent
# - ToolUseError: Represents malformed/unparseable tool calls
# - ToolResultBlock: Represents the result of tool execution to send back to agent
# - AssistantMessage: Container for all content blocks in a single agent response
from .types import AgentOptions, TextBlock, ToolUseBlock, ToolUseError, ToolResultBlock, AssistantMessage

# Tool system for defining callable functions the agent can use
# - @tool: Decorator to convert Python functions into agent-callable tools
# - Tool: Dataclass representing a tool definition with schema and execution logic
from .tools import tool, Tool

# Lifecycle hooks system for monitoring and controlling agent behavior
# Hook Events (data classes passed to hook handlers):
# - PreToolUseEvent: Fired before a tool is executed (can modify/block execution)
# - PostToolUseEvent: Fired after a tool executes (can modify/block results)
# - UserPromptSubmitEvent: Fired before processing user input (can modify/sanitize input)
# - HookEvent: Base type for all hook events
#
# Hook Control:
# - HookDecision: Return type for hooks to control execution (continue/block/modify)
# - HookHandler: Type alias for hook function signature
#
# Hook Constants (used to register hooks in AgentOptions):
# - HOOK_PRE_TOOL_USE: String constant for pre-tool-use hook type
# - HOOK_POST_TOOL_USE: String constant for post-tool-use hook type
# - HOOK_USER_PROMPT_SUBMIT: String constant for user-prompt-submit hook type
from .hooks import (
    PreToolUseEvent,
    PostToolUseEvent,
    UserPromptSubmitEvent,
    HookEvent,
    HookDecision,
    HookHandler,
    HOOK_PRE_TOOL_USE,
    HOOK_POST_TOOL_USE,
    HOOK_USER_PROMPT_SUBMIT,
)

# Explicit public API definition for type checkers and documentation tools
# This list controls what gets exported when users do "from open_agent import *"
# Note: While star imports are generally discouraged, this ensures only intended
# public APIs are exposed and helps IDEs with autocomplete
__all__ = [
    # Core client API
    "query",  # Single-turn query function
    "Client",  # Multi-turn conversation client
    "AgentOptions",  # Configuration object
    # Message content types
    "TextBlock",  # Text content from agent
    "ToolUseBlock",  # Tool call request from agent
    "ToolUseError",  # Malformed tool call error
    "ToolResultBlock",  # Tool execution result
    "AssistantMessage",  # Complete agent response wrapper
    # Tool system
    "tool",  # Decorator for defining tools
    "Tool",  # Tool definition class
    # Hooks system - Events
    "PreToolUseEvent",  # Pre-execution hook event
    "PostToolUseEvent",  # Post-execution hook event
    "UserPromptSubmitEvent",  # User input hook event
    "HookEvent",  # Base hook event type
    # Hooks system - Control
    "HookDecision",  # Hook decision return type
    "HookHandler",  # Hook function type alias
    # Hooks system - Constants
    "HOOK_PRE_TOOL_USE",  # Pre-tool hook constant
    "HOOK_POST_TOOL_USE",  # Post-tool hook constant
    "HOOK_USER_PROMPT_SUBMIT",  # User prompt hook constant
]
