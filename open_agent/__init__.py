"""Open Agent SDK - Claude Agent SDK-style API for local LLMs"""
from .client import query, Client
from .types import AgentOptions, TextBlock, ToolUseBlock, ToolUseError, ToolResultBlock, AssistantMessage
from .tools import tool, Tool
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

__all__ = [
    "query",
    "Client",
    "AgentOptions",
    "TextBlock",
    "ToolUseBlock",
    "ToolUseError",
    "ToolResultBlock",
    "AssistantMessage",
    "tool",
    "Tool",
    # Hooks
    "PreToolUseEvent",
    "PostToolUseEvent",
    "UserPromptSubmitEvent",
    "HookEvent",
    "HookDecision",
    "HookHandler",
    "HOOK_PRE_TOOL_USE",
    "HOOK_POST_TOOL_USE",
    "HOOK_USER_PROMPT_SUBMIT",
]
