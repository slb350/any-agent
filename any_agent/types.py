"""Type definitions matching claude-agent-sdk patterns"""
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TextBlock:
    """Text content from assistant"""
    text: str
    type: Literal["text"] = "text"


@dataclass
class ToolUseBlock:
    """Tool call from assistant"""
    id: str
    name: str
    input: dict
    type: Literal["tool_use"] = "tool_use"


@dataclass
class AssistantMessage:
    """Full assistant message"""
    role: Literal["assistant"] = "assistant"
    content: list[TextBlock | ToolUseBlock] = field(default_factory=list)


@dataclass
class ToolUseError:
    """Error when tool call fails to parse"""
    error: str
    raw_data: str | None = None
    type: Literal["tool_use_error"] = "tool_use_error"


@dataclass
class AgentOptions:
    """Configuration for agent"""
    system_prompt: str
    model: str
    base_url: str
    max_turns: int = 1
    max_tokens: int | None = 4096  # Default 4096, None uses provider default
    temperature: float = 0.7
    api_key: str = "not-needed"
