"""Type definitions matching claude-agent-sdk patterns"""
from dataclasses import dataclass, field
from typing import Literal, Optional


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
    """
    Configuration for agent.

    The SDK keeps configuration minimal - you provide the model and base_url.
    Use the config module helpers if you want environment variable or provider
    shorthand support in your agent code.

    Args:
        system_prompt: System prompt for the model
        model: Model name (e.g., "qwen2.5-32b-instruct")
        base_url: OpenAI-compatible endpoint URL
        max_turns: Maximum conversation turns
        max_tokens: Tokens to generate (None uses provider default)
        temperature: Sampling temperature
        api_key: API key (most local servers don't need this)

    Examples:
        # Explicit configuration (simplest)
        AgentOptions(
            system_prompt="...",
            model="qwen2.5-32b-instruct",
            base_url="http://localhost:1234/v1"
        )

        # Using config helpers in your agent (optional)
        from any_agent.config import get_base_url, get_model

        AgentOptions(
            system_prompt="...",
            model=get_model("qwen2.5-32b"),  # Falls back to ANY_AGENT_MODEL env var
            base_url=get_base_url(provider="ollama")  # Uses provider default
        )
    """
    system_prompt: str
    model: str
    base_url: str
    max_turns: int = 1
    max_tokens: int | None = 4096  # Default 4096, None uses provider default
    temperature: float = 0.7
    api_key: str = "not-needed"
