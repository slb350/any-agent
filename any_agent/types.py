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

    The base_url is resolved in this order:
    1. Explicit base_url parameter
    2. Environment variable ANY_AGENT_BASE_URL
    3. Provider default (if provider specified)
    4. Default to LM Studio (http://localhost:1234/v1)

    Args:
        system_prompt: System prompt for the model
        model: Model name (e.g., "qwen2.5-32b-instruct")
        base_url: OpenAI-compatible endpoint URL (optional)
        provider: Provider shorthand (lmstudio, ollama, llamacpp, vllm)
        max_turns: Maximum conversation turns
        max_tokens: Tokens to generate (None uses provider default)
        temperature: Sampling temperature
        api_key: API key (most local servers don't need this)

    Examples:
        # Explicit URL
        AgentOptions(system_prompt="...", model="...", base_url="http://server:1234/v1")

        # Use environment variable ANY_AGENT_BASE_URL
        AgentOptions(system_prompt="...", model="...")

        # Use provider default
        AgentOptions(system_prompt="...", model="...", provider="ollama")
    """
    system_prompt: str
    model: str
    base_url: Optional[str] = None
    provider: Optional[str] = None
    max_turns: int = 1
    max_tokens: int | None = 4096  # Default 4096, None uses provider default
    temperature: float = 0.7
    api_key: str = "not-needed"

    def __post_init__(self):
        """Resolve base_url from config if not provided"""
        if self.base_url is None:
            from .config import get_base_url
            self.base_url = get_base_url(base_url=None, provider=self.provider)
