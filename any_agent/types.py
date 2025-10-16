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

    Configuration resolution order:

    base_url:
    1. Explicit base_url parameter
    2. Environment variable ANY_AGENT_BASE_URL
    3. Provider default (if provider specified)
    4. Default to LM Studio (http://localhost:1234/v1)

    model:
    1. Explicit model parameter
    2. Environment variable ANY_AGENT_MODEL
    3. Required - raises error if not provided

    Args:
        system_prompt: System prompt for the model
        model: Model name (optional, uses ANY_AGENT_MODEL env var if not provided)
        base_url: OpenAI-compatible endpoint URL (optional)
        provider: Provider shorthand (lmstudio, ollama, llamacpp, vllm)
        max_turns: Maximum conversation turns
        max_tokens: Tokens to generate (None uses provider default)
        temperature: Sampling temperature
        api_key: API key (most local servers don't need this)

    Examples:
        # Explicit everything
        AgentOptions(system_prompt="...", model="qwen2.5-32b", base_url="http://server:1234/v1")

        # Use environment variables
        # export ANY_AGENT_BASE_URL="https://server.com/v1"
        # export ANY_AGENT_MODEL="qwen2.5-32b-instruct"
        AgentOptions(system_prompt="...")

        # Mix explicit and environment
        AgentOptions(system_prompt="...", model="llama3.1:70b", provider="ollama")

    Raises:
        ValueError: If model is not provided and ANY_AGENT_MODEL is not set
    """
    system_prompt: str
    model: Optional[str] = None
    base_url: Optional[str] = None
    provider: Optional[str] = None
    max_turns: int = 1
    max_tokens: int | None = 4096  # Default 4096, None uses provider default
    temperature: float = 0.7
    api_key: str = "not-needed"

    def __post_init__(self):
        """Resolve base_url and model from config if not provided"""
        # Resolve base_url
        if self.base_url is None:
            from .config import get_base_url
            self.base_url = get_base_url(base_url=None, provider=self.provider)

        # Resolve model
        if self.model is None:
            from .config import get_model
            self.model = get_model(model=None)

        # Model is required
        if self.model is None:
            raise ValueError(
                "Model must be specified either as a parameter, "
                "via environment variable ANY_AGENT_MODEL, "
                "or in a config file."
            )
