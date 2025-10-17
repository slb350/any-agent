"""Type definitions for Open Agent SDK"""
from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tools import Tool
    from .hooks import HookHandler


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
class ToolUseError:
    """Error when tool call fails to parse"""
    error: str
    raw_data: str | None = None
    type: Literal["tool_use_error"] = "tool_use_error"


@dataclass
class ToolResultBlock:
    """Tool execution result"""
    tool_use_id: str
    content: str | dict[str, Any] | list[Any]
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


@dataclass
class AssistantMessage:
    """Full assistant message"""
    role: Literal["assistant"] = "assistant"
    content: list[TextBlock | ToolUseBlock | ToolUseError | ToolResultBlock] = field(default_factory=list)


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
        tools: List of Tool instances for function calling (optional)
        hooks: Dict of hook event names to handler lists (optional)
        auto_execute_tools: Enable automatic tool execution (default False for backward compatibility)
        max_tool_iterations: Maximum tool execution iterations to prevent infinite loops (default 5)
        max_turns: Maximum conversation turns
        max_tokens: Tokens to generate (None uses provider default)
        temperature: Sampling temperature
        timeout: Request timeout in seconds (default 60.0)
        api_key: API key (most local servers don't need this)

    Examples:
        # Explicit configuration (simplest)
        AgentOptions(
            system_prompt="...",
            model="qwen2.5-32b-instruct",
            base_url="http://localhost:1234/v1"
        )

        # With automatic tool execution
        AgentOptions(
            system_prompt="...",
            model="qwen2.5-32b-instruct",
            base_url="http://localhost:1234/v1",
            tools=[add, subtract, multiply],
            auto_execute_tools=True,
            max_tool_iterations=10
        )

        # Using config helpers in your agent (optional)
        from open_agent.config import get_base_url, get_model

        AgentOptions(
            system_prompt="...",
            model=get_model("qwen2.5-32b"),  # Falls back to OPEN_AGENT_MODEL env var
            base_url=get_base_url(provider="ollama")  # Uses provider default
        )
    """
    system_prompt: str
    model: str
    base_url: str
    tools: list["Tool"] = field(default_factory=list)
    hooks: dict[str, list["HookHandler"]] | None = None
    auto_execute_tools: bool = False
    max_tool_iterations: int = 5
    max_turns: int = 1
    max_tokens: int | None = 4096  # Default 4096, None uses provider default
    temperature: float = 0.7
    timeout: float = 60.0  # Request timeout in seconds
    api_key: str = "not-needed"

    def __post_init__(self):
        """Validate configuration"""
        # Basic URL validation
        if not self.base_url:
            raise ValueError("base_url cannot be empty")

        if not (self.base_url.startswith("http://") or self.base_url.startswith("https://")):
            raise ValueError(f"base_url must start with http:// or https://, got: {self.base_url}")

        # Basic model validation
        if not self.model:
            raise ValueError("model cannot be empty")
