"""
Configuration helpers for Open Agent SDK

This module provides optional convenience functions for configuration management.
The SDK core (types.py, client.py) deliberately does NOT use these helpers to
keep the API explicit and simple. However, agent applications can use these
helpers to support environment variables, provider shortcuts, and config files.

Design Philosophy:
- Configuration helpers are OPT-IN (core SDK doesn't use them)
- Explicit is better than implicit (no magic behavior in core)
- Provide flexible fallback chains for convenience
- Support common local LLM providers out of the box
- Environment variables for deployment flexibility
- Config files for persistent settings (optional YAML dependency)

Usage Pattern:
    Instead of hardcoding in your agent:
        AgentOptions(
            system_prompt="...",
            model="qwen2.5-32b-instruct",
            base_url="http://localhost:1234/v1"
        )

    Use helpers for flexibility:
        AgentOptions(
            system_prompt="...",
            model=get_model("qwen2.5-32b"),  # Falls back to env var
            base_url=get_base_url(provider="ollama")  # Uses provider default
        )

    Now users can override via environment variables:
        export OPEN_AGENT_BASE_URL="http://server:1234/v1"
        export OPEN_AGENT_MODEL="llama3.1:70b"
        python your_agent.py  # Uses env vars instead of hardcoded values

Why These Helpers Are Separate:
The core SDK (AgentOptions, Client, query) requires explicit configuration
because explicit is more predictable and debuggable. These helpers add optional
convenience for agent applications that want environment variable support or
provider shortcuts without adding magic to the core.
"""

import os
from typing import Optional
from pathlib import Path

# Default endpoint URLs for common local LLM providers
# These are the standard default ports used by each provider's documentation
# All providers support OpenAI-compatible API endpoints at these URLs
#
# To use a provider:
#   get_base_url(provider="ollama")  -> "http://localhost:11434/v1"
#
# To override for custom setup:
#   export OPEN_AGENT_BASE_URL="http://my-server:8080/v1"
#   get_base_url(provider="ollama")  -> "http://my-server:8080/v1" (env var wins)

# Default ports verified January 2025. Check provider documentation for current defaults
# as these may change with updates to external software.
# Sources: Ollama FAQ (11434), llama.cpp docs (8080), vLLM docs (8000)
PROVIDER_DEFAULTS = {
    "lmstudio": "http://localhost:1234/v1",  # LM Studio common default (customizable with --port)
    "ollama": "http://localhost:11434/v1",  # Ollama default API endpoint (verified)
    "llamacpp": "http://localhost:8080/v1",  # llama.cpp server default (verified)
    "vllm": "http://localhost:8000/v1",  # vLLM default OpenAI-compatible endpoint (verified)
}


def get_base_url(
    base_url: Optional[str] = None,
    provider: Optional[str] = None
) -> str:
    """
    Resolve LLM server base URL from multiple sources with priority-based fallback.

    This function implements a flexible configuration resolution strategy that
    allows users to specify the base_url in multiple ways, with explicit parameters
    taking precedence over environment variables, which take precedence over
    provider defaults.

    Fallback Chain (highest to lowest priority):
        1. Explicit base_url parameter (most specific, wins over everything)
        2. OPEN_AGENT_BASE_URL environment variable (deployment-time override)
        3. Provider default from PROVIDER_DEFAULTS (convenient shortcut)
        4. LM Studio default (sensible fallback for local development)

    Args:
        base_url (Optional[str]): Explicit OpenAI-compatible endpoint URL.
            If provided, this takes precedence over all other sources.
            Must include protocol and /v1 suffix.
            Example: "http://192.168.1.100:1234/v1"

        provider (Optional[str]): Provider shorthand name. Case-insensitive.
            Supported values:
            - "lmstudio": http://localhost:1234/v1
            - "ollama": http://localhost:11434/v1
            - "llamacpp": http://localhost:8080/v1
            - "vllm": http://localhost:8000/v1
            If provider name not recognized, falls back to LM Studio default.

    Returns:
        str: Resolved base URL string, always valid (never None)

    Examples:
        Explicit URL (highest priority):
        >>> get_base_url("http://custom-server:8080/v1")
        'http://custom-server:8080/v1'

        Environment variable override:
        >>> os.environ["OPEN_AGENT_BASE_URL"] = "http://prod-server:1234/v1"
        >>> get_base_url()
        'http://prod-server:1234/v1'
        >>> get_base_url(provider="ollama")  # Env var still wins
        'http://prod-server:1234/v1'

        Provider shortcut:
        >>> get_base_url(provider="ollama")
        'http://localhost:11434/v1'
        >>> get_base_url(provider="LLAMACPP")  # Case-insensitive
        'http://localhost:8080/v1'

        Default fallback (LM Studio):
        >>> get_base_url()
        'http://localhost:1234/v1'

    Usage in Agent Code:
        # Development: use LM Studio
        opts = AgentOptions(
            system_prompt="...",
            model=get_model("qwen2.5"),
            base_url=get_base_url()  # Defaults to LM Studio
        )

        # Production: override via env var
        # export OPEN_AGENT_BASE_URL="http://production-server:8000/v1"
        # No code changes needed!

        # Quick provider switching:
        base_url=get_base_url(provider="ollama")  # Use Ollama instead

    Note:
        This function always returns a valid URL string, never None. The LM Studio
        default ensures there's always a sensible fallback for local development.
    """
    # Priority 1: Explicit parameter wins over everything
    # Use case: Hardcoded URL for specific server or testing
    if base_url:
        return base_url

    # Priority 2: Environment variable (deployment-time configuration)
    # Use case: Production deployment without code changes
    # Example: export OPEN_AGENT_BASE_URL="http://10.0.0.5:1234/v1"
    env_url = os.environ.get("OPEN_AGENT_BASE_URL")
    if env_url:
        return env_url

    # Priority 3: Provider default (convenience shortcut)
    # Use case: Quick switching between local providers
    # Example: get_base_url(provider="ollama") for Ollama default
    if provider:
        provider_lower = provider.lower()  # Case-insensitive lookup
        if provider_lower in PROVIDER_DEFAULTS:
            return PROVIDER_DEFAULTS[provider_lower]

    # Priority 4: Fallback to LM Studio (most common local development setup)
    # LM Studio is widely used and has good UI for beginners
    return PROVIDER_DEFAULTS["lmstudio"]


def get_model(model: Optional[str] = None, *, prefer_env: bool = True) -> Optional[str]:
    """
    Resolve model name from multiple sources with environment variable override.

    Unlike base_url (which has sensible defaults), model names are highly specific
    to the task, provider, and available models. This function provides a flexible
    way to specify models while allowing environment variable overrides for
    deployment-time configuration changes.

    Resolution Strategy:
        When prefer_env=True (default):
            1. Check OPEN_AGENT_MODEL environment variable first
            2. Fall back to model parameter if env var unset
            3. Return None if both are unset

        When prefer_env=False:
            1. Use model parameter if provided
            2. Ignore environment variable entirely
            3. Return None if model parameter unset

    Args:
        model (Optional[str]): Fallback model name/identifier. Provider-specific:
            - Ollama: "qwen2.5:32b", "llama3.1:70b", "codellama:13b"
            - LM Studio: Whatever name shown in LM Studio UI
            - llama.cpp: Usually "local-model" or server-specific name
            - vLLM: Model path or HuggingFace identifier
            Example: "qwen2.5-32b-instruct"
            Note: Model names are examples only - availability varies by provider and time

        prefer_env (bool): If True (default), OPEN_AGENT_MODEL environment variable
            takes precedence over the model parameter. If False, model parameter
            is used as-is and environment variable is ignored.
            Use prefer_env=False when you want to ignore environment overrides.

    Returns:
        Optional[str]: Resolved model name, or None if no model specified anywhere.
            AgentOptions validation will catch None and raise a descriptive error.

    Examples:
        Basic usage with fallback:
        >>> get_model("qwen2.5-32b-instruct")
        'qwen2.5-32b-instruct'

        Environment variable override (default behavior):
        >>> os.environ["OPEN_AGENT_MODEL"] = "llama3.1:70b"
        >>> get_model("qwen2.5-32b")  # Env var wins
        'llama3.1:70b'

        Explicit model, ignore environment:
        >>> os.environ["OPEN_AGENT_MODEL"] = "llama3.1:70b"
        >>> get_model("qwen2.5-32b", prefer_env=False)  # Parameter wins
        'qwen2.5-32b'

        Environment variable only:
        >>> os.environ["OPEN_AGENT_MODEL"] = "codellama:13b"
        >>> get_model()  # No fallback, uses env var
        'codellama:13b'

        No model specified (returns None):
        >>> get_model()  # No env var, no parameter
        None

    Usage Patterns:
        # Pattern 1: Hardcoded fallback with env override (most common)
        model = get_model("qwen2.5-32b-instruct")
        # Development: uses qwen2.5-32b-instruct
        # Production: export OPEN_AGENT_MODEL="llama3.1:70b" (overrides)

        # Pattern 2: Environment variable required (no fallback)
        model = get_model()
        if not model:
            raise ValueError("OPEN_AGENT_MODEL environment variable required")

        # Pattern 3: Always use specific model (ignore environment)
        model = get_model("specialized-model", prefer_env=False)

    Why No Default Model?
        Unlike base_url (which defaults to LM Studio localhost), there's no
        universal default model that works across providers. Model availability
        depends on:
        - What models user has downloaded (Ollama)
        - What models are loaded in LM Studio
        - What model the llama.cpp server is serving
        - Hardware constraints (7B vs 70B models)

        Requiring explicit model specification prevents confusing errors when
        a default model doesn't exist on the user's system.

    Note:
        Model names are case-sensitive and provider-specific. Check your LLM
        server's model list endpoint (/v1/models) for available model names.
    """
    # Priority 1: Environment variable (when prefer_env=True, which is default)
    # Use case: Override model at deployment time without code changes
    # Example: export OPEN_AGENT_MODEL="llama3.1:70b"
    if prefer_env:
        env_model = os.environ.get("OPEN_AGENT_MODEL")
        if env_model:
            return env_model

    # Priority 2: Explicit parameter (fallback or only source if prefer_env=False)
    # Use case: Hardcoded model for specific agent/task
    if model:
        return model

    # Priority 3: Check env var again if prefer_env=True and model param was None
    # This handles the case where get_model() is called with no arguments
    # and we want to use the environment variable if it exists
    if prefer_env:
        return os.environ.get("OPEN_AGENT_MODEL")

    # No model specified anywhere - return None
    # AgentOptions.__post_init__ will validate and raise descriptive error
    return None


def load_config_file(config_path: Optional[Path] = None) -> dict:
    """
    Load agent configuration from YAML file with standard search paths.

    This function provides an optional YAML-based configuration system for users
    who prefer config files over environment variables or hardcoded values. The
    YAML dependency is optional - if PyYAML is not installed, this returns an
    empty dict without raising an error.

    Search Path Priority (when config_path is None):
        1. ./open-agent.yaml (project-specific config, highest priority)
        2. ~/.config/open-agent/config.yaml (XDG Base Directory standard)
        3. ~/.open-agent.yaml (fallback home directory config)

    The first file found in the search path is loaded. If no files exist or
    PyYAML is not installed, returns an empty dictionary.

    Args:
        config_path (Optional[Path]): Explicit path to YAML config file.
            If provided, only this path is checked (no default search).
            If None, checks standard locations in priority order.
            Example: Path("/etc/my-agent/config.yaml")

    Returns:
        dict: Configuration dictionary loaded from YAML file, or empty dict if:
            - No config file found in search paths
            - PyYAML package not installed (optional dependency)
            - Config file exists but is empty
            - Config file has invalid YAML (returns {}, doesn't raise)

    Config File Format (YAML):
        ```yaml
        # Server connection
        base_url: http://localhost:1234/v1
        model: qwen2.5-32b-instruct
        api_key: optional-key-here

        # Generation parameters
        temperature: 0.7
        max_tokens: 4096
        max_turns: 10

        # Tool execution
        auto_execute_tools: true
        max_tool_iterations: 5
        ```

    Example Usage:
        Load from standard locations:
        >>> config = load_config_file()
        >>> config.get("model", "default-model")
        'qwen2.5-32b-instruct'  # From ~/.open-agent.yaml

        Use config to build AgentOptions:
        >>> config = load_config_file()
        >>> opts = AgentOptions(
        ...     system_prompt="You are a helpful assistant",
        ...     model=config.get("model", "qwen2.5-32b"),
        ...     base_url=config.get("base_url", get_base_url()),
        ...     temperature=config.get("temperature", 0.7)
        ... )

        Load from specific path:
        >>> config = load_config_file(Path("./my-agent-config.yaml"))

    Priority Combination (recommended pattern):
        ```python
        # Load base config from file
        config = load_config_file()

        # Environment variables override config file
        # Command-line args would override environment variables
        opts = AgentOptions(
            system_prompt="...",
            model=get_model(config.get("model")),  # Env > config > param
            base_url=get_base_url(config.get("base_url")),  # Env > config > param
            temperature=config.get("temperature", 0.7)
        )
        ```

    Implementation Notes:
        - Uses yaml.safe_load() for security (no arbitrary Python execution)
        - Returns {} instead of raising on missing PyYAML (graceful degradation)
        - Returns {} instead of None for easier dict.get() usage
        - File I/O errors are NOT caught - will raise if file exists but unreadable
        - YAML parsing errors are NOT caught - will raise yaml.YAMLError on invalid YAML
          (fail-fast approach: malformed config is a user error, not a recoverable condition)

    PyYAML Installation:
        ```bash
        pip install open-agent-sdk[yaml]
        # or directly (see pyproject.toml for current version requirements)
        pip install pyyaml
        ```

    Why YAML is Optional:
        Not all users want config files. Many prefer explicit code or environment
        variables. Making PyYAML optional keeps the core SDK lightweight and avoids
        forcing a dependency on users who don't need it.
    """
    # Try to import PyYAML (optional dependency)
    try:
        import yaml
    except ImportError:
        # PyYAML not installed - gracefully return empty config
        # This allows the SDK to work without YAML support
        # Users who want config files should install with: pip install open-agent-sdk[yaml]
        return {}

    # Build list of paths to search for config file
    search_paths = []

    if config_path:
        # Explicit path provided - only check this one
        search_paths.append(config_path)
    else:
        # Default search locations (checked in priority order)
        search_paths.extend([
            # 1. Project directory (highest priority, for project-specific settings)
            Path.cwd() / "open-agent.yaml",

            # 2. XDG Base Directory standard location (Linux/Unix best practice)
            Path.home() / ".config" / "open-agent" / "config.yaml",

            # 3. Home directory dotfile (fallback for compatibility)
            Path.home() / ".open-agent.yaml",
        ])

    # Search paths in order, return first file found
    for path in search_paths:
        if path.exists():
            with open(path) as f:
                # Use safe_load for security (no code execution)
                # yaml.safe_load returns None for empty files, so we use `or {}`
                return yaml.safe_load(f) or {}

    # No config file found in any search location
    return {}
