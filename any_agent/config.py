"""Configuration management for any-agent"""
import os
from typing import Optional
from pathlib import Path

# Default endpoints for common providers
PROVIDER_DEFAULTS = {
    "lmstudio": "http://localhost:1234/v1",
    "ollama": "http://localhost:11434/v1",
    "llamacpp": "http://localhost:8080/v1",
    "vllm": "http://localhost:8000/v1",
}


def get_base_url(
    base_url: Optional[str] = None,
    provider: Optional[str] = None
) -> str:
    """
    Get base URL from multiple sources with fallback chain:
    1. Explicit base_url parameter
    2. Environment variable ANY_AGENT_BASE_URL
    3. Provider default (if provider specified)
    4. Default to LM Studio localhost

    Args:
        base_url: Explicit base URL (highest priority)
        provider: Provider name (lmstudio, ollama, llamacpp, vllm)

    Returns:
        Base URL string

    Examples:
        >>> get_base_url("http://custom:8080/v1")
        'http://custom:8080/v1'

        >>> os.environ["ANY_AGENT_BASE_URL"] = "http://server:1234/v1"
        >>> get_base_url()
        'http://server:1234/v1'

        >>> get_base_url(provider="ollama")
        'http://localhost:11434/v1'
    """
    # 1. Explicit parameter (highest priority)
    if base_url:
        return base_url

    # 2. Environment variable
    env_url = os.environ.get("ANY_AGENT_BASE_URL")
    if env_url:
        return env_url

    # 3. Provider default
    if provider:
        provider_lower = provider.lower()
        if provider_lower in PROVIDER_DEFAULTS:
            return PROVIDER_DEFAULTS[provider_lower]

    # 4. Default to LM Studio
    return PROVIDER_DEFAULTS["lmstudio"]


def get_model(model: Optional[str] = None) -> Optional[str]:
    """
    Get model name from multiple sources with fallback chain:
    1. Explicit model parameter
    2. Environment variable ANY_AGENT_MODEL
    3. Return None (model must be specified somewhere)

    Args:
        model: Explicit model name (highest priority)

    Returns:
        Model name string or None

    Examples:
        >>> get_model("qwen2.5-32b-instruct")
        'qwen2.5-32b-instruct'

        >>> os.environ["ANY_AGENT_MODEL"] = "llama3.1:70b"
        >>> get_model()
        'llama3.1:70b'

    Note:
        Unlike base_url, there's no sensible default model that works
        across all providers. Model must be specified explicitly,
        via environment variable, or in config file.
    """
    # 1. Explicit parameter (highest priority)
    if model:
        return model

    # 2. Environment variable
    env_model = os.environ.get("ANY_AGENT_MODEL")
    if env_model:
        return env_model

    # 3. No default - model is agent/task specific
    return None


def load_config_file(config_path: Optional[Path] = None) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file. If None, checks:
                    1. ./any-agent.yaml
                    2. ~/.config/any-agent/config.yaml
                    3. ~/.any-agent.yaml

    Returns:
        Configuration dictionary

    Config file format:
        base_url: http://localhost:1234/v1
        model: qwen2.5-32b-instruct
        temperature: 0.7
        max_tokens: 4096
    """
    try:
        import yaml
    except ImportError:
        # YAML is optional dependency
        return {}

    search_paths = []

    if config_path:
        search_paths.append(config_path)
    else:
        # Default search locations
        search_paths.extend([
            Path.cwd() / "any-agent.yaml",
            Path.home() / ".config" / "any-agent" / "config.yaml",
            Path.home() / ".any-agent.yaml",
        ])

    for path in search_paths:
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}

    return {}
