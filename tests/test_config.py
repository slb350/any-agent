"""Tests for config module"""
import os
import pytest
from pathlib import Path
from open_agent.config import get_base_url, get_model, PROVIDER_DEFAULTS, load_config_file


def test_get_base_url_explicit():
    """Test explicit base_url has highest priority"""
    url = get_base_url(base_url="http://custom:8080/v1")
    assert url == "http://custom:8080/v1"


def test_get_base_url_env_var(monkeypatch):
    """Test environment variable is used when no explicit URL"""
    monkeypatch.setenv("OPEN_AGENT_BASE_URL", "http://env-server:1234/v1")
    url = get_base_url()
    assert url == "http://env-server:1234/v1"


def test_get_base_url_explicit_overrides_env(monkeypatch):
    """Test explicit URL overrides environment variable"""
    monkeypatch.setenv("OPEN_AGENT_BASE_URL", "http://env-server:1234/v1")
    url = get_base_url(base_url="http://explicit:8080/v1")
    assert url == "http://explicit:8080/v1"


def test_get_base_url_provider_lmstudio():
    """Test provider default for LM Studio"""
    url = get_base_url(provider="lmstudio")
    assert url == "http://localhost:1234/v1"


def test_get_base_url_provider_ollama():
    """Test provider default for Ollama"""
    url = get_base_url(provider="ollama")
    assert url == "http://localhost:11434/v1"


def test_get_base_url_provider_llamacpp():
    """Test provider default for llama.cpp"""
    url = get_base_url(provider="llamacpp")
    assert url == "http://localhost:8080/v1"


def test_get_base_url_provider_vllm():
    """Test provider default for vLLM"""
    url = get_base_url(provider="vllm")
    assert url == "http://localhost:8000/v1"


def test_get_base_url_provider_case_insensitive():
    """Test provider name is case-insensitive"""
    url = get_base_url(provider="OLLAMA")
    assert url == "http://localhost:11434/v1"


def test_get_base_url_provider_unknown():
    """Test unknown provider falls back to default"""
    url = get_base_url(provider="unknown-provider")
    assert url == PROVIDER_DEFAULTS["lmstudio"]


def test_get_base_url_default():
    """Test default is LM Studio when nothing specified"""
    url = get_base_url()
    assert url == "http://localhost:1234/v1"


def test_get_base_url_explicit_overrides_provider():
    """Test explicit URL overrides provider default"""
    url = get_base_url(
        base_url="http://custom:8080/v1",
        provider="ollama"
    )
    assert url == "http://custom:8080/v1"


def test_get_base_url_env_overrides_provider(monkeypatch):
    """Test environment variable overrides provider default"""
    monkeypatch.setenv("OPEN_AGENT_BASE_URL", "http://env-server:1234/v1")
    url = get_base_url(provider="ollama")
    assert url == "http://env-server:1234/v1"


def test_load_config_file_no_yaml():
    """Test load_config_file returns empty dict when YAML not installed"""
    # This will work even without YAML installed since it catches ImportError
    config = load_config_file(Path("/nonexistent/path/config.yaml"))
    assert isinstance(config, dict)


def test_load_config_file_nonexistent():
    """Test load_config_file returns empty dict for nonexistent file"""
    config = load_config_file(Path("/nonexistent/path/config.yaml"))
    assert config == {}


def test_provider_defaults_exist():
    """Test that all expected providers have defaults"""
    assert "lmstudio" in PROVIDER_DEFAULTS
    assert "ollama" in PROVIDER_DEFAULTS
    assert "llamacpp" in PROVIDER_DEFAULTS
    assert "vllm" in PROVIDER_DEFAULTS


def test_provider_defaults_format():
    """Test that provider defaults are valid URLs"""
    for provider, url in PROVIDER_DEFAULTS.items():
        assert url.startswith("http://") or url.startswith("https://")
        assert "/v1" in url


# Model configuration tests


def test_get_model_returns_fallback_when_env_missing(monkeypatch):
    """Fallback parameter should be used when env var is absent."""
    monkeypatch.delenv("OPEN_AGENT_MODEL", raising=False)
    model = get_model(model="qwen2.5-32b-instruct")
    assert model == "qwen2.5-32b-instruct"


def test_get_model_env_var_overrides_fallback(monkeypatch):
    """Environment variable should override provided fallback by default."""
    monkeypatch.setenv("OPEN_AGENT_MODEL", "llama3.1:70b")
    model = get_model(model="qwen2.5-32b-instruct")
    assert model == "llama3.1:70b"


def test_get_model_can_ignore_env(monkeypatch):
    """prefer_env=False should force the fallback model."""
    monkeypatch.setenv("OPEN_AGENT_MODEL", "llama3.1:70b")
    model = get_model(model="qwen2.5-32b-instruct", prefer_env=False)
    assert model == "qwen2.5-32b-instruct"


def test_get_model_none_when_not_set(monkeypatch):
    """Test returns None when nothing specified."""
    monkeypatch.delenv("OPEN_AGENT_MODEL", raising=False)
    model = get_model()
    assert model is None
