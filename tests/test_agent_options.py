"""Tests for AgentOptions dataclass"""
import os
import pytest
from any_agent.types import AgentOptions


def test_agent_options_explicit_model():
    """Test AgentOptions with explicit model"""
    options = AgentOptions(
        system_prompt="Test",
        model="qwen2.5-32b-instruct"
    )
    assert options.model == "qwen2.5-32b-instruct"
    assert options.base_url == "http://localhost:1234/v1"  # Default


def test_agent_options_model_from_env(monkeypatch):
    """Test AgentOptions resolves model from environment variable"""
    monkeypatch.setenv("ANY_AGENT_MODEL", "llama3.1:70b")

    options = AgentOptions(
        system_prompt="Test"
    )
    assert options.model == "llama3.1:70b"


def test_agent_options_no_model_raises_error():
    """Test AgentOptions raises ValueError when model not provided"""
    with pytest.raises(ValueError, match="Model must be specified"):
        AgentOptions(
            system_prompt="Test"
        )


def test_agent_options_explicit_model_overrides_env(monkeypatch):
    """Test explicit model overrides environment variable"""
    monkeypatch.setenv("ANY_AGENT_MODEL", "llama3.1:70b")

    options = AgentOptions(
        system_prompt="Test",
        model="qwen2.5-32b-instruct"
    )
    assert options.model == "qwen2.5-32b-instruct"


def test_agent_options_base_url_and_model_from_env(monkeypatch):
    """Test both base_url and model from environment variables"""
    monkeypatch.setenv("ANY_AGENT_BASE_URL", "https://server.com/v1")
    monkeypatch.setenv("ANY_AGENT_MODEL", "qwen2.5-32b-instruct")

    options = AgentOptions(
        system_prompt="Test"
    )
    assert options.base_url == "https://server.com/v1"
    assert options.model == "qwen2.5-32b-instruct"


def test_agent_options_provider_and_explicit_model():
    """Test provider shorthand with explicit model"""
    options = AgentOptions(
        system_prompt="Test",
        model="llama3.1:70b",
        provider="ollama"
    )
    assert options.base_url == "http://localhost:11434/v1"
    assert options.model == "llama3.1:70b"


def test_agent_options_full_explicit():
    """Test all parameters explicit"""
    options = AgentOptions(
        system_prompt="Test",
        model="qwen2.5-32b-instruct",
        base_url="http://custom:8080/v1",
        max_turns=10,
        temperature=0.5
    )
    assert options.model == "qwen2.5-32b-instruct"
    assert options.base_url == "http://custom:8080/v1"
    assert options.max_turns == 10
    assert options.temperature == 0.5
