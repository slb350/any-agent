"""Tests for AgentOptions dataclass"""
import os
import pytest
from open_agent.types import AgentOptions


def test_agent_options_minimal():
    """Test AgentOptions with required parameters"""
    options = AgentOptions(
        system_prompt="Test",
        model="qwen2.5-32b-instruct",
        base_url="http://localhost:1234/v1"
    )
    assert options.model == "qwen2.5-32b-instruct"
    assert options.base_url == "http://localhost:1234/v1"
    assert options.system_prompt == "Test"


def test_agent_options_with_defaults():
    """Test AgentOptions default values"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )
    assert options.max_turns == 1
    assert options.max_tokens == 4096
    assert options.temperature == 0.7
    assert options.api_key == "not-needed"


def test_agent_options_missing_model():
    """Test AgentOptions requires model parameter"""
    with pytest.raises(TypeError):
        AgentOptions(
            system_prompt="Test",
            base_url="http://localhost:1234/v1"
        )


def test_agent_options_missing_base_url():
    """Test AgentOptions requires base_url parameter"""
    with pytest.raises(TypeError):
        AgentOptions(
            system_prompt="Test",
            model="test-model"
        )


def test_agent_options_full_explicit():
    """Test all parameters explicit"""
    options = AgentOptions(
        system_prompt="Test",
        model="qwen2.5-32b-instruct",
        base_url="http://custom:8080/v1",
        max_turns=10,
        temperature=0.5,
        max_tokens=2048,
        api_key="test-key"
    )
    assert options.model == "qwen2.5-32b-instruct"
    assert options.base_url == "http://custom:8080/v1"
    assert options.max_turns == 10
    assert options.temperature == 0.5
    assert options.max_tokens == 2048
    assert options.api_key == "test-key"


def test_agent_options_max_tokens_none():
    """Test max_tokens can be None"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1",
        max_tokens=None
    )
    assert options.max_tokens is None


def test_agent_options_timeout_default():
    """Test timeout has default value of 60 seconds"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )
    assert options.timeout == 60.0


def test_agent_options_custom_timeout():
    """Test custom timeout value"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1",
        timeout=120.0
    )
    assert options.timeout == 120.0


def test_agent_options_invalid_url_no_protocol():
    """Test validation rejects URL without protocol"""
    with pytest.raises(ValueError, match="base_url must start with http:// or https://"):
        AgentOptions(
            system_prompt="Test",
            model="test-model",
            base_url="localhost:1234/v1"
        )


def test_agent_options_invalid_url_empty():
    """Test validation rejects empty URL"""
    with pytest.raises(ValueError, match="base_url cannot be empty"):
        AgentOptions(
            system_prompt="Test",
            model="test-model",
            base_url=""
        )


def test_agent_options_invalid_model_empty():
    """Test validation rejects empty model"""
    with pytest.raises(ValueError, match="model cannot be empty"):
        AgentOptions(
            system_prompt="Test",
            model="",
            base_url="http://localhost:1234/v1"
        )


def test_agent_options_valid_https_url():
    """Test validation accepts HTTPS URL"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="https://secure.example.com/v1"
    )
    assert options.base_url == "https://secure.example.com/v1"
