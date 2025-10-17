"""Tests for AgentOptions dataclass"""
import os
import pytest
from any_agent.types import AgentOptions


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
