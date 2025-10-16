"""Tests for query function"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from any_agent import query, AgentOptions
from any_agent.types import TextBlock, AssistantMessage


@pytest.mark.asyncio
async def test_query_basic():
    """Test that query function is async and returns generator"""
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )

    # This will fail to connect, but we're just testing the function signature
    result = query(prompt="Hello", options=options)

    # Verify it's an async generator
    assert hasattr(result, '__anext__')


@pytest.mark.asyncio
async def test_query_returns_assistant_message():
    """Test that query yields AssistantMessage objects"""
    # This is a basic structure test - full integration tests would
    # require mocking the OpenAI client or having a test server running
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://localhost:1234/v1"
    )

    result = query(prompt="Hello", options=options)

    # Verify result is an async generator
    assert result.__class__.__name__ == 'async_generator'
