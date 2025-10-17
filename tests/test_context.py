"""Tests for context management utilities"""
import sys
from unittest.mock import patch

import pytest

from open_agent.context import estimate_tokens, truncate_messages


class TestEstimateTokens:
    """Test token estimation functionality"""

    def test_estimate_tokens_with_tiktoken(self):
        """Test token estimation when tiktoken is available"""
        # Skip if tiktoken not installed
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            pytest.skip("tiktoken not installed")

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"},
        ]

        tokens = estimate_tokens(messages)

        # Should be roughly 30-40 tokens with tiktoken
        assert 20 < tokens < 50, f"Expected 20-50 tokens, got {tokens}"

    def test_estimate_tokens_fallback(self):
        """Test fallback to character-based estimation"""
        # Mock tiktoken import to fail
        with patch.dict(sys.modules, {"tiktoken": None}):
            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Hello!"},
            ]

            tokens = estimate_tokens(messages)

            # Character-based: ~40 chars / 4 = ~10 tokens
            assert tokens > 0, "Should return non-zero token count"

    def test_estimate_tokens_fallback_short_message(self):
        """Fallback for very short content should still return >=1 token"""
        with patch.dict(sys.modules, {"tiktoken": None}):
            messages = [{"role": "user", "content": "Hi"}]

            tokens = estimate_tokens(messages)

            # "user" (4 chars) + "Hi" (2 chars) = 6 chars -> ceil(6/4) = 2 tokens
            assert tokens >= 1, "Short messages should return at least 1 token"

    def test_estimate_tokens_empty_messages(self):
        """Test with empty message list"""
        tokens = estimate_tokens([])
        assert tokens >= 0, "Empty messages should return non-negative count"

    def test_estimate_tokens_complex_content(self):
        """Test with tool use blocks and complex content"""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll help you with that."},
                    {
                        "type": "tool_use",
                        "name": "calculator",
                        "input": {"expression": "2+2"},
                    },
                ],
            },
        ]

        tokens = estimate_tokens(messages)
        assert tokens > 0, "Should handle complex content structures"

    def test_estimate_tokens_includes_nested_function_arguments(self):
        """Token estimation should include nested tool/function arguments"""
        arguments = '{"query": "select * from table"}'
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "db_query",
                        "input": {"sql": arguments},
                    }
                ],
            }
        ]

        with patch.dict(sys.modules, {"tiktoken": None}):
            tokens = estimate_tokens(messages)

        # Should account for argument length (len(arguments) == 33)
        assert tokens >= 9, f"Expected nested arguments to contribute tokens, got {tokens}"

    def test_estimate_tokens_custom_model(self):
        """Test with custom model parameter"""
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            pytest.skip("tiktoken not installed")

        messages = [{"role": "user", "content": "Test message"}]

        # Should handle different model names
        tokens1 = estimate_tokens(messages, model="gpt-3.5-turbo")
        tokens2 = estimate_tokens(messages, model="gpt-4")

        # Both should return reasonable counts
        assert tokens1 > 0
        assert tokens2 > 0

    def test_estimate_tokens_unknown_model_fallback(self):
        """Test fallback when model not recognized by tiktoken"""
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            pytest.skip("tiktoken not installed")

        messages = [{"role": "user", "content": "Test"}]

        # Should not crash with unknown model, use default encoding
        tokens = estimate_tokens(messages, model="unknown-model-12345")
        assert tokens > 0


class TestTruncateMessages:
    """Test message truncation functionality"""

    def test_truncate_basic(self):
        """Test basic truncation"""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Message 3"},
        ]

        truncated = truncate_messages(messages, keep=2)

        # Should keep system + 2 most recent = 3 total
        assert len(truncated) == 3
        assert truncated[0]["role"] == "system"
        assert truncated[-1]["content"] == "Message 3"

    def test_truncate_no_system_message(self):
        """Test truncation without system message"""
        messages = [
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
            {"role": "assistant", "content": "Response 2"},
        ]

        truncated = truncate_messages(messages, keep=2)

        # Should keep last 2 messages only
        assert len(truncated) == 2
        assert truncated[0]["content"] == "Message 2"
        assert truncated[1]["content"] == "Response 2"

    def test_truncate_preserve_system_false(self):
        """Test truncation without preserving system message"""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
        ]

        truncated = truncate_messages(messages, keep=2, preserve_system=False)

        # Should keep last 2 only, no system
        assert len(truncated) == 2
        assert truncated[0]["content"] == "Message 1"
        assert truncated[1]["content"] == "Response 1"

    def test_truncate_keep_all(self):
        """Test when keep >= message count"""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
        ]

        truncated = truncate_messages(messages, keep=10)

        # Should return copy of all messages
        assert len(truncated) == len(messages)
        assert truncated == messages
        assert truncated is not messages  # Should be a copy

    def test_truncate_empty_messages(self):
        """Test with empty message list"""
        truncated = truncate_messages([])
        assert truncated == []

    def test_truncate_keep_zero(self):
        """Test with keep=0"""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Message"},
        ]

        truncated = truncate_messages(messages, keep=0)

        # Should still keep system message if preserve_system=True
        assert len(truncated) == 1
        assert truncated[0]["role"] == "system"

    def test_truncate_keep_one(self):
        """Test with keep=1"""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Message 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Message 2"},
        ]

        truncated = truncate_messages(messages, keep=1)

        # Should keep system + 1 most recent = 2 total
        assert len(truncated) == 2
        assert truncated[0]["role"] == "system"
        assert truncated[1]["content"] == "Message 2"

    def test_truncate_does_not_modify_original(self):
        """Test that original messages are not modified"""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Message 1"},
            {"role": "user", "content": "Message 2"},
        ]
        original_length = len(messages)

        truncated = truncate_messages(messages, keep=1)

        # Original should be unchanged
        assert len(messages) == original_length
        assert len(truncated) < original_length

    def test_truncate_preserves_message_structure(self):
        """Test that message structure is preserved"""
        messages = [
            {"role": "system", "content": "System"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Response"},
                    {"type": "tool_use", "id": "1", "name": "test", "input": {}},
                ],
            },
        ]

        truncated = truncate_messages(messages, keep=5)

        # Should preserve complex content structure
        assert len(truncated) == 2
        assert isinstance(truncated[1]["content"], list)
        assert len(truncated[1]["content"]) == 2


class TestIntegration:
    """Integration tests combining multiple utilities"""

    def test_estimate_then_truncate_workflow(self):
        """Test realistic workflow: estimate tokens, then truncate if needed"""
        messages = []

        # Build up a long conversation
        messages.append({"role": "system", "content": "You are a helpful assistant"})
        for i in range(50):
            messages.append({"role": "user", "content": f"Question {i}"})
            messages.append({"role": "assistant", "content": f"Response {i}"})

        # Estimate tokens
        tokens = estimate_tokens(messages)
        assert tokens > 0

        # Truncate if needed (always true with 50 turns)
        if len(messages) > 10:
            truncated = truncate_messages(messages, keep=10)
            assert len(truncated) == 11  # system + 10 recent
            assert truncated[0]["role"] == "system"

            # Verify truncated has fewer tokens
            truncated_tokens = estimate_tokens(truncated)
            assert truncated_tokens < tokens

    def test_realistic_agent_conversation(self):
        """Test with realistic agent conversation pattern"""
        messages = [
            {"role": "system", "content": "You are a code review assistant"},
            {"role": "user", "content": "Review this function: def foo(): pass"},
            {
                "role": "assistant",
                "content": "I'll review that function. Let me analyze it.",
            },
            {"role": "user", "content": "What did you find?"},
            {
                "role": "assistant",
                "content": "The function is too simple, needs docstring and type hints.",
            },
        ]

        # Should handle realistic conversation
        tokens = estimate_tokens(messages)
        assert tokens > 0

        truncated = truncate_messages(messages, keep=3)
        assert len(truncated) == 4  # system + 3 recent
        assert truncated[0]["role"] == "system"
