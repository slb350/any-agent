"""Tests for hooks system"""
import pytest

from open_agent import (
    AgentOptions,
    Client,
    query,
    PreToolUseEvent,
    PostToolUseEvent,
    UserPromptSubmitEvent,
    HookDecision,
    HOOK_PRE_TOOL_USE,
    HOOK_POST_TOOL_USE,
    HOOK_USER_PROMPT_SUBMIT,
    ToolUseBlock,
    ToolUseError,
)


class TestPreToolUseHook:
    """Test PreToolUse hook functionality"""

    @pytest.mark.asyncio
    async def test_pre_tool_use_allows_execution(self, fake_client_with_tools):
        """PreToolUse hook that returns None allows execution"""
        hook_calls = []

        async def allow_hook(event: PreToolUseEvent) -> HookDecision | None:
            hook_calls.append(("allow", event.tool_name))
            return None  # Allow execution

        options = AgentOptions(
            system_prompt="Test",
            model="test-model",
            base_url="http://localhost:1234/v1",
            tools=fake_client_with_tools["tools"],
            hooks={HOOK_PRE_TOOL_USE: [allow_hook]},
        )

        async with Client(options) as client:
            await client.query("Call the calculator")

            blocks = []
            async for block in client.receive_messages():
                blocks.append(block)

        # Hook should be called
        assert len(hook_calls) == 1
        assert hook_calls[0][0] == "allow"
        assert hook_calls[0][1] == "calculator"

        # Tool should be executed normally
        assert any(isinstance(b, ToolUseBlock) and b.name == "calculator" for b in blocks)

    @pytest.mark.asyncio
    async def test_pre_tool_use_blocks_execution(self, fake_client_with_tools):
        """PreToolUse hook can block tool execution"""
        async def block_hook(event: PreToolUseEvent) -> HookDecision:
            if event.tool_name == "calculator":
                return HookDecision(continue_=False, reason="Calculator blocked")
            return HookDecision()

        options = AgentOptions(
            system_prompt="Test",
            model="test-model",
            base_url="http://localhost:1234/v1",
            tools=fake_client_with_tools["tools"],
            hooks={HOOK_PRE_TOOL_USE: [block_hook]},
        )

        async with Client(options) as client:
            await client.query("Call the calculator")

            blocks = []
            async for block in client.receive_messages():
                blocks.append(block)

        # Should yield ToolUseError instead of ToolUseBlock
        errors = [b for b in blocks if isinstance(b, ToolUseError)]
        assert len(errors) == 1
        assert "Calculator blocked" in errors[0].error

    @pytest.mark.asyncio
    async def test_pre_tool_use_modifies_input(self, fake_client_with_tools):
        """PreToolUse hook can modify tool input"""
        original_input = None
        modified_input = None

        async def modify_hook(event: PreToolUseEvent) -> HookDecision:
            nonlocal original_input
            original_input = event.tool_input.copy()
            return HookDecision(
                modified_input={"a": 100, "b": 200},
                reason="Input modified"
            )

        options = AgentOptions(
            system_prompt="Test",
            model="test-model",
            base_url="http://localhost:1234/v1",
            tools=fake_client_with_tools["tools"],
            hooks={HOOK_PRE_TOOL_USE: [modify_hook]},
        )

        async with Client(options) as client:
            await client.query("Call the calculator")

            blocks = []
            async for block in client.receive_messages():
                blocks.append(block)
                if isinstance(block, ToolUseBlock):
                    modified_input = block.input

        # Original input should differ from modified
        assert original_input is not None
        assert modified_input is not None
        assert original_input != modified_input
        assert modified_input == {"a": 100, "b": 200}

    @pytest.mark.asyncio
    async def test_pre_tool_use_with_query_function(self, fake_client_with_tools):
        """PreToolUse hook works with standalone query() function"""
        hook_calls = []

        async def track_hook(event: PreToolUseEvent) -> None:
            hook_calls.append(event.tool_name)

        options = AgentOptions(
            system_prompt="Test",
            model="test-model",
            base_url="http://localhost:1234/v1",
            tools=fake_client_with_tools["tools"],
            hooks={HOOK_PRE_TOOL_USE: [track_hook]},
        )

        blocks = []
        async for msg in query("Call calculator", options):
            blocks.extend(msg.content)

        assert len(hook_calls) == 1
        assert hook_calls[0] == "calculator"


class TestPostToolUseHook:
    """Test PostToolUse hook functionality"""

    @pytest.mark.asyncio
    async def test_post_tool_use_observes_result(self, fake_client_with_tools):
        """PostToolUse hook can observe tool execution results"""
        hook_calls = []

        async def observe_hook(event: PostToolUseEvent) -> None:
            hook_calls.append({
                "tool_name": event.tool_name,
                "tool_input": event.tool_input,
                "tool_result": event.tool_result,
            })

        options = AgentOptions(
            system_prompt="Test",
            model="test-model",
            base_url="http://localhost:1234/v1",
            tools=fake_client_with_tools["tools"],
            hooks={HOOK_POST_TOOL_USE: [observe_hook]},
        )

        async with Client(options) as client:
            await client.query("Call calculator")

            # Consume all blocks first
            blocks = []
            async for block in client.receive_messages():
                blocks.append(block)

            # Find tool block and add result AFTER consuming response
            tool_blocks = [b for b in blocks if isinstance(b, ToolUseBlock)]
            assert len(tool_blocks) == 1

            # Simulate tool execution - this triggers the hook
            await client.add_tool_result(
                tool_call_id=tool_blocks[0].id,
                content="Result: 42"
            )

        # Hook should be called when result is added
        assert len(hook_calls) == 1
        assert hook_calls[0]["tool_name"] == "calculator"
        assert hook_calls[0]["tool_result"] == "Result: 42"

    @pytest.mark.asyncio
    async def test_post_tool_use_logs_with_reason(self, fake_client_with_tools):
        """PostToolUse hook can provide reason for logging"""
        async def logging_hook(event: PostToolUseEvent) -> HookDecision:
            return HookDecision(reason="Tool execution logged for audit")

        options = AgentOptions(
            system_prompt="Test",
            model="test-model",
            base_url="http://localhost:1234/v1",
            tools=fake_client_with_tools["tools"],
            hooks={HOOK_POST_TOOL_USE: [logging_hook]},
        )

        async with Client(options) as client:
            await client.query("Call calculator")

            # Consume all blocks first
            blocks = []
            async for block in client.receive_messages():
                blocks.append(block)

            # Find tool block and add result
            tool_blocks = [b for b in blocks if isinstance(b, ToolUseBlock)]
            if tool_blocks:
                await client.add_tool_result(tool_blocks[0].id, "42")

        # No assertion needed - just verify no exceptions


class TestUserPromptSubmitHook:
    """Test UserPromptSubmit hook functionality"""

    @pytest.mark.asyncio
    async def test_user_prompt_submit_allows_query(self, fake_client_no_tools):
        """UserPromptSubmit hook that returns None allows query"""
        hook_calls = []

        async def track_hook(event: UserPromptSubmitEvent) -> None:
            hook_calls.append(event.prompt)

        options = AgentOptions(
            system_prompt="Test",
            model="test-model",
            base_url="http://localhost:1234/v1",
            hooks={HOOK_USER_PROMPT_SUBMIT: [track_hook]},
        )

        async with Client(options) as client:
            await client.query("Hello")
            async for _ in client.receive_messages():
                pass

        assert len(hook_calls) == 1
        assert hook_calls[0] == "Hello"

    @pytest.mark.asyncio
    async def test_user_prompt_submit_blocks_query(self, fake_client_no_tools):
        """UserPromptSubmit hook can block queries"""
        async def block_hook(event: UserPromptSubmitEvent) -> HookDecision:
            if "forbidden" in event.prompt.lower():
                return HookDecision(continue_=False, reason="Forbidden word detected")
            return HookDecision()

        options = AgentOptions(
            system_prompt="Test",
            model="test-model",
            base_url="http://localhost:1234/v1",
            hooks={HOOK_USER_PROMPT_SUBMIT: [block_hook]},
        )

        async with Client(options) as client:
            with pytest.raises(RuntimeError, match="Query blocked by hook"):
                await client.query("This contains forbidden content")

    @pytest.mark.asyncio
    async def test_user_prompt_submit_modifies_prompt(self, fake_client_no_tools):
        """UserPromptSubmit hook can modify prompt"""
        modified_prompts = []

        async def modify_hook(event: UserPromptSubmitEvent) -> HookDecision:
            return HookDecision(
                modified_prompt=event.prompt + " [SANITIZED]",
                reason="Added safety marker"
            )

        options = AgentOptions(
            system_prompt="Test",
            model="test-model",
            base_url="http://localhost:1234/v1",
            hooks={HOOK_USER_PROMPT_SUBMIT: [modify_hook]},
        )

        async with Client(options) as client:
            await client.query("Hello")
            async for _ in client.receive_messages():
                pass

            # Check that modified prompt is in history
            history = client.history
            user_messages = [m for m in history if m.get("role") == "user"]
            assert len(user_messages) == 1
            assert "[SANITIZED]" in user_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_user_prompt_submit_with_query_function(self, fake_client_no_tools):
        """UserPromptSubmit hook works with standalone query() function"""
        hook_calls = []

        async def track_hook(event: UserPromptSubmitEvent) -> None:
            hook_calls.append(event.prompt)

        options = AgentOptions(
            system_prompt="Test",
            model="test-model",
            base_url="http://localhost:1234/v1",
            hooks={HOOK_USER_PROMPT_SUBMIT: [track_hook]},
        )

        async for _ in query("Test prompt", options):
            pass

        assert len(hook_calls) == 1
        assert hook_calls[0] == "Test prompt"


class TestMultipleHooks:
    """Test multiple hooks running in sequence"""

    @pytest.mark.asyncio
    async def test_multiple_hooks_run_in_order(self, fake_client_no_tools):
        """Multiple hooks run sequentially, first decision wins"""
        call_order = []

        async def hook1(event: UserPromptSubmitEvent) -> None:
            call_order.append("hook1")
            return None

        async def hook2(event: UserPromptSubmitEvent) -> HookDecision:
            call_order.append("hook2")
            return HookDecision(continue_=False, reason="Blocked by hook2")

        async def hook3(event: UserPromptSubmitEvent) -> None:
            call_order.append("hook3")  # Should not be called
            return None

        options = AgentOptions(
            system_prompt="Test",
            model="test-model",
            base_url="http://localhost:1234/v1",
            hooks={HOOK_USER_PROMPT_SUBMIT: [hook1, hook2, hook3]},
        )

        async with Client(options) as client:
            with pytest.raises(RuntimeError, match="Blocked by hook2"):
                await client.query("Test")

        # hook1 returns None (continue), hook2 returns decision (short-circuit)
        assert call_order == ["hook1", "hook2"]

    @pytest.mark.asyncio
    async def test_hook_exception_propagates(self, fake_client_no_tools):
        """Exceptions in hooks propagate to caller"""
        async def failing_hook(event: UserPromptSubmitEvent) -> None:
            raise ValueError("Hook failed")

        options = AgentOptions(
            system_prompt="Test",
            model="test-model",
            base_url="http://localhost:1234/v1",
            hooks={HOOK_USER_PROMPT_SUBMIT: [failing_hook]},
        )

        async with Client(options) as client:
            with pytest.raises(ValueError, match="Hook failed"):
                await client.query("Test")


class TestHookEventData:
    """Test that hooks receive correct event data"""

    @pytest.mark.asyncio
    async def test_pre_tool_use_event_has_history(self, fake_client_with_tools):
        """PreToolUse event includes conversation history"""
        received_event = None

        async def capture_hook(event: PreToolUseEvent) -> None:
            nonlocal received_event
            received_event = event

        options = AgentOptions(
            system_prompt="Test",
            model="test-model",
            base_url="http://localhost:1234/v1",
            tools=fake_client_with_tools["tools"],
            hooks={HOOK_PRE_TOOL_USE: [capture_hook]},
        )

        async with Client(options) as client:
            await client.query("First message")
            async for _ in client.receive_messages():
                pass

        assert received_event is not None
        assert isinstance(received_event, PreToolUseEvent)
        assert received_event.tool_name == "calculator"
        assert isinstance(received_event.tool_input, dict)
        assert isinstance(received_event.tool_use_id, str)
        assert isinstance(received_event.history, list)
        # History should have user message
        assert len(received_event.history) >= 1

    @pytest.mark.asyncio
    async def test_post_tool_use_event_has_result(self, fake_client_with_tools):
        """PostToolUse event includes tool result"""
        received_event = None

        async def capture_hook(event: PostToolUseEvent) -> None:
            nonlocal received_event
            received_event = event

        options = AgentOptions(
            system_prompt="Test",
            model="test-model",
            base_url="http://localhost:1234/v1",
            tools=fake_client_with_tools["tools"],
            hooks={HOOK_POST_TOOL_USE: [capture_hook]},
        )

        async with Client(options) as client:
            await client.query("Call calculator")

            # Consume all blocks first
            blocks = []
            async for block in client.receive_messages():
                blocks.append(block)

            # Find tool block and add result
            tool_blocks = [b for b in blocks if isinstance(b, ToolUseBlock)]
            if tool_blocks:
                await client.add_tool_result(tool_blocks[0].id, "Result: 100")

        assert received_event is not None
        assert isinstance(received_event, PostToolUseEvent)
        assert received_event.tool_result == "Result: 100"
