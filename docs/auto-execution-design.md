# Tool Auto-Execution Implementation Plan

## Branch: tool-auto

## Status: Ready to implement

## Overview

Add automatic tool execution to reduce boilerplate for simple agents while maintaining full visibility and hook support. Users who want automatic execution set `auto_execute_tools=True` in AgentOptions, and the SDK handles the execute-and-continue loop automatically.

## Current Manual Flow (Status Quo)

```python
# User must manually:
# 1. Detect ToolUseBlock
# 2. Look up tool from registry
# 3. Execute tool
# 4. Add result to history
# 5. Continue conversation with empty prompt
# 6. Read final response

async with Client(options) as client:
    await client.query("What's 25 + 17?")

    async for block in client.receive_messages():
        if isinstance(block, ToolUseBlock):
            tool = tool_registry[block.name]
            result = await tool.execute(block.input)
            await client.add_tool_result(block.id, result, name=block.name)
            await client.query("")  # Continue
            async for response in client.receive_messages():
                if hasattr(response, 'text'):
                    print(response.text)
```

## Proposed Auto-Execution Flow

```python
options = AgentOptions(
    system_prompt="You are a calculator assistant",
    model="qwen2.5-32b-instruct",
    base_url="http://localhost:1234/v1",
    tools=[add, subtract, multiply, divide],
    auto_execute_tools=True,      # NEW: Enable auto-execution
    max_tool_iterations=5,         # NEW: Safety limit (default 5)
)

async with Client(options) as client:
    await client.query("What's 25 + 17?")

    # SDK automatically:
    # - Detects ToolUseBlocks
    # - Executes tools
    # - Adds results to history
    # - Continues conversation
    # - Loops until text-only response

    async for block in client.receive_messages():
        if isinstance(block, ToolUseBlock):
            # Still yielded for visibility/logging!
            print(f"🔧 Tool used: {block.name}")
        elif isinstance(block, TextBlock):
            # Final answer after all tools executed
            print(f"Answer: {block.text}")
```

## Design Decisions (FINALIZED)

### 1. API Changes to AgentOptions

```python
@dataclass
class AgentOptions:
    # ... existing fields ...
    auto_execute_tools: bool = False           # NEW: Enable auto-execution (default False)
    max_tool_iterations: int = 5               # NEW: Safety limit (default 5, not 10)
```

**Rationale for max_tool_iterations=5:**
- User feedback: "default to something low (3–5) so we fail fast if the model loops"
- 5 is reasonable for most multi-step tasks
- Callers needing more can explicitly raise it
- Better to fail fast than loop endlessly

### 2. Tool Registry (Build Once at __init__)

```python
class Client:
    def __init__(self, options: AgentOptions):
        # ... existing initialization ...

        # NEW: Build tool registry once
        self._tool_registry: dict[str, Tool] = {}
        if options.tools:
            for tool in options.tools:
                if tool.name in self._tool_registry:
                    raise ValueError(f"Duplicate tool name: {tool.name}")
                self._tool_registry[tool.name] = tool
```

**Rationale:**
- Build once, not per-loop (performance)
- Validate unique names upfront (clear error message)
- Quick O(1) lookups during execution
- Fail fast on duplicate names

### 3. Refactor receive_messages() Architecture

**Current structure:**
```python
async def receive_messages(self):
    # ... entire implementation inline ...
```

**New structure:**
```python
async def receive_messages(self):
    """Public API - dispatches to auto or manual mode"""
    if self.options.auto_execute_tools:
        async for block in self._auto_execute_loop():
            yield block
    else:
        async for block in self._receive_once():
            yield block

async def _receive_once(self):
    """Private helper - one turn of receiving messages"""
    # ... move current receive_messages() body here ...
    # This is the "manual" flow we have now
```

**Rationale:**
- Keeps iterator contract identical (no breaking changes)
- Clear separation: auto vs manual mode
- `_receive_once()` is reusable for both paths
- Ensures history updates happen exactly once per turn

### 4. Continuation Method (_continue_turn)

```python
async def _continue_turn(self):
    """
    Continue the conversation after tool execution without adding new user message.

    IMPORTANT: Does NOT fire UserPromptSubmit hooks (already fired on initial query).
    IMPORTANT: Does NOT add new user message to history.
    IMPORTANT: Reuses existing conversation context.
    """
    messages = format_messages(
        self.options.system_prompt,
        "",  # Empty prompt for continuation
        self.message_history  # Already includes tool results
    )

    request_params: dict[str, Any] = {
        "model": self.options.model,
        "messages": messages,
        "max_tokens": self.options.max_tokens,
        "temperature": self.options.temperature,
        "stream": True,
    }

    if self.options.tools:
        request_params["tools"] = format_tools(self.options.tools)

    try:
        response_stream = await self.client.chat.completions.create(**request_params)
    except Exception:
        self.response_stream = None
        self._aggregator = None
        raise

    self.response_stream = response_stream
    self._aggregator = ToolCallAggregator()
```

**Rationale (from user feedback):**
- Don't use `await self.query("")` - confuses models and re-fires hooks
- Create dedicated method for continuation
- Reuses last user entry implicitly via history
- Skips UserPromptSubmit hooks (already fired)
- Preserves message_history correctly
- Clearer intent than empty string prompt

### 5. Auto-Execution Loop Implementation

```python
async def _auto_execute_loop(self):
    """
    Automatically execute tools until we get a text-only response.

    Yields:
        - TextBlock: Text from assistant
        - ToolUseBlock: Tool call (before execution, for visibility)
        - ToolUseError: Execution errors or hook blocks

    Loop terminates when:
        - No tools in response (text-only answer)
        - max_tool_iterations reached
    """
    iteration = 0

    while iteration < self.options.max_tool_iterations:
        has_tools = False

        # Get response from model (one turn)
        async for block in self._receive_once():
            yield block  # Always yield for visibility

            if isinstance(block, ToolUseBlock):
                has_tools = True

                # Look up tool
                tool = self._tool_registry.get(block.name)
                if not tool:
                    # Unknown tool - yield error and continue
                    error = ToolUseError(
                        error=f"Unknown tool: {block.name}",
                        raw_data=str(block.input)
                    )
                    yield error

                    # Add error result to history so model sees it
                    await self.add_tool_result(
                        tool_call_id=block.id,
                        content={"error": f"Unknown tool: {block.name}"},
                        name=block.name
                    )
                    continue

                # Execute tool
                try:
                    result = await tool.execute(block.input)
                    await self.add_tool_result(block.id, result, name=block.name)
                except Exception as e:
                    # Tool execution failed - yield error and add to history
                    error = ToolUseError(error=str(e), raw_data=str(block.input))
                    yield error

                    # Add error result to history so model sees it
                    await self.add_tool_result(
                        tool_call_id=block.id,
                        content={"error": str(e)},
                        name=block.name
                    )

        # Check if we're done
        if not has_tools:
            # No tools in response - we got final answer
            break

        # Continue conversation for next turn
        await self._continue_turn()
        iteration += 1

    if iteration >= self.options.max_tool_iterations:
        logger.warning(
            f"Hit max_tool_iterations ({iteration}). "
            "Increase max_tool_iterations if this is expected."
        )
```

**Key implementation notes:**
1. **Always yield ToolUseBlocks** - even in auto mode, for logging/monitoring
2. **Tool execution errors** - both yield ToolUseError AND add error to history
3. **Unknown tools** - same pattern (yield error, add to history)
4. **PreToolUse hook blocks** - `_receive_once()` already yields ToolUseError, we don't execute
5. **PostToolUse hook** - fires in `add_tool_result()` as usual
6. **Max iterations** - warn but don't raise (graceful degradation)

### 6. Hook Integration (NO CHANGES NEEDED)

**PreToolUse:**
- Already fires in `_receive_once()` before yielding ToolUseBlock
- If blocked, `_receive_once()` yields ToolUseError
- Auto-execution loop sees ToolUseError, doesn't attempt execution
- ✅ Works automatically

**PostToolUse:**
- Fires in `add_tool_result()` when we add tool result to history
- Auto-execution calls `add_tool_result()` same as manual
- ✅ Works automatically

**UserPromptSubmit:**
- Fires in initial `query()` call
- Does NOT fire in `_continue_turn()` (by design)
- ✅ Works automatically

**Conclusion:** Hooks require ZERO changes. Beautiful!

### 7. Error Handling Strategy

| Error Case | Action | History Entry | Loop Continues? |
|------------|--------|---------------|-----------------|
| Unknown tool | Yield ToolUseError | Add error result | Yes |
| Tool execution exception | Yield ToolUseError | Add error result | Yes |
| PreToolUse hook blocks | Yield ToolUseError (from _receive_once) | No tool result added | Yes |
| Max iterations hit | Log warning | N/A | No |
| Network error | Propagate exception | N/A | No |

**Key principle:** Tool errors are recoverable (continue loop). Network/API errors are not.

### 8. Standalone query() - NO AUTO-EXECUTION

**Decision:** Do NOT add auto-execution to standalone `query()` function.

**Rationale (from user feedback):**
- `query()` is for single-shot requests
- Needing tools in auto mode is a sign you should use `Client`
- Keeps `query()` simple and focused
- Multi-turn tool loops belong in stateful `Client`

## Breaking Changes

**NONE!**
- Default `auto_execute_tools=False` preserves all existing behavior
- New fields are optional
- All existing code works unchanged

## Implementation Checklist

### Phase 1: Core Implementation
- [ ] Add `auto_execute_tools` and `max_tool_iterations` to `AgentOptions` in types.py
- [ ] Build `_tool_registry` in `Client.__init__()` with duplicate name validation
- [ ] Rename current `receive_messages()` body to `_receive_once()`
- [ ] Implement `_continue_turn()` method (no hooks, no new user message)
- [ ] Implement `_auto_execute_loop()` with error handling
- [ ] Update `receive_messages()` to dispatch to auto vs manual mode

### Phase 2: Testing
- [ ] Unit test: Tool registry validation (duplicate names)
- [ ] Unit test: `_auto_execute_loop()` with fake tools
  - One successful tool
  - One exception during execution
  - One unknown tool name
- [ ] Integration test: Auto-execution with real Client
  - Multiple tool calls
  - Mixed text and tool responses
  - History consistency check
- [ ] Hook regression tests:
  - PreToolUse blocking prevents execution
  - PostToolUse sees actual results
  - UserPromptSubmit fires once, not on continuation
- [ ] Max iteration guard test:
  - Model loops infinitely
  - Verify warning logged
  - Verify stops at limit
- [ ] Backward compatibility test:
  - `auto_execute_tools=False` behaves identically to v0.2.4
  - All existing tests pass unchanged

### Phase 3: Documentation
- [ ] Update README.md with auto-execution example
- [ ] Update docs/features.md (mark as completed)
- [ ] Update docs/technical-design.md with auto-execution section
- [ ] Create or update example: `examples/calculator_auto.py`
- [ ] Update CHANGELOG.md for v0.2.5

### Phase 4: Examples
- [ ] Create `examples/calculator_auto.py` showing auto-execution
- [ ] Update `examples/calculator_tools.py` to show manual mode (preserve as reference)
- [ ] Add comment in examples explaining when to use each mode

## File Changes Required

### open_agent/types.py
- Add two fields to `AgentOptions` dataclass

### open_agent/client.py
- Add `_tool_registry` initialization in `__init__()`
- Refactor `receive_messages()` → `_receive_once()`
- Implement `_continue_turn()`
- Implement `_auto_execute_loop()`
- Update `receive_messages()` to dispatch

### tests/test_client.py
- Add tool registry validation tests
- Add auto-execution unit tests
- Add backward compatibility tests

### tests/test_hooks.py (minimal changes)
- Add test: hooks work with auto-execution
- Add test: UserPromptSubmit doesn't fire on continuation

### tests/integration/test_auto_execution.py (NEW FILE)
- Integration tests with fake AsyncOpenAI
- Multi-turn tool execution scenarios
- Error handling scenarios

### examples/calculator_auto.py (NEW FILE)
- Demonstrates auto-execution mode
- Shows before/after comparison

### docs/
- Update README.md, features.md, technical-design.md
- Update CHANGELOG.md

## Testing Strategy (Detailed)

### Unit Tests (tests/test_client.py)

**Test 1: Tool registry validation**
```python
def test_client_rejects_duplicate_tool_names():
    tool1 = tool("add", "Add", {"a": int, "b": int})(lambda args: args["a"] + args["b"])
    tool2 = tool("add", "Add2", {"x": int, "y": int})(lambda args: args["x"] + args["y"])

    options = AgentOptions(
        system_prompt="test",
        model="test",
        base_url="http://test",
        tools=[tool1, tool2]
    )

    with pytest.raises(ValueError, match="Duplicate tool name: add"):
        Client(options)
```

**Test 2: Auto-execution with successful tool**
```python
@pytest.mark.asyncio
async def test_auto_execute_successful_tool(fake_openai):
    # Setup: Model returns tool call, then text response
    fake_openai["enqueue"](
        tool_call_chunks(tool_id="call-1", name="add", arguments='{"a": 25, "b": 17}'),
        text_chunks("The answer is 42")
    )

    add_tool = tool("add", "Add", {"a": int, "b": int})(lambda args: {"result": args["a"] + args["b"]})

    options = AgentOptions(
        system_prompt="test",
        model="test",
        base_url="http://test",
        tools=[add_tool],
        auto_execute_tools=True
    )

    async with Client(options) as client:
        await client.query("Calculate 25 + 17")

        blocks = []
        async for block in client.receive_messages():
            blocks.append(block)

        # Should yield: ToolUseBlock, TextBlock
        assert len(blocks) == 2
        assert isinstance(blocks[0], ToolUseBlock)
        assert blocks[0].name == "add"
        assert isinstance(blocks[1], TextBlock)
        assert "42" in blocks[1].text
```

**Test 3: Tool execution exception**
```python
@pytest.mark.asyncio
async def test_auto_execute_tool_exception(fake_openai):
    fake_openai["enqueue"](
        tool_call_chunks(tool_id="call-1", name="divide", arguments='{"a": 10, "b": 0}'),
        text_chunks("Cannot divide by zero")
    )

    def divide_handler(args):
        if args["b"] == 0:
            raise ValueError("Division by zero")
        return {"result": args["a"] / args["b"]}

    divide_tool = tool("divide", "Divide", {"a": int, "b": int})(divide_handler)

    options = AgentOptions(
        system_prompt="test",
        model="test",
        base_url="http://test",
        tools=[divide_tool],
        auto_execute_tools=True
    )

    async with Client(options) as client:
        await client.query("Divide 10 by 0")

        blocks = []
        async for block in client.receive_messages():
            blocks.append(block)

        # Should yield: ToolUseBlock, ToolUseError, TextBlock
        assert len(blocks) == 3
        assert isinstance(blocks[0], ToolUseBlock)
        assert isinstance(blocks[1], ToolUseError)
        assert "Division by zero" in blocks[1].error
        assert isinstance(blocks[2], TextBlock)
```

**Test 4: Unknown tool name**
```python
@pytest.mark.asyncio
async def test_auto_execute_unknown_tool(fake_openai):
    fake_openai["enqueue"](
        tool_call_chunks(tool_id="call-1", name="nonexistent", arguments='{}'),
        text_chunks("Tool not found")
    )

    options = AgentOptions(
        system_prompt="test",
        model="test",
        base_url="http://test",
        tools=[],  # No tools registered
        auto_execute_tools=True
    )

    async with Client(options) as client:
        await client.query("Use nonexistent tool")

        blocks = []
        async for block in client.receive_messages():
            blocks.append(block)

        assert len(blocks) == 3
        assert isinstance(blocks[0], ToolUseBlock)
        assert isinstance(blocks[1], ToolUseError)
        assert "Unknown tool: nonexistent" in blocks[1].error
```

**Test 5: Max iterations limit**
```python
@pytest.mark.asyncio
async def test_auto_execute_max_iterations(fake_openai):
    # Model keeps calling same tool infinitely
    for _ in range(10):  # Enqueue 10 tool calls
        fake_openai["enqueue"](
            tool_call_chunks(tool_id=f"call-{_}", name="loop", arguments='{}')
        )

    loop_tool = tool("loop", "Loop", {})(lambda args: {"status": "looping"})

    options = AgentOptions(
        system_prompt="test",
        model="test",
        base_url="http://test",
        tools=[loop_tool],
        auto_execute_tools=True,
        max_tool_iterations=3
    )

    async with Client(options) as client:
        await client.query("Start loop")

        blocks = []
        async for block in client.receive_messages():
            blocks.append(block)

        # Should stop at 3 iterations (not 10)
        tool_blocks = [b for b in blocks if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) == 3
```

### Hook Integration Tests (tests/test_hooks.py)

**Test 6: PreToolUse blocks execution in auto mode**
```python
@pytest.mark.asyncio
async def test_pre_tool_use_blocks_auto_execution(fake_openai):
    fake_openai["enqueue"](
        tool_call_chunks(tool_id="call-1", name="dangerous", arguments='{}'),
        text_chunks("Operation blocked")
    )

    async def block_hook(event):
        if event.tool_name == "dangerous":
            return HookDecision(continue_=False, reason="Blocked")
        return None

    dangerous_tool = tool("dangerous", "Dangerous", {})(lambda args: {"result": "executed"})

    options = AgentOptions(
        system_prompt="test",
        model="test",
        base_url="http://test",
        tools=[dangerous_tool],
        auto_execute_tools=True,
        hooks={HOOK_PRE_TOOL_USE: [block_hook]}
    )

    async with Client(options) as client:
        await client.query("Use dangerous tool")

        blocks = []
        async for block in client.receive_messages():
            blocks.append(block)

        # Should yield: ToolUseBlock, ToolUseError (from hook), TextBlock
        # Tool should NOT execute (no add_tool_result call)
        assert len(blocks) == 3
        assert isinstance(blocks[1], ToolUseError)
        assert "Blocked" in blocks[1].error
```

**Test 7: UserPromptSubmit fires once, not on continuation**
```python
@pytest.mark.asyncio
async def test_user_prompt_submit_fires_once(fake_openai):
    fake_openai["enqueue"](
        tool_call_chunks(tool_id="call-1", name="add", arguments='{"a": 1, "b": 2}'),
        text_chunks("Result: 3")
    )

    hook_calls = []

    async def track_hook(event):
        hook_calls.append(event.prompt)
        return None

    add_tool = tool("add", "Add", {"a": int, "b": int})(lambda args: {"result": args["a"] + args["b"]})

    options = AgentOptions(
        system_prompt="test",
        model="test",
        base_url="http://test",
        tools=[add_tool],
        auto_execute_tools=True,
        hooks={HOOK_USER_PROMPT_SUBMIT: [track_hook]}
    )

    async with Client(options) as client:
        await client.query("Calculate 1 + 2")

        async for _ in client.receive_messages():
            pass

        # Hook should fire exactly once (initial query, not continuation)
        assert len(hook_calls) == 1
        assert hook_calls[0] == "Calculate 1 + 2"
```

### Backward Compatibility Test

**Test 8: auto_execute_tools=False behaves identically to v0.2.4**
```python
@pytest.mark.asyncio
async def test_manual_mode_unchanged(fake_openai):
    fake_openai["enqueue"](
        tool_call_chunks(tool_id="call-1", name="add", arguments='{"a": 1, "b": 2}')
    )

    add_tool = tool("add", "Add", {"a": int, "b": int})(lambda args: {"result": args["a"] + args["b"]})

    options = AgentOptions(
        system_prompt="test",
        model="test",
        base_url="http://test",
        tools=[add_tool],
        auto_execute_tools=False  # Explicit manual mode
    )

    async with Client(options) as client:
        await client.query("Calculate 1 + 2")

        blocks = []
        async for block in client.receive_messages():
            blocks.append(block)

        # Should yield only ToolUseBlock (no auto-execution)
        assert len(blocks) == 1
        assert isinstance(blocks[0], ToolUseBlock)

        # User must manually execute and continue
        result = await add_tool.execute(blocks[0].input)
        await client.add_tool_result(blocks[0].id, result, name="add")

        # History should have tool result
        assert len(client.history) == 3  # user, assistant, tool
```

## Example Code Updates

### examples/calculator_auto.py (NEW FILE)

```python
#!/usr/bin/env python3
"""
Example: Calculator with Auto-Execution

Demonstrates automatic tool execution - the SDK handles the execute-and-continue
loop automatically, reducing boilerplate.

Compare this to calculator_tools.py (manual mode) to see the difference.

Usage:
    python examples/calculator_auto.py
"""

import asyncio
from open_agent import tool, Client, AgentOptions, ToolUseBlock, TextBlock

# Define calculator tools
@tool("add", "Add two numbers", {"a": float, "b": float})
async def add(args):
    return {"result": args["a"] + args["b"]}

@tool("subtract", "Subtract two numbers", {"a": float, "b": float})
async def subtract(args):
    return {"result": args["a"] - args["b"]}

@tool("multiply", "Multiply two numbers", {"a": float, "b": float})
async def multiply(args):
    return {"result": args["a"] * args["b"]}

@tool("divide", "Divide two numbers", {"a": float, "b": float})
async def divide(args):
    if args["b"] == 0:
        raise ValueError("Cannot divide by zero")
    return {"result": args["a"] / args["b"]}


async def main():
    print("Calculator Agent with AUTO-EXECUTION")
    print("=" * 50)
    print()

    # Configure agent with auto-execution enabled
    options = AgentOptions(
        system_prompt=(
            "You are a helpful calculator assistant. "
            "Use the provided tools to perform calculations. "
            "Always show your work and explain the result."
        ),
        model="qwen2.5-32b-instruct",
        base_url="http://localhost:1234/v1",
        tools=[add, subtract, multiply, divide],
        auto_execute_tools=True,      # 🎯 Enable auto-execution
        max_tool_iterations=5,         # Safety limit
        temperature=0.1,
    )

    # Example calculations
    queries = [
        "What is 25 plus 17?",
        "Calculate 144 divided by 12",
        "What's 7 times 8, then add 5?",  # Multi-step!
    ]

    for query in queries:
        print(f"User: {query}")
        print("-" * 50)

        async with Client(options) as client:
            await client.query(query)

            # SDK automatically handles:
            # - Tool execution
            # - Adding results to history
            # - Continuing conversation
            # You just read the blocks!

            async for block in client.receive_messages():
                if isinstance(block, ToolUseBlock):
                    # Still see tool calls for visibility
                    print(f"🔧 Using tool: {block.name}")
                    print(f"   Arguments: {block.input}")
                elif isinstance(block, TextBlock):
                    # Final answer after tools executed
                    print(f"\nAssistant: {block.text}")

        print()
        print("=" * 50)
        print()


if __name__ == "__main__":
    asyncio.run(main())
```

## Version & Release

**Target Version:** v0.2.5

**Release Type:** Minor (new feature, backward compatible)

**Git Workflow:**
1. Implement on `tool-auto` branch
2. All tests pass (118 existing + new auto-execution tests)
3. Merge to `main`
4. Tag `v0.2.5`
5. Publish to PyPI

## Success Criteria

- [ ] All 118 existing tests pass unchanged
- [ ] New auto-execution tests pass (at least 8 new tests)
- [ ] Example `calculator_auto.py` runs successfully
- [ ] README.md clearly explains auto vs manual mode
- [ ] Zero breaking changes (default behavior preserved)
- [ ] Performance: No significant overhead in manual mode
- [ ] Documentation: Users understand when to use each mode

## Questions for Review (Already Answered)

1. ✅ Max iterations default? **Answer: 5 (not 10, fail fast)**
2. ✅ Build registry when? **Answer: Once in __init__, validate unique names**
3. ✅ Empty prompt for continuation? **Answer: No, use _continue_turn() method**
4. ✅ Standalone query() auto mode? **Answer: No, only Client**
5. ✅ Error handling strategy? **Answer: Both yield ToolUseError AND add to history**

## Post-Implementation TODO

After merging to main:
- [ ] Update docs/features.md (mark auto-execution as completed)
