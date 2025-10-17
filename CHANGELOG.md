# Changelog

All notable changes to Open Agent SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-10-17

### Added
- **Automatic Tool Execution** - Major quality-of-life improvement for tool-using agents
  - `auto_execute_tools=True` flag in `AgentOptions` - tools execute automatically without boilerplate
  - `max_tool_iterations` parameter (default 5) - safety limit to prevent infinite tool loops
  - Tools execute and results feed back automatically until text-only response
  - Simplifies code significantly - no manual `add_tool_result()` + `query("")` loop needed
  - Works seamlessly with hooks - PreToolUse/PostToolUse fire before/after each execution
  - Error handling: Unknown tools and execution failures yield `ToolUseError` for monitoring
  - Backward compatible: defaults to `False` (manual mode) to preserve existing behavior
- Internal improvements:
  - `_tool_registry` built at Client init with duplicate name validation
  - `_continue_turn()` method for hook-free conversation continuation
  - `_receive_once()` helper ensures stream fully consumed before tool execution
  - `_auto_execute_loop()` orchestrates automatic execution with proper error handling
- Test coverage: 10 comprehensive tests in `tests/test_auto_execution.py` (128 total)
  - Tool registry validation (duplicate detection)
  - Auto-execution flow (basic, hooks, errors)
  - Max iterations safety limit
  - Unknown tool handling
  - Execution error handling
  - Backward compatibility (manual mode)
  - Mixed mode scenarios

### Changed
- Updated `examples/calculator_tools.py` to demonstrate both automatic and manual modes
  - Leads with automatic execution (recommended)
  - Shows manual mode as advanced option
  - Consolidated from separate examples for clarity
- Error payloads now pass structured dicts (not JSON strings) to `add_tool_result()`
  - `{"error": "...", "tool": name}` format for better hook integration
  - PostToolUse hooks receive structured objects
- Updated README "Function Calling with Tools" section
  - Leads with automatic execution pattern
  - Manual mode shown as advanced option
  - Updated API Reference to include new parameters

### Fixed
- `_continue_turn()` now uses empty prompt instead of replaying user's original question
  - Prevents confusing the model with repeated queries
  - Mirrors manual `query("")` pattern correctly

### Technical
- All 128 tests passing (10 new auto-execution tests)
- Clean separation: auto mode via `_auto_execute_loop()`, manual via original `receive_messages()` flow
- Tool execution fully consumed before continuation (prevents message interleaving)
- Hooks work identically in both auto and manual modes
- Safety-first design: fail-fast with low default iteration limit

### Design Philosophy
- **Pit of success**: Auto-execution is what most users want, now it's the easiest path
- **Manual when needed**: Advanced users keep full control for custom execution logic
- **Safety limits**: `max_tool_iterations` prevents runaway loops
- **Hook integration**: PreToolUse can still block/modify, PostToolUse still observes
- **Backward compatible**: Existing code continues working unchanged

## [0.2.4] - 2025-10-17

### Added
- **Hooks System** - Pythonic lifecycle hooks for monitoring and controlling agent execution
  - `PreToolUse` hook - Intercept tool execution before it happens
    - Block dangerous operations
    - Modify tool inputs (e.g., redirect paths, sanitize parameters)
    - Security gates and approval workflows
  - `PostToolUse` hook - Monitor tool results after execution
    - Audit logging and compliance tracking
    - Error monitoring and alerting
    - Result validation
  - `UserPromptSubmit` hook - Intercept user input before processing
    - Input sanitization and validation
    - Content filtering
    - Automatic safety instructions
  - Pythonic design: Clean event dataclasses (`PreToolUseEvent`, `PostToolUseEvent`, `UserPromptSubmitEvent`)
  - Simple return types: Return `None` to continue, `HookDecision` to control/modify, raise to abort
  - No JSON envelopes or matcher DSL - handlers branch inside coroutines
  - Sequential execution: First non-None decision wins (short-circuit behavior)
  - Works with both `Client` and standalone `query()` function
- New types exported:
  - `PreToolUseEvent`, `PostToolUseEvent`, `UserPromptSubmitEvent`, `HookEvent`
  - `HookDecision`, `HookHandler`
  - Constants: `HOOK_PRE_TOOL_USE`, `HOOK_POST_TOOL_USE`, `HOOK_USER_PROMPT_SUBMIT`
- New example: `examples/hooks_example.py` - 4 comprehensive patterns
  - Security gates (blocking/redirecting dangerous operations)
  - Audit logging (compliance tracking)
  - Input sanitization (validation and safety)
  - Combined hooks (layered control)
- Test coverage: 14 comprehensive tests in `tests/test_hooks.py` (118 total)
  - PreToolUse tests (allow, block, modify input)
  - PostToolUse tests (observe results, logging)
  - UserPromptSubmit tests (allow, block, modify prompt)
  - Multiple hooks sequencing
  - Exception handling
  - Event data validation
  - Works with both Client and query() contexts

### Changed
- `Client.add_tool_result()` is now async (breaking change for PostToolUse hook support)
  - Old: `client.add_tool_result(tool_id, result)`
  - New: `await client.add_tool_result(tool_id, result)`
- Updated all existing tests to use async `add_tool_result()`
- Updated integration tests for async compatibility

### Technical
- Hooks run inline on the event loop - spawn tasks for heavy work
- No blocking I/O in hook handlers (document this for users)
- Clean integration with existing streaming and tool execution flow
- All 118 tests passing (14 new hook tests)
- Local-first design: No CLI subprocess, no control protocol overhead

### Design Philosophy
- **Claude parity, local-first**: Familiar patterns without CLI complexity
- **Pythonic over protocol**: Dataclasses and coroutines, not JSON messages
- **Inline execution**: Hooks run synchronously in the event loop
- **Explicit control**: Users decide what to monitor and when to intervene
- **Production-ready**: Essential for logging, monitoring, security in real agents

## [0.2.3] - 2025-10-17

### Added
- **Context Management Utilities (Opt-In)** - Manual history management helpers
  - `estimate_tokens(messages, model)` - Token counting with tiktoken (~90% accurate) + character-based fallback (~75-85%)
  - `truncate_messages(messages, keep, preserve_system)` - Simple truncation utility preserving system prompt
  - Optional `[context]` dependency for better accuracy: `pip install open-agent-sdk[context]`
  - Helper functions `_iter_all_strings()` and `_iter_string_values()` for clean recursive string extraction
  - Conservative token estimation using `math.ceil()` (rounds up to prevent underestimation)
- New example: `examples/context_management.py` - 4 detailed usage patterns
  - Pattern 1: Stateless agents (recommended for single tasks)
  - Pattern 2: Manual truncation at natural breakpoints
  - Pattern 3: Token budget monitoring with periodic checks
  - Pattern 4: External memory (RAG-lite pattern)
- Test coverage: 19 comprehensive tests in `tests/test_context.py` (104 total)
  - Token estimation tests (tiktoken + fallback modes)
  - Truncation tests (edge cases, system preservation)
  - Integration tests with realistic workflows
  - Tests for nested tool/function arguments

### Changed
- Updated README with "Context Management" section
- Enhanced `docs/technical-design.md` with Section 5 (context.py) and Design Decision #8
- Moved internal design docs to `archive/` folder (not published with package)

### Design Philosophy
- **Opt-in, not automatic**: No silent mutations of conversation history
- **User control**: Users decide when and how to manage context
- **Domain-specific strategies**: Copy editing ≠ research agents ≠ code reviewers
- **Explicit over implicit**: Manual management empowers power users
- **Intentionally NOT building automatic compaction** due to:
  - Token accuracy varies by model family (Qwen, Llama, Mistral have different tokenizers)
  - Risk of silently breaking context or tool chains
  - Natural model limits (8k-32k tokens) exist regardless
  - Users understand their domain better than generic heuristics

### Technical
- All 104 tests passing (19 new context tests)
- Clean implementation: 170 LOC for context.py
- Optional dependency integration (tiktoken)
- Comprehensive documentation (4 patterns, design rationale)

## [0.2.2] - 2025-10-17

### Added
- **Tool System (Phase 1 Complete)** - First-class function calling support
  - `@tool` decorator for defining tools with clean API
  - Automatic Python type to JSON Schema conversion (`str`, `int`, `float`, `bool`, `list`, `dict`)
  - Support for simple schemas (`{"param": str}`) and full JSON Schema
  - `Tool` class with `execute()` and `to_openai_format()` methods
  - `AgentOptions.tools` field for registering tools
  - `ToolResultBlock` type for structured tool results
  - Sync handler support - automatically wraps non-async functions
  - Optional parameter handling with 3 methods:
    - JSON Schema `default` field
    - Explicit `required: False` flag
    - Convenience `optional: True` flag
  - `format_tools()` utility for OpenAI API format conversion
- New examples:
  - `examples/simple_tool.py` - Minimal tool usage example
  - `examples/calculator_tools.py` - Full calculator with 4 tools (add/subtract/multiply/divide)
- Test coverage:
  - `tests/test_tools.py` - 16 comprehensive tests
  - Type conversion tests
  - Schema conversion tests (simple, mixed, JSON Schema, empty, optional)
  - Tool decorator tests (creation, execution, complex schema, sync handlers)
  - OpenAI format tests
  - Error handling tests

### Changed
- Updated README with "Function Calling with Tools" section
- Enhanced API Reference documentation with Tool system details
- Updated project structure documentation
- Client and query functions now send tools to API when configured
- AssistantMessage content now includes ToolResultBlock

### Technical
- Tools integrate seamlessly with streaming and multi-turn conversations
- Direct OpenAI tools API (no MCP dependency for simplicity)
- All 85 tests passing (16 new tool tests)
- Production-ready quality with comprehensive edge case coverage

## [0.2.1] - 2025-10-16

### Fixed
- Corrected all remaining references to old `ANY_AGENT_*` environment variables
- Updated to use `OPEN_AGENT_*` consistently throughout documentation and examples
- Simplified CHANGELOG to properly reflect v0.1.0 as the initial release

## [0.1.0] - 2025-10-16

### Added
- Initial SDK implementation with `query()` and `Client` class
- Multi-turn conversation support with context retention
- Tool use detection via `ToolUseBlock` streaming
- `Client.add_tool_result()` method for feeding tool outputs back to model
- Configuration flexibility with environment variables and provider shortcuts
- Config helpers: `get_model()` and `get_base_url()`
- Provider support for LM Studio, Ollama, llama.cpp, vLLM
- Basic URL validation for `base_url` parameter
- Configurable timeout parameter (default 60 seconds)
- Proper resource cleanup - no httpx transport leaks
- Comprehensive examples including tool use patterns
- Practical agent examples: Git Commit Agent and Log Analyzer Agent
- Provider compatibility documentation with verified configurations
- Test coverage for core functionality
- Verified compatibility with llama.cpp, LM Studio, and Ollama

### Changed
- `AgentOptions` now requires explicit `model` and `base_url` parameters
- AsyncOpenAI client properly closed after each request
- User messages only added to history after successful API call
- README reshaped with real-world agent walkthroughs and configuration guidance
- `get_model()` now prioritizes environment overrides while allowing explicit fallbacks
- Updated documentation to reflect streaming semantics and provider verification
- Renamed project to **Open Agent SDK**, including environment variables (`OPEN_AGENT_*`) and config file defaults
- Module import path is now `open_agent` (replaces legacy `any_agent` name from pre-release builds)

### Fixed
- Resource leaks from unclosed AsyncOpenAI clients
- Corrupted conversation history on network failures
- Streaming duplication with accumulative providers such as LM Studio
- Tool call aggregation resilience when arguments arrive incrementally

### Security
- Added `_safe_eval()` in examples to replace unsafe `eval()`
- URL validation prevents malformed endpoints

---

## Version History

- **0.2.4** - Hooks system for lifecycle monitoring and control (2025-10-17)
- **0.2.3** - Context management utilities (opt-in, manual) (2025-10-17)
- **0.2.2** - Tool system with @tool decorator (2025-10-17)
- **0.2.1** - Environment variable consistency fix (2025-10-16)
- **0.2.0** - Module rename to `open_agent` (2025-10-16)
- **0.1.0** - Initial public release (2025-10-16)
