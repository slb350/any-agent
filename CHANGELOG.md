# Changelog

All notable changes to Open Agent SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-16

### BREAKING CHANGES
- Changed module name from `any_agent` to `open_agent`
- All imports now use `from open_agent import ...` instead of `from any_agent import ...`

### Changed
- Renamed package module to match PyPI package name for consistency
- Updated all examples and documentation to use new import path
- Better alignment between package name (open-agent-sdk) and module name (open_agent)

### Migration Guide
If upgrading from v0.1.0:
```python
# Old (v0.1.0)
from any_agent import query, AgentOptions

# New (v0.2.0)
from open_agent import query, AgentOptions
```

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

- **0.1.0** - Initial public release (2025-10-16)
