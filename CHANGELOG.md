# Changelog

All notable changes to Any-Agent SDK will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Test coverage for core functionality
- Verified compatibility with llama.cpp, LM Studio, and Ollama

### Changed
- `AgentOptions` now requires explicit `model` and `base_url` parameters
- AsyncOpenAI client properly closed after each request
- User messages only added to history after successful API call

### Fixed
- Resource leaks from unclosed AsyncOpenAI clients
- Corrupted conversation history on network failures

### Security
- Added `_safe_eval()` in examples to replace unsafe `eval()`
- URL validation prevents malformed endpoints

## [0.1.0] - TBD

Initial release - see Unreleased section above.

---

## Version History

- **0.1.0** - Initial public release (upcoming)