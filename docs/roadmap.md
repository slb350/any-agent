# Open Agent SDK Roadmap

## Current Status (v0.1.0 - Released)

The SDK core is functionally complete with:
- ✅ Streaming query() and multi-turn Client
- ✅ Tool use detection via ToolUseBlock
- ✅ Tool result handling with `Client.add_tool_result()`
- ✅ Proper resource cleanup (no httpx transport leaks)
- ✅ Configuration flexibility (env vars, providers, helpers)
- ✅ URL validation and configurable timeouts
- ✅ Tested with Ollama (kimi-k2:1t-cloud), LM Studio, and llama.cpp servers

## v0.1.0 Release (Published to PyPI)

### ✅ Completed Features
- Tool use loop example (`examples/tool_use_agent.py`) demonstrating `add_tool_result()`
- Provider validation for LM Studio, Ollama, and llama.cpp documented in `docs/provider-compatibility.md`
- PyPI readiness: license, CHANGELOG, README refresh, `pyproject.toml` metadata/classifiers
- Practical showcase agents added (`examples/git_commit_agent.py`, `examples/log_analyzer_agent.py`) and featured in the README
- Published to PyPI as `open-agent-sdk`
- GitHub repository public at https://github.com/slb350/open-agent-sdk

## Near Term (v0.1.1+)

### Future Documentation Improvements
- [ ] Generate API reference from docstrings (`docs/api-reference.md`)
- [ ] Write migration guide from claude-agent-sdk (`docs/migration-guide.md`)
- [ ] Add troubleshooting guide (`docs/troubleshooting.md`)
- [ ] Add GitHub workflow for automated releases

### Provider Coverage
- [ ] Test and document vLLM
- [ ] Test Text Generation WebUI
- [ ] Create provider compatibility matrix

## Medium Term (v0.2.0)

### 1. Conversation Helpers (Priority: Medium)
Optional utilities for common patterns:
```python
# Save/load conversation
await client.save_to_file("chat.json")
client = await Client.from_file("chat.json", options)

# Conversation summary
summary = await client.summarize(max_tokens=100)
```

### 2. Retry & Error Recovery (Priority: Medium)
- Configurable retry strategy for network failures
- Automatic reconnection on stream interruption
- Better error messages with suggested fixes
- **File**: `any_agent/retry.py`

### 3. Advanced Examples (Priority: Low)
- RAG agent with embeddings
- Code generation agent with execution
- Research agent with web search
- Multi-agent coordination example
- **Directory**: `examples/advanced/`

## Long Term (v0.3.0+)

### 1. Performance Optimizations
- Connection pooling for multiple agents
- Response caching (optional)
- Batch request support
- Token usage tracking

### 2. Extended Provider Support
- Anthropic Claude via proxy
- Google Gemini via proxy
- Custom provider interface

### 3. Developer Experience
- CLI tool for testing endpoints
- Config file validation
- Interactive agent builder
- VS Code extension

## Non-Goals (Out of Scope)

The SDK intentionally does NOT provide:
- ❌ Built-in vector stores or RAG
- ❌ Agent orchestration framework
- ❌ Prompt management system
- ❌ Built-in tool implementations
- ❌ Memory/storage backends
- ❌ UI components

These are left to the agent developer to implement based on their specific needs.

## Community Contributions Welcome

Areas where contributions would be valuable:
- Provider compatibility reports
- Example agents for specific use cases
- Documentation improvements
- Bug reports with reproducible tests
- Performance benchmarks

## Success Metrics

The SDK is successful when:
1. Agents can migrate from claude-agent-sdk with minimal changes
2. Works reliably with major local model servers
3. Tool use patterns are clear and documented
4. Resource leaks are eliminated
5. Community adopts for local LLM projects

## Timeline

- **Week 1**: Tool use example, provider testing
- **Week 2**: PyPI release (v0.1.0)
- **Month 2**: v0.2.0 with conversation helpers
- **Month 3**: v0.3.0 with performance optimizations

## Questions for Design Decisions

Before v0.1.0:
1. Should we provide a default retry strategy?
2. Should tool schemas be validated?
3. Should we add telemetry/logging helpers?
4. Should we support function calling syntax?

These decisions will be made based on user feedback during the preview period.
