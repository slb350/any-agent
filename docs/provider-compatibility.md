# Provider Compatibility

This document tracks real-world testing results with different OpenAI-compatible providers.

## Tested Providers

### ✅ Ollama (Fully Compatible)

**Tested Version**: localhost:11434
**Last Verified**: 2025-10-16 (macOS / Apple Silicon)
**Models Tested**:
- kimi-k2:1t-cloud (cloud-proxied)

**Results**:
- ✅ Simple queries
- ✅ Multi-turn conversations with context
- ✅ Streaming responses
- ✅ Config helpers (environment variables)
- ✅ Custom timeouts

**Notes**:
- Clean streaming without text duplication
- Excellent context retention
- Fast response times for cloud-proxied models

---

### ✅ LM Studio (Fully Compatible)

**Tested Version**: Network server (e.g., http://192.168.1.100:1234/v1)
**Last Verified**: 2025-10-16 (LAN-hosted server)
**Models Tested**:
- qwen/qwen3-30b-a3b-2507

**Results**:
- ✅ Simple queries
- ✅ Multi-turn conversations with context
- ✅ Streaming responses
- ✅ Config helpers (environment variables)
- ✅ Custom timeouts
- ✅ HTTPS endpoints

**Notes**:
- Streaming shows text accumulation (each chunk includes all previous text)
- Perfect context retention across turns
- Accepts invalid model names gracefully (falls back to default model)
- Works with both local and network endpoints

**Example Configuration**:
```python
from open_agent import AgentOptions

# Network LM Studio
options = AgentOptions(
    system_prompt="You are a helpful assistant",
    model="qwen/qwen3-30b-a3b-2507",
    base_url="http://192.168.1.100:1234/v1"  # Example network server
)

# Local LM Studio (default)
options = AgentOptions(
    system_prompt="You are a helpful assistant",
    model="your-local-model",
    base_url="http://localhost:1234/v1"
)
```

---

### ✅ llama.cpp (Fully Compatible)

**Tested Version**: Build 6783 (ceff6bb2) with Metal support
**Last Verified**: 2025-10-16 (local build with Metal)
**Models Tested**:
- TinyLlama 1.1B (Q4_K_M quantization)

**Results**:
- ✅ Simple queries
- ✅ Multi-turn conversations with context
- ✅ Streaming responses
- ✅ Config helpers (provider shortcut)
- ✅ OpenAI-compatible endpoints

**Notes**:
- Accumulative streaming (each chunk includes all previous text)
- Excellent Metal (GPU) acceleration on Apple Silicon
- Perfect context retention across turns
- Provider shortcut `llamacpp` maps to `http://localhost:8080/v1`
- Server requires model file path when starting

**Example Setup**:
```bash
# Build llama.cpp with Metal support (macOS)
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --parallel 8

# Start server with model
./build/bin/llama-server -m path/to/model.gguf --host 0.0.0.0 --port 8080
```

**Example Configuration**:
```python
from open_agent import AgentOptions
from open_agent.config import get_base_url

options = AgentOptions(
    system_prompt="You are a helpful assistant",
    model="your-model.gguf",  # Model name from server
    base_url=get_base_url(provider="llamacpp")  # or "http://localhost:8080/v1"
)
```

---

### ⏳ vLLM

**Expected Endpoint**: http://localhost:8000/v1
**Status**: Not tested yet

**Configuration**:
```python
from open_agent import AgentOptions
from open_agent.config import get_base_url

options = AgentOptions(
    system_prompt="...",
    model="model-name",
    base_url=get_base_url(provider="vllm")
)
```

---

## Common Patterns

### Streaming Behavior

Different providers handle streaming differently:

1. **Clean Streaming** (Ollama): Each chunk contains only new text
2. **Accumulative Streaming** (LM Studio): Each chunk contains all text so far
3. **Mixed**: Some providers switch between modes

The SDK handles all patterns correctly - agents see consistent TextBlock objects regardless.

### Model Validation

Providers differ in how they handle invalid model names:

- **Ollama**: Returns error for unknown models
- **LM Studio**: Falls back to default model (no error)
- **Others**: Varies by configuration

### Context Management

All tested providers properly maintain conversation context when using the Client class. The message history format (OpenAI-compatible) works universally.

### Performance Notes

- **Local models**: Response time depends on hardware
- **Cloud-proxied models**: Add network latency but often faster for large models
- **Streaming**: First token latency varies significantly between providers

## Troubleshooting

### Connection Refused

```python
# Check if server is running
curl http://localhost:1234/v1/models
```

### SSL/HTTPS Issues

```python
# For self-signed certificates (not recommended for production)
options = AgentOptions(
    ...,
    api_key="not-needed"  # Some servers require any non-empty value
)
```

### Timeout Issues

```python
# Increase timeout for slow models
options = AgentOptions(
    ...,
    timeout=120.0  # 2 minutes for very large models
)
```

### Invalid Model Names

Test available models first:
```bash
# LM Studio
curl https://your-server/v1/models

# Ollama
ollama list
```

## Contributing

If you test with a new provider, please update this document with:
1. Provider name and version
2. Test results (use examples/test_lmstudio.py as template)
3. Any quirks or special configuration needed
4. Example configuration code
