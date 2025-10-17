# Configuration Guide

Open Agent SDK uses config helpers (`get_model()` and `get_base_url()`) to provide flexible configuration, making it easy to switch between development and production without changing code.

## Philosophy

AgentOptions requires explicit `model` and `base_url` parameters. Use config helpers in your agent code to resolve these from environment variables, fallbacks, or provider defaults.

## Config Helper Resolution

### get_model() Resolution

Resolves `model` in this priority order (default behaviour):

1. **Environment variable** - `ANY_AGENT_MODEL`
2. **Fallback parameter** - `get_model("default-model")`
3. **Returns None** if not provided

Need to force a specific model even when the environment variable is set? Call `get_model("model-name", prefer_env=False)`.

### get_base_url() Resolution

Resolves `base_url` in this priority order:

1. **Explicit parameter** - `get_base_url(base_url="http://...")`
2. **Environment variable** - `ANY_AGENT_BASE_URL`
3. **Provider shorthand** - `get_base_url(provider="ollama")`
4. **Default** - LM Studio on localhost

## Method 1: Explicit Configuration

Simplest approach - pass everything explicitly:

```python
from any_agent import AgentOptions

options = AgentOptions(
    system_prompt="You are a helpful assistant.",
    model="qwen2.5-32b-instruct",
    base_url="http://192.168.1.100:1234/v1"  # Example network server
)
```

## Method 2: Environment Variables with Config Helpers

Set environment variables once, use everywhere:

```bash
# In your shell or .env file
export ANY_AGENT_BASE_URL="http://192.168.1.100:1234/v1"
export ANY_AGENT_MODEL="qwen/qwen3-30b-a3b-2507"
```

Then in Python:

```python
from any_agent import AgentOptions
from any_agent.config import get_model, get_base_url

# Config helpers read from environment
options = AgentOptions(
    system_prompt="You are a helpful assistant.",
    model=get_model(),
    base_url=get_base_url()
)
```

**Benefits:**
- Easy to switch between dev/prod environments
- No hardcoded URLs or model names in code
- Can be set in CI/CD pipelines or .env files

## Method 3: Provider Shortcuts

Use built-in defaults for common providers:

```python
from any_agent import AgentOptions
from any_agent.config import get_base_url

# For Ollama
options = AgentOptions(
    system_prompt="You are a helpful assistant.",
    model="llama3.1:70b",
    base_url=get_base_url(provider="ollama")  # → http://localhost:11434/v1
)

# For LM Studio
options = AgentOptions(
    system_prompt="You are a helpful assistant.",
    model="qwen2.5-32b-instruct",
    base_url=get_base_url(provider="lmstudio")  # → http://localhost:1234/v1
)
```

### Available Providers

| Provider   | Default URL                    |
|------------|--------------------------------|
| lmstudio   | http://localhost:1234/v1       |
| ollama     | http://localhost:11434/v1      |
| llamacpp   | http://localhost:8080/v1       |
| vllm       | http://localhost:8000/v1       |

## Method 4: Fallback Values

Provide fallbacks when environment variables aren't set:

```python
from any_agent import AgentOptions
from any_agent.config import get_model, get_base_url

options = AgentOptions(
    system_prompt="You are a helpful assistant.",
    model=get_model("qwen2.5-32b-instruct"),       # Fallback model
    base_url=get_base_url(provider="lmstudio")     # Fallback to LM Studio
)
```

If `ANY_AGENT_MODEL` is set, it uses that; otherwise uses the fallback.
To ignore the environment variable for a particular lookup, call `get_model("...", prefer_env=False)`.

## YAML Configuration (Optional)

Install YAML support:

```bash
pip install open-agent-sdk[yaml]
```

Create a config file:

```yaml
# any-agent.yaml or ~/.config/any-agent/config.yaml
base_url: http://localhost:1234/v1
model: qwen2.5-32b-instruct
temperature: 0.7
max_tokens: 4096
```

Load and use:

```python
from any_agent import AgentOptions
from any_agent.config import load_config_file

config = load_config_file()

options = AgentOptions(
    system_prompt="You are a helpful assistant.",
    **config
)
```

## Priority Examples

### Explicit Parameter Wins

```python
import os
from any_agent import AgentOptions
from any_agent.config import get_base_url

os.environ["ANY_AGENT_BASE_URL"] = "http://env-server:1234/v1"

options = AgentOptions(
    system_prompt="Test",
    model="test",
    base_url="http://explicit:8080/v1"  # This wins over env var
)

print(options.base_url)
# Output: http://explicit:8080/v1
```

### Environment Variable Wins Over Fallback

```python
import os
from any_agent import AgentOptions
from any_agent.config import get_base_url

os.environ["ANY_AGENT_BASE_URL"] = "http://env-server:1234/v1"

options = AgentOptions(
    system_prompt="Test",
    model="test",
    base_url=get_base_url(provider="ollama")  # Env var wins over provider
)

print(options.base_url)
# Output: http://env-server:1234/v1
```

### Provider Shortcut Wins Over Default

```python
from any_agent import AgentOptions
from any_agent.config import get_base_url

options = AgentOptions(
    system_prompt="Test",
    model="test",
    base_url=get_base_url(provider="ollama")  # Uses Ollama, not default LM Studio
)

print(options.base_url)
# Output: http://localhost:11434/v1
```

## Best Practices

### Development

Use provider shortcuts with fallbacks for quick local testing:

```python
from any_agent import AgentOptions
from any_agent.config import get_base_url

options = AgentOptions(
    system_prompt="Test",
    model="qwen2.5-32b-instruct",
    base_url=get_base_url(provider="lmstudio")  # Quick localhost setup
)
```

### Production

Use environment variables with config helpers to avoid hardcoding:

```bash
# In production environment
export ANY_AGENT_BASE_URL="https://lmstudio.production-server.com/v1"
export ANY_AGENT_MODEL="qwen/qwen3-30b-a3b-2507"
```

```python
from any_agent import AgentOptions
from any_agent.config import get_model, get_base_url

# Code stays the same - config comes from environment
options = AgentOptions(
    system_prompt="Production agent",
    model=get_model("qwen2.5-32b-instruct"),     # Fallback if env not set
    base_url=get_base_url(provider="lmstudio")   # Fallback if env not set
)
```

### Testing

Override with explicit parameters in tests:

```python
from any_agent import AgentOptions

def test_agent():
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://test-server:1234/v1"  # Explicit for testing
    )
```

## Complete Examples

### Full Environment Variable Configuration

```bash
# Set environment variables
export ANY_AGENT_BASE_URL="http://localhost:1234/v1"
export ANY_AGENT_MODEL="qwen/qwen3-30b-a3b-2507"
```

```python
import asyncio
from any_agent import query, AgentOptions, TextBlock
from any_agent.config import get_model, get_base_url

async def main():
    # Config helpers read from environment
    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model=get_model(),
        base_url=get_base_url()
    )

    result = query(prompt="What is 2+2?", options=options)

    async for msg in result:
        for block in msg.content:
            if isinstance(block, TextBlock):
                print(block.text, end="", flush=True)

asyncio.run(main())
```

### Explicit Configuration

```python
from any_agent import AgentOptions

options = AgentOptions(
    system_prompt="...",
    model="qwen2.5-32b-instruct",
    base_url="http://server:1234/v1"
)
```

### Mixed Configuration with Fallbacks

```python
from any_agent import AgentOptions
from any_agent.config import get_model, get_base_url

# Env vars override fallbacks
# export ANY_AGENT_BASE_URL="https://server.com/v1"

options = AgentOptions(
    system_prompt="...",
    model="qwen2.5-32b-instruct",                 # Explicit model
    base_url=get_base_url(provider="lmstudio")    # Env var or fallback
)
```
