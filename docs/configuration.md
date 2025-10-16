# Configuration Guide

Any-Agent SDK supports multiple ways to configure the server endpoint, making it easy to switch between local and network servers without changing code.

## Configuration Methods

The SDK resolves `base_url` in this priority order:

1. **Explicit parameter** (highest priority)
2. **Environment variable** `ANY_AGENT_BASE_URL`
3. **Provider shorthand** (lmstudio, ollama, llamacpp, vllm)
4. **Default** to LM Studio on localhost

## Method 1: Explicit URL

Highest priority - always wins:

```python
from any_agent import AgentOptions

options = AgentOptions(
    system_prompt="You are a helpful assistant.",
    model="qwen2.5-32b-instruct",
    base_url="https://lmstudio.localbrandonfamily.com/v1"
)
```

## Method 2: Environment Variable

Set once, use everywhere:

```bash
# In your shell or .env file
export ANY_AGENT_BASE_URL="https://lmstudio.localbrandonfamily.com/v1"
```

Then in Python:

```python
from any_agent import AgentOptions

# No base_url needed - uses environment variable
options = AgentOptions(
    system_prompt="You are a helpful assistant.",
    model="qwen2.5-32b-instruct"
)
```

## Method 3: Provider Shorthand

Use built-in defaults for common providers:

```python
from any_agent import AgentOptions

# For Ollama
options = AgentOptions(
    system_prompt="You are a helpful assistant.",
    model="llama3.1:70b",
    provider="ollama"  # → http://localhost:11434/v1
)

# For LM Studio
options = AgentOptions(
    system_prompt="You are a helpful assistant.",
    model="qwen2.5-32b-instruct",
    provider="lmstudio"  # → http://localhost:1234/v1
)
```

### Available Providers

| Provider   | Default URL                    |
|------------|--------------------------------|
| lmstudio   | http://localhost:1234/v1       |
| ollama     | http://localhost:11434/v1      |
| llamacpp   | http://localhost:8080/v1       |
| vllm       | http://localhost:8000/v1       |

## Method 4: Default (No Configuration)

Defaults to LM Studio on localhost:

```python
from any_agent import AgentOptions

options = AgentOptions(
    system_prompt="You are a helpful assistant.",
    model="qwen2.5-32b-instruct"
    # Uses http://localhost:1234/v1
)
```

## YAML Configuration (Optional)

Install YAML support:

```bash
pip install any-agent[yaml]
```

Create a config file:

```yaml
# any-agent.yaml or ~/.config/any-agent/config.yaml
base_url: https://lmstudio.localbrandonfamily.com/v1
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

### Explicit Overrides Everything

```python
import os

os.environ["ANY_AGENT_BASE_URL"] = "http://env-server:1234/v1"

options = AgentOptions(
    system_prompt="Test",
    model="test",
    base_url="http://explicit:8080/v1",  # This wins
    provider="ollama"
)

print(options.base_url)
# Output: http://explicit:8080/v1
```

### Environment Overrides Provider

```python
import os

os.environ["ANY_AGENT_BASE_URL"] = "http://env-server:1234/v1"

options = AgentOptions(
    system_prompt="Test",
    model="test",
    provider="ollama"  # Ignored, env var wins
)

print(options.base_url)
# Output: http://env-server:1234/v1
```

### Provider Overrides Default

```python
options = AgentOptions(
    system_prompt="Test",
    model="test",
    provider="ollama"  # Uses Ollama default, not LM Studio default
)

print(options.base_url)
# Output: http://localhost:11434/v1
```

## Best Practices

### Development

Use defaults or provider shorthand for quick local testing:

```python
options = AgentOptions(
    system_prompt="Test",
    model="qwen2.5-32b-instruct"
    # Automatically uses localhost:1234
)
```

### Production

Use environment variables to avoid hardcoding:

```bash
# In production environment
export ANY_AGENT_BASE_URL="https://lmstudio.production-server.com/v1"
```

```python
# Code stays the same
options = AgentOptions(
    system_prompt="Production agent",
    model="qwen2.5-32b-instruct"
)
```

### Testing

Override with explicit URLs in tests:

```python
def test_agent():
    options = AgentOptions(
        system_prompt="Test",
        model="test-model",
        base_url="http://test-server:1234/v1"  # Override for testing
    )
```

## Complete Example

```python
import asyncio
from any_agent import query, AgentOptions, TextBlock

async def main():
    # Configure once, multiple ways to set base_url:
    # 1. Set ANY_AGENT_BASE_URL environment variable, or
    # 2. Use provider="ollama", or
    # 3. Use base_url="http://...", or
    # 4. Let it default to localhost:1234

    options = AgentOptions(
        system_prompt="You are a helpful assistant.",
        model="qwen2.5-32b-instruct",
        # No base_url needed if env var is set
    )

    result = query(prompt="What is 2+2?", options=options)

    async for msg in result:
        for block in msg.content:
            if isinstance(block, TextBlock):
                print(block.text, end="", flush=True)

asyncio.run(main())
```
