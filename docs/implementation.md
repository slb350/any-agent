# Any-Agent SDK Implementation Plan

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Project Setup](#project-setup)
4. [Phase 1: Foundation](#phase-1-foundation)
5. [Phase 2: Context Management](#phase-2-context-management)
6. [Phase 3: Tool Framework](#phase-3-tool-framework)
7. [Phase 4: Session Management](#phase-4-session-management)
8. [Phase 5: Memory & Persistence](#phase-5-memory--persistence)
9. [Phase 6: Polish & Documentation](#phase-6-polish--documentation)
10. [Testing Strategy](#testing-strategy)
11. [Deployment & Release](#deployment--release)

---

## Overview

This document provides a detailed, phase-by-phase implementation plan for building the Any-Agent SDK. Each phase produces working, testable code with clear deliverables and success criteria.

### Implementation Philosophy

- **Build incrementally**: Each phase produces working, testable code
- **Test continuously**: Write tests alongside implementation
- **Document as you go**: Update docs with each feature
- **Validate early**: Test with real use cases as soon as possible
- **Iterate based on feedback**: Adjust plans based on learnings

### Phase Overview

| Phase | Focus | Key Deliverable |
|-------|-------|-----------------|
| 1 | Foundation | Basic Agent with LiteLLM integration |
| 2 | Context Management | Automatic token budget handling |
| 3 | Tool Framework | Tool registration and execution |
| 4 | Session Management | Stateful conversations |
| 5 | Memory & Persistence | SQLite storage and history |
| 6 | Polish & Documentation | Production-ready package |

---

## Prerequisites

### Development Environment

**Required:**
- Python 3.10 or higher
- pip or poetry for dependency management
- Git for version control
- SQLite (included with Python)

**Recommended:**
- Virtual environment (venv or conda)
- IDE with Python support (VS Code, PyCharm, etc.)
- LM Studio or Ollama for local testing
- OpenAI API key for cloud testing

### Dependencies

Create a `pyproject.toml` file:

```toml
[tool.poetry]
name = "any-agent"
version = "0.1.0"
description = "Agent SDK for OpenAI-compatible endpoints"
authors = ["Your Name <your.email@example.com>"]
readme = "README.md"
license = "MIT"

[tool.poetry.dependencies]
python = "^3.10"
litellm = "^1.51.0"
aiosqlite = "^0.19.0"
tiktoken = "^0.7.0"
pydantic = "^2.0"
asyncio = "^3.4.3"

[tool.poetry.dev-dependencies]
pytest = "^8.0"
pytest-asyncio = "^0.23"
pytest-cov = "^4.1"
black = "^24.0"
ruff = "^0.3"
mypy = "^1.8"
pre-commit = "^3.6"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

### Project Structure

```
any-agent/
├── any_agent/
│   ├── __init__.py
│   ├── agent.py              # Core Agent class
│   ├── options.py            # AgentOptions and enums
│   ├── context.py            # Context management
│   ├── tools.py              # Tool framework
│   ├── session.py            # Session management
│   ├── memory.py             # Memory and persistence
│   ├── errors.py             # Exception classes
│   └── utils/
│       ├── __init__.py
│       ├── litellm_wrapper.py
│       ├── validation.py
│       └── logging.py
├── tests/
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_context.py
│   ├── test_tools.py
│   ├── test_session.py
│   ├── test_memory.py
│   └── fixtures/
│       └── mock_responses.py
├── examples/
│   ├── simple_query.py
│   ├── with_tools.py
│   ├── stateful_session.py
│   └── copy_editor.py
├── docs/
│   ├── technical-design.md
│   ├── implementation.md
│   ├── api-reference.md
│   └── user-guide.md
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
└── CLAUDE.md
```

---

## Project Setup

### Step 1: Initialize Repository

```bash
# Create project directory
mkdir any-agent
cd any-agent

# Initialize git
git init

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db

# Any-Agent specific
*.db
.any_agent/
EOF
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install poetry (optional but recommended)
pip install poetry

# Initialize poetry project
poetry init

# Or install dependencies with pip
pip install litellm aiosqlite tiktoken pydantic
pip install pytest pytest-asyncio pytest-cov black ruff mypy
```

### Step 3: Create Project Structure

```bash
# Create directory structure
mkdir -p any_agent/utils
mkdir -p tests/fixtures
mkdir -p examples
mkdir -p docs

# Create __init__.py files
touch any_agent/__init__.py
touch any_agent/utils/__init__.py
touch tests/__init__.py
```

### Step 4: Set Up Pre-commit Hooks

```bash
# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.1
    hooks:
      - id: black
        language_version: python3.10

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.3.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
EOF

# Install pre-commit
pip install pre-commit
pre-commit install
```

---

## Phase 1: Foundation

### Objective

Build the core Agent class with basic query functionality using LiteLLM. Establish project architecture and patterns.

### Step 1: Core Types and Exceptions

**File: `any_agent/errors.py`**

```python
"""
Exception classes for Any-Agent SDK.
"""


class AnyAgentError(Exception):
    """Base exception for Any-Agent SDK."""
    pass


class ConfigurationError(AnyAgentError):
    """Configuration-related errors."""
    pass


class ProviderError(AnyAgentError):
    """Provider-specific errors."""

    def __init__(self, message: str, provider: str = None, original_error: Exception = None):
        super().__init__(message)
        self.provider = provider
        self.original_error = original_error


class ContextOverflowError(AnyAgentError):
    """Context exceeds token limits."""

    def __init__(self, message: str, current_tokens: int = None, max_tokens: int = None):
        super().__init__(message)
        self.current_tokens = current_tokens
        self.max_tokens = max_tokens


class ToolExecutionError(AnyAgentError):
    """Tool execution failed."""

    def __init__(self, message: str, tool_name: str = None, original_error: Exception = None):
        super().__init__(message)
        self.tool_name = tool_name
        self.original_error = original_error


class SessionError(AnyAgentError):
    """Session management errors."""
    pass


class MemoryError(AnyAgentError):
    """Memory storage errors."""
    pass


class ValidationError(AnyAgentError):
    """Input validation errors."""
    pass
```

**File: `any_agent/options.py`**

```python
"""
Configuration options and enums for Any-Agent SDK.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any, Union


class ContextStrategy(str, Enum):
    """Strategies for managing context when approaching token limits."""
    SLIDING_WINDOW = "sliding"
    TRUNCATE_OLDEST = "truncate"
    SUMMARIZE = "summarize"
    ADAPTIVE = "adaptive"


class ToolChoice(str, Enum):
    """How the agent should choose tools."""
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


@dataclass
class AgentOptions:
    """
    Configuration options for an Agent instance.

    Example:
        ```python
        options = AgentOptions(
            system_prompt="You are a helpful assistant",
            temperature=0.7,
            context_strategy=ContextStrategy.SLIDING_WINDOW
        )
        ```
    """

    # Model configuration
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 4000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Context management
    context_strategy: ContextStrategy = ContextStrategy.SLIDING_WINDOW
    context_window_size: Optional[int] = None
    preserve_system_prompt: bool = True
    max_history_messages: int = 50

    # Tool configuration
    enable_tools: bool = True
    tool_choice: Union[ToolChoice, str] = ToolChoice.AUTO
    parallel_tool_calls: bool = True
    max_tool_iterations: int = 10

    # Memory configuration
    enable_memory: bool = True
    memory_db_path: str = "~/.any_agent/memory.db"
    log_interactions: bool = True

    # Streaming configuration
    stream_by_default: bool = False
    stream_chunk_size: int = 10

    # Error handling
    retry_on_error: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_models: List[str] = field(default_factory=list)

    # Observability
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = False
    metrics_endpoint: Optional[str] = None

    # LiteLLM passthrough options
    litellm_kwargs: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration options."""
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")

        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")

        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")

        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
```

### Step 2: LiteLLM Wrapper

**File: `any_agent/utils/litellm_wrapper.py`**

```python
"""
Wrapper around LiteLLM for Any-Agent SDK.
"""

from typing import Optional, List, Dict, Any, AsyncIterator
import litellm
from litellm import acompletion, completion
import logging

from ..errors import ProviderError

logger = logging.getLogger(__name__)


class LiteLLMWrapper:
    """
    Thin wrapper around LiteLLM for Any-Agent SDK.

    Handles provider-specific configuration and response normalization.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        organization: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize LiteLLM wrapper.

        Args:
            api_key: API key for provider
            api_base: Base URL for API calls
            api_version: API version (for Azure, etc.)
            organization: Organization ID (for OpenAI)
            verbose: Enable verbose logging
        """
        self.api_key = api_key
        self.api_base = api_base
        self.api_version = api_version
        self.organization = organization

        # Configure LiteLLM
        if api_key:
            litellm.api_key = api_key
        if api_base:
            litellm.api_base = api_base
        if api_version:
            litellm.api_version = api_version
        if organization:
            litellm.organization = organization

        litellm.set_verbose = verbose

        # Disable cache by default (can be overridden)
        litellm.cache = None

    async def complete(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> Any:
        """
        Execute completion with LiteLLM.

        Args:
            model: Model identifier
            messages: Message list
            stream: Whether to stream response
            **kwargs: Additional parameters for LiteLLM

        Returns:
            LiteLLM response object or async iterator

        Raises:
            ProviderError: If the provider call fails
        """
        try:
            logger.debug(f"Calling LiteLLM with model={model}, stream={stream}")

            response = await acompletion(
                model=model,
                messages=messages,
                stream=stream,
                **kwargs
            )

            return response

        except Exception as e:
            logger.error(f"LiteLLM call failed: {e}")
            raise ProviderError(
                f"Provider call failed: {str(e)}",
                provider=self._extract_provider(model),
                original_error=e
            )

    def _extract_provider(self, model: str) -> str:
        """Extract provider name from model identifier."""
        if "/" in model:
            return model.split("/")[0]
        elif "gpt" in model.lower():
            return "openai"
        elif "claude" in model.lower():
            return "anthropic"
        else:
            return "unknown"

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """
        Get model information from LiteLLM.

        Args:
            model: Model identifier

        Returns:
            Dictionary with model metadata
        """
        try:
            from litellm import get_model_info

            info = get_model_info(model)
            return {
                'max_tokens': info.get('max_tokens', 4096),
                'input_cost': info.get('input_cost_per_token', 0),
                'output_cost': info.get('output_cost_per_token', 0),
                'supports_functions': info.get('supports_function_calling', False),
                'supports_vision': info.get('supports_vision', False)
            }
        except Exception as e:
            logger.warning(f"Could not get model info for {model}: {e}")
            # Return defaults
            return {
                'max_tokens': 4096,
                'input_cost': 0,
                'output_cost': 0,
                'supports_functions': False,
                'supports_vision': False
            }

    async def count_tokens(self, model: str, messages: List[Dict]) -> int:
        """
        Count tokens using LiteLLM's token counter.

        Args:
            model: Model identifier
            messages: Message list

        Returns:
            Number of tokens
        """
        try:
            from litellm import token_counter
            return token_counter(model=model, messages=messages)
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback: rough estimation (4 chars = 1 token)
            total_chars = sum(len(m.get('content', '')) for m in messages)
            return total_chars // 4
```

### Step 3: Core Agent Class

**File: `any_agent/agent.py`**

```python
"""
Core Agent class for Any-Agent SDK.
"""

from typing import Optional, List, Dict, Any, Union, AsyncIterator
import asyncio
import logging
from datetime import datetime

from .options import AgentOptions, ContextStrategy
from .errors import ConfigurationError, ProviderError
from .utils.litellm_wrapper import LiteLLMWrapper

logger = logging.getLogger(__name__)


class Agent:
    """
    Main Agent class that orchestrates all components.

    This class provides a high-level interface for creating AI agents
    that can maintain context, use tools, and persist memory across
    conversations.

    Example:
        ```python
        agent = Agent(model="gpt-4")
        response = await agent.query("What is machine learning?")
        print(response)
        ```
    """

    def __init__(
        self,
        model: str,
        provider: Optional[str] = None,
        options: Optional[AgentOptions] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        organization: Optional[str] = None
    ):
        """
        Initialize an Agent instance.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3", "ollama/llama3.1")
            provider: Optional provider override
            options: Configuration options
            api_key: API key for the provider
            api_base: Base URL for API calls
            api_version: API version (for Azure)
            organization: Organization ID (for OpenAI)

        Raises:
            ConfigurationError: If configuration is invalid
        """
        self.model = model
        self.provider = provider or self._detect_provider(model)
        self.options = options or AgentOptions()

        # Validate options
        try:
            self.options.validate()
        except ValueError as e:
            raise ConfigurationError(f"Invalid options: {e}")

        # Initialize LiteLLM wrapper
        self.llm = LiteLLMWrapper(
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            organization=organization,
            verbose=self.options.log_level == "DEBUG"
        )

        # Get model info
        self.model_info = self.llm.get_model_info(model)

        # Session storage (will be populated as sessions are created)
        self._sessions: Dict[str, Any] = {}

        logger.info(f"Agent initialized with model={model}, provider={self.provider}")

    async def query(
        self,
        prompt: Union[str, List[Dict[str, str]]],
        *,
        session_id: Optional[str] = None,
        stream: Optional[bool] = None,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """
        Execute a query against the model.

        Args:
            prompt: User prompt as string or message list
            session_id: Optional session ID for stateful conversations
            stream: Whether to stream the response
            **kwargs: Additional arguments passed to LiteLLM

        Returns:
            Response string or async iterator of chunks if streaming

        Example:
            ```python
            # Simple query
            response = await agent.query("Explain quantum computing")

            # Streaming query
            async for chunk in await agent.query("Tell a story", stream=True):
                print(chunk, end="")
            ```
        """
        # Prepare messages
        messages = self._prepare_messages(prompt)

        # Merge kwargs with default options
        call_kwargs = self._prepare_kwargs(**kwargs)

        # Determine if streaming
        should_stream = stream if stream is not None else self.options.stream_by_default

        # Execute with retries
        try:
            response = await self._execute_with_retries(
                messages=messages,
                stream=should_stream,
                **call_kwargs
            )

            # Process response
            if should_stream:
                return self._stream_response(response)
            else:
                return self._extract_content(response)

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def _prepare_messages(
        self,
        prompt: Union[str, List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """
        Prepare messages including system prompt.

        Args:
            prompt: User prompt as string or message list

        Returns:
            List of message dictionaries
        """
        messages = []

        # Add system prompt if configured
        if self.options.system_prompt:
            messages.append({
                "role": "system",
                "content": self.options.system_prompt
            })

        # Add user prompt
        if isinstance(prompt, str):
            messages.append({
                "role": "user",
                "content": prompt
            })
        else:
            # Already a message list
            messages.extend(prompt)

        return messages

    def _prepare_kwargs(self, **kwargs) -> Dict[str, Any]:
        """Prepare kwargs for LiteLLM call."""
        call_kwargs = {
            'temperature': self.options.temperature,
            'max_tokens': self.options.max_tokens,
            'top_p': self.options.top_p,
            'frequency_penalty': self.options.frequency_penalty,
            'presence_penalty': self.options.presence_penalty,
        }

        # Merge with user-provided kwargs
        call_kwargs.update(kwargs)

        # Add LiteLLM passthrough options
        call_kwargs.update(self.options.litellm_kwargs)

        return call_kwargs

    async def _execute_with_retries(
        self,
        messages: List[Dict],
        stream: bool,
        **kwargs
    ) -> Any:
        """
        Execute LiteLLM call with retry logic.

        Args:
            messages: Message list
            stream: Whether to stream
            **kwargs: Additional parameters

        Returns:
            LiteLLM response

        Raises:
            ProviderError: If all retries fail
        """
        models_to_try = [self.model] + self.options.fallback_models
        last_error = None

        for attempt, model in enumerate(models_to_try):
            try:
                logger.debug(f"Attempt {attempt + 1} with model {model}")

                response = await self.llm.complete(
                    model=model,
                    messages=messages,
                    stream=stream,
                    **kwargs
                )

                return response

            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                # Check if we should retry
                if attempt < len(models_to_try) - 1:
                    # Wait before retry with exponential backoff
                    if self.options.retry_on_error:
                        delay = self.options.retry_delay * (2 ** attempt)
                        logger.debug(f"Retrying in {delay}s...")
                        await asyncio.sleep(delay)
                        continue

                # No more retries
                break

        # All attempts failed
        raise last_error or ProviderError("Query failed after all retries")

    def _extract_content(self, response: Any) -> str:
        """
        Extract content from LiteLLM response.

        Args:
            response: LiteLLM response object

        Returns:
            Response content as string
        """
        try:
            return response.choices[0].message.content
        except (AttributeError, IndexError) as e:
            logger.error(f"Failed to extract content: {e}")
            return ""

    async def _stream_response(self, response: Any) -> AsyncIterator[str]:
        """
        Stream response chunks.

        Args:
            response: LiteLLM streaming response

        Yields:
            Response chunks
        """
        try:
            async for chunk in response:
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield delta.content
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise

    def _detect_provider(self, model: str) -> str:
        """
        Detect provider from model identifier.

        Args:
            model: Model identifier

        Returns:
            Provider name
        """
        model_lower = model.lower()

        if "/" in model:
            # Format: provider/model
            return model.split("/")[0]
        elif "gpt" in model_lower or "o1" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "google"
        elif "llama" in model_lower or "mistral" in model_lower:
            return "local"
        else:
            return "unknown"
```

### Step 4: Testing and Documentation

**File: `tests/test_agent.py`**

```python
"""
Tests for core Agent class.
"""

import pytest
from any_agent import Agent, AgentOptions
from any_agent.errors import ConfigurationError, ProviderError


@pytest.mark.asyncio
async def test_agent_initialization():
    """Test basic agent initialization."""
    agent = Agent(model="gpt-3.5-turbo")
    assert agent.model == "gpt-3.5-turbo"
    assert agent.provider == "openai"


@pytest.mark.asyncio
async def test_agent_with_options():
    """Test agent initialization with custom options."""
    options = AgentOptions(
        system_prompt="You are helpful",
        temperature=0.5
    )
    agent = Agent(model="gpt-4", options=options)
    assert agent.options.temperature == 0.5
    assert agent.options.system_prompt == "You are helpful"


@pytest.mark.asyncio
async def test_invalid_options():
    """Test that invalid options raise ConfigurationError."""
    with pytest.raises(ConfigurationError):
        options = AgentOptions(temperature=3.0)  # Invalid
        Agent(model="gpt-4", options=options)


@pytest.mark.asyncio
async def test_simple_query(mock_litellm):
    """Test simple query execution."""
    agent = Agent(model="gpt-3.5-turbo")
    response = await agent.query("Hello")
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.asyncio
async def test_prepare_messages():
    """Test message preparation."""
    agent = Agent(
        model="gpt-4",
        options=AgentOptions(system_prompt="You are helpful")
    )

    # String prompt
    messages = agent._prepare_messages("Hello")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    # Message list prompt
    messages = agent._prepare_messages([
        {"role": "user", "content": "Hello"}
    ])
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_provider_detection():
    """Test provider detection from model name."""
    agent_openai = Agent(model="gpt-4")
    assert agent_openai.provider == "openai"

    agent_anthropic = Agent(model="claude-3")
    assert agent_anthropic.provider == "anthropic"

    agent_ollama = Agent(model="ollama/llama3.1")
    assert agent_ollama.provider == "ollama"
```

**File: `tests/fixtures/mock_responses.py`**

```python
"""
Mock responses for testing.
"""

from typing import Any
from dataclasses import dataclass


@dataclass
class MockChoice:
    """Mock choice object."""
    message: Any
    delta: Any = None


@dataclass
class MockMessage:
    """Mock message object."""
    content: str
    role: str = "assistant"


@dataclass
class MockResponse:
    """Mock LiteLLM response."""
    choices: list

    @classmethod
    def create(cls, content: str):
        return cls(choices=[
            MockChoice(message=MockMessage(content=content))
        ])


# Common mock responses
MOCK_RESPONSES = {
    "simple": MockResponse.create("This is a test response."),
    "empty": MockResponse.create(""),
    "long": MockResponse.create("This is a much longer test response " * 100),
}
```

**File: `examples/simple_query.py`**

```python
"""
Simple query example.
"""

import asyncio
from any_agent import Agent


async def main():
    # Create agent
    agent = Agent(model="gpt-3.5-turbo")

    # Execute query
    print("Querying agent...")
    response = await agent.query("What is machine learning in one sentence?")

    print(f"\nResponse: {response}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Phase 1 Deliverables

✅ Core exception hierarchy
✅ Configuration options and enums
✅ LiteLLM wrapper with error handling
✅ Basic Agent class with query method
✅ Provider detection
✅ Retry logic
✅ Unit tests for core functionality
✅ Simple usage example

### Phase 1 Success Criteria

- [ ] Can create Agent instance
- [ ] Can execute simple queries against OpenAI
- [ ] Can execute queries against local Ollama/LM Studio
- [ ] Retry logic works for transient failures
- [ ] All unit tests pass
- [ ] Example code runs successfully

---

## Phase 2: Context Management

### Objective

Implement automatic context management with token counting and window strategies.

### Step 1: Token Counter

**File: `any_agent/context.py`** (Part 1)

```python
"""
Context management and token counting.
"""

from typing import List, Dict, Optional, Tuple
import logging
import tiktoken
from enum import Enum

from .options import ContextStrategy
from .errors import ContextOverflowError

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Model-aware token counting with fallback estimation.
    """

    # Token limits per model family
    MODEL_LIMITS = {
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-3.5-turbo": 16385,
        "gpt-3.5-turbo-16k": 16385,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-2": 100000,
        "llama-3.1-405b": 128000,
        "llama-3.1-70b": 128000,
        "llama-3.1-8b": 128000,
        "llama-3-70b": 8192,
        "llama-3-8b": 8192,
        "llama-2-70b": 4096,
        "llama-2-13b": 4096,
        "llama-2-7b": 4096,
        "mistral-large": 32768,
        "mistral-medium": 32768,
        "mistral-small": 32768,
        "mixtral-8x7b": 32768,
        "default": 4096
    }

    def __init__(self, model: str):
        """
        Initialize token counter for specific model.

        Args:
            model: Model identifier
        """
        self.model = model
        self.encoding = self._get_encoding(model)
        self.limit = self._get_limit(model)

        logger.debug(f"TokenCounter initialized: model={model}, limit={self.limit}")

    def _get_encoding(self, model: str):
        """Get the appropriate tokenizer for the model."""
        try:
            # Try to get model-specific encoding
            if "gpt" in model.lower():
                return tiktoken.encoding_for_model(model)
            else:
                # Fallback to cl100k_base (GPT-4) for estimation
                return tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Could not load encoding for {model}: {e}")
            return tiktoken.get_encoding("cl100k_base")

    def _get_limit(self, model: str) -> int:
        """Get token limit for the model."""
        model_lower = model.lower()

        # Check exact matches first
        if model_lower in self.MODEL_LIMITS:
            return self.MODEL_LIMITS[model_lower]

        # Check partial matches
        for key, limit in self.MODEL_LIMITS.items():
            if key in model_lower:
                return limit

        # Default
        logger.warning(f"Unknown model {model}, using default limit")
        return self.MODEL_LIMITS["default"]

    def count(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Number of tokens
        """
        if not isinstance(text, str):
            return 0

        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback: rough estimation (4 chars = 1 token)
            return len(text) // 4

    def count_messages(self, messages: List[Dict]) -> int:
        """
        Count tokens in a message list.

        Accounts for message formatting overhead based on model.

        Args:
            messages: List of message dictionaries

        Returns:
            Total token count
        """
        total = 0

        for message in messages:
            # Message overhead (role, formatting)
            # OpenAI format: <|start|>{role}\n{content}<|end|>\n
            total += 4

            # Role tokens
            role = message.get("role", "")
            total += self.count(role)

            # Content tokens
            content = message.get("content", "")
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                # Multi-modal content
                for item in content:
                    if isinstance(item, dict):
                        if "text" in item:
                            total += self.count(item["text"])
                        elif "image_url" in item:
                            # Image tokens (rough estimate)
                            total += 85  # Rough estimate for image

        # Add overhead for chat format
        total += 2

        return total
```

### Step 2: Context Manager

**File: `any_agent/context.py`** (Part 2)

```python
class ContextManager:
    """
    Manages conversation context to stay within token limits.

    Implements multiple strategies for handling context overflow:
    - Sliding window: Keep most recent messages
    - Truncation: Remove oldest messages
    - Summarization: Compress old messages (future)
    - Adaptive: Choose strategy based on content type
    """

    def __init__(
        self,
        model: str,
        strategy: ContextStrategy = ContextStrategy.SLIDING_WINDOW,
        max_tokens: Optional[int] = None,
        reserve_tokens: int = 1000
    ):
        """
        Initialize context manager.

        Args:
            model: Model identifier for token counting
            strategy: Context management strategy
            max_tokens: Override model's default token limit
            reserve_tokens: Tokens to reserve for model response
        """
        self.model = model
        self.strategy = strategy
        self.token_counter = TokenCounter(model)
        self.max_tokens = max_tokens or self.token_counter.limit
        self.reserve_tokens = reserve_tokens
        self.available_tokens = self.max_tokens - reserve_tokens

        logger.info(
            f"ContextManager initialized: "
            f"max_tokens={self.max_tokens}, "
            f"available={self.available_tokens}, "
            f"strategy={strategy}"
        )

    async def manage_context(
        self,
        messages: List[Dict],
        preserve_system: bool = True,
        preserve_last_n: int = 1
    ) -> List[Dict]:
        """
        Apply context management strategy to messages.

        Args:
            messages: List of message dictionaries
            preserve_system: Always keep system message
            preserve_last_n: Always keep last N user messages

        Returns:
            Managed list of messages within token budget
        """
        current_tokens = self.token_counter.count_messages(messages)

        logger.debug(
            f"Context management: {current_tokens}/{self.available_tokens} tokens"
        )

        # If within budget, return as-is
        if current_tokens <= self.available_tokens:
            return messages

        logger.info(f"Context overflow, applying {self.strategy} strategy")

        # Apply strategy
        if self.strategy == ContextStrategy.SLIDING_WINDOW:
            return await self._sliding_window(
                messages, preserve_system, preserve_last_n
            )
        elif self.strategy == ContextStrategy.TRUNCATE_OLDEST:
            return await self._truncate_oldest(
                messages, preserve_system, preserve_last_n
            )
        elif self.strategy == ContextStrategy.ADAPTIVE:
            return await self._adaptive_strategy(
                messages, preserve_system, preserve_last_n
            )

        # Fallback to sliding window
        return await self._sliding_window(
            messages, preserve_system, preserve_last_n
        )

    async def _sliding_window(
        self,
        messages: List[Dict],
        preserve_system: bool,
        preserve_last_n: int
    ) -> List[Dict]:
        """
        Keep most recent messages within token budget.
        """
        result = []
        token_count = 0

        # Separate message types
        system_messages = [m for m in messages if m["role"] == "system"]
        other_messages = [m for m in messages if m["role"] != "system"]

        # Add system message first
        if preserve_system and system_messages:
            system_msg = system_messages[-1]
            system_tokens = self.token_counter.count_messages([system_msg])

            if system_tokens < self.available_tokens:
                result.append(system_msg)
                token_count += system_tokens
            else:
                logger.warning("System message exceeds token budget")

        # Reserve space for last N messages
        preserved_messages = (
            other_messages[-preserve_last_n:] if preserve_last_n > 0 else []
        )
        preserved_tokens = self.token_counter.count_messages(preserved_messages)

        if token_count + preserved_tokens > self.available_tokens:
            # Can't fit preserved messages
            logger.warning("Cannot fit all preserved messages")
            # Keep at least the last message
            result.extend(other_messages[-1:])
            return result

        # Add messages from recent to old
        remaining_budget = self.available_tokens - token_count - preserved_tokens
        sliding_messages = []

        for message in reversed(other_messages[:-preserve_last_n or None]):
            msg_tokens = self.token_counter.count_messages([message])
            if msg_tokens <= remaining_budget:
                sliding_messages.insert(0, message)
                remaining_budget -= msg_tokens
            else:
                break

        result.extend(sliding_messages)
        result.extend(preserved_messages)

        final_tokens = self.token_counter.count_messages(result)
        logger.info(
            f"Sliding window applied: {len(result)} messages, {final_tokens} tokens"
        )

        return result

    async def _truncate_oldest(
        self,
        messages: List[Dict],
        preserve_system: bool,
        preserve_last_n: int
    ) -> List[Dict]:
        """
        Remove oldest messages to fit within budget.
        """
        result = messages.copy()
        current_tokens = self.token_counter.count_messages(result)

        # Identify what to preserve
        preserved_indices = set()

        if preserve_system:
            for i, msg in enumerate(result):
                if msg["role"] == "system":
                    preserved_indices.add(i)
                    break

        if preserve_last_n > 0:
            for i in range(max(0, len(result) - preserve_last_n), len(result)):
                preserved_indices.add(i)

        # Remove from beginning until within budget
        while current_tokens > self.available_tokens and len(result) > len(preserved_indices):
            for i in range(len(result)):
                if i not in preserved_indices:
                    result.pop(i)
                    current_tokens = self.token_counter.count_messages(result)
                    break

        logger.info(
            f"Truncation applied: {len(result)} messages, {current_tokens} tokens"
        )

        return result

    async def _adaptive_strategy(
        self,
        messages: List[Dict],
        preserve_system: bool,
        preserve_last_n: int
    ) -> List[Dict]:
        """
        Choose strategy based on conversation characteristics.
        """
        total_messages = len(messages)
        avg_length = sum(
            len(m.get("content", "")) for m in messages
        ) / max(total_messages, 1)

        # Heuristics
        if avg_length > 500:
            # Long messages, use truncation
            logger.debug("Adaptive: using truncation (long messages)")
            return await self._truncate_oldest(
                messages, preserve_system, preserve_last_n
            )
        else:
            # Short messages, use sliding window
            logger.debug("Adaptive: using sliding window (short messages)")
            return await self._sliding_window(
                messages, preserve_system, preserve_last_n
            )
```

### Step 3: Integration and Testing

**Update `any_agent/agent.py`** to use ContextManager:

```python
# Add to Agent.__init__:
from .context import ContextManager

self.context_manager = ContextManager(
    model=model,
    strategy=self.options.context_strategy,
    max_tokens=self.options.context_window_size
)

# Add to Agent.query before LiteLLM call:
# Apply context management
messages = await self.context_manager.manage_context(
    messages,
    preserve_system=self.options.preserve_system_prompt,
    preserve_last_n=1
)
```

**File: `tests/test_context.py`**

```python
"""
Tests for context management.
"""

import pytest
from any_agent.context import TokenCounter, ContextManager
from any_agent.options import ContextStrategy


def test_token_counter_init():
    """Test token counter initialization."""
    counter = TokenCounter("gpt-4")
    assert counter.limit > 0


def test_count_tokens():
    """Test token counting."""
    counter = TokenCounter("gpt-4")

    # Simple text
    count = counter.count("Hello world")
    assert count > 0
    assert count < 10

    # Empty text
    count = counter.count("")
    assert count == 0


def test_count_messages():
    """Test message token counting."""
    counter = TokenCounter("gpt-4")

    messages = [
        {"role": "system", "content": "You are helpful"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]

    count = counter.count_messages(messages)
    assert count > 0


@pytest.mark.asyncio
async def test_context_manager_no_overflow():
    """Test that messages within budget are unchanged."""
    manager = ContextManager("gpt-4")

    messages = [
        {"role": "user", "content": "Short message"}
    ]

    result = await manager.manage_context(messages)
    assert result == messages


@pytest.mark.asyncio
async def test_sliding_window():
    """Test sliding window strategy."""
    manager = ContextManager(
        "gpt-4",
        strategy=ContextStrategy.SLIDING_WINDOW,
        max_tokens=100  # Very small for testing
    )

    # Create many messages
    messages = [
        {"role": "system", "content": "System message"}
    ]

    for i in range(20):
        messages.append({"role": "user", "content": f"Message {i}"})
        messages.append({"role": "assistant", "content": f"Response {i}"})

    result = await manager.manage_context(messages, preserve_last_n=2)

    # Should be reduced
    assert len(result) < len(messages)

    # System message should be preserved
    assert result[0]["role"] == "system"

    # Last messages should be preserved
    assert result[-1] == messages[-1]
```

**Example: `examples/context_management.py`**

```python
"""
Context management example.
"""

import asyncio
from any_agent import Agent, AgentOptions, ContextStrategy


async def main():
    # Create agent with small context window
    options = AgentOptions(
        context_strategy=ContextStrategy.SLIDING_WINDOW,
        context_window_size=1000,  # Small for demonstration
        system_prompt="You are a helpful assistant with limited memory"
    )

    agent = Agent(model="gpt-3.5-turbo", options=options)

    session = await agent.create_session()

    # Have a long conversation
    for i in range(20):
        await session.send(f"This is message number {i}. Remember this number!")
        response = await session.receive()
        print(f"Turn {i}: {response[:50]}...")

    # Ask about early messages (likely forgotten due to context window)
    await session.send("What was message number 5?")
    response = await session.receive()
    print(f"\nRecall test: {response}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Phase 2 Deliverables

✅ Token counter with model-specific limits
✅ Multiple context management strategies
✅ Sliding window implementation
✅ Truncation strategy
✅ Adaptive strategy selection
✅ Integration with Agent class
✅ Comprehensive tests
✅ Context management example

### Phase 2 Success Criteria

- [ ] Token counting is accurate within 5%
- [ ] Sliding window keeps most recent messages
- [ ] Long conversations don't exceed token limits
- [ ] System prompt is preserved when requested
- [ ] Performance overhead is < 10ms per query
- [ ] All tests pass

---

## Phase 3: Tool Framework

### Objective

Implement tool registration, definition, and execution with support for both sync and async functions.

*Due to length constraints, providing the high-level structure for Phases 3-6*

### Step 1: Tool Core

**Deliverables:**
- `any_agent/tools.py` with Tool, ToolParameter, ToolRegistry classes
- Function introspection for automatic parameter detection
- OpenAI function calling format conversion
- Tool decorator for easy registration

### Step 2: Tool Execution

**Deliverables:**
- Async execution engine
- Timeout handling
- Retry logic for tool failures
- Parallel execution support

### Step 3: Integration and Testing

**Deliverables:**
- Integration with Agent.query()
- Tool call detection and handling
- Multi-turn tool conversations
- Tests and examples

---

## Phase 4: Session Management

### Objective

Implement stateful conversation sessions with context preservation.

### Step 1: Session Core

**Deliverables:**
- `any_agent/session.py` with AgentSession class
- Turn tracking
- Message history management
- send() / receive() API

### Step 2: Session Features

**Deliverables:**
- Session persistence
- Context restoration
- Session summarization
- Clear/end session methods

### Step 3: Integration and Testing

**Deliverables:**
- Session creation via Agent
- Load/save sessions
- Tests and examples

---

## Phase 5: Memory & Persistence

### Objective

Implement SQLite-based persistence for interactions, sessions, and tool executions.

### Step 1: Memory Core

**Deliverables:**
- `any_agent/memory.py` with MemoryStore class
- SQLite schema
- Async database operations
- Interaction logging

### Step 2: Advanced Features

**Deliverables:**
- Session persistence
- Search functionality
- Statistics and analytics
- Memory cleanup utilities

### Step 3: Integration and Testing

**Deliverables:**
- Automatic interaction logging
- Session save/load
- Tests and examples

---

## Phase 6: Polish & Documentation

### Objective

Production-ready package with comprehensive documentation.

### Step 1: Code Quality

**Tasks:**
- Code review and refactoring
- Performance optimization
- Error message improvements
- Type hint verification

### Step 2: Documentation

**Deliverables:**
- API reference documentation
- User guide
- Architecture documentation
- Migration guide from Claude SDK

### Step 3: Examples and Packaging

**Deliverables:**
- 5-10 comprehensive examples
- PyPI package setup
- GitHub repository setup
- CI/CD pipeline

### Step 4: Release

**Tasks:**
- Final testing
- Version 0.1.0 release
- PyPI publish
- Announcement

---

## Testing Strategy

### Unit Tests

Each component has dedicated unit tests:
- `test_agent.py`: Core agent functionality
- `test_context.py`: Context management
- `test_tools.py`: Tool framework
- `test_session.py`: Session management
- `test_memory.py`: Persistence

### Integration Tests

Test component interactions:
- Agent + Context Manager
- Agent + Tools
- Agent + Session + Memory
- Full end-to-end scenarios

### Provider Tests

Test with multiple providers:
- OpenAI GPT models
- Anthropic Claude
- Local Ollama
- LM Studio

### Performance Tests

Benchmark critical paths:
- Token counting speed
- Context management overhead
- Query latency
- Memory operations

---

## Deployment & Release

### Package Structure

```
any-agent-0.1.0/
├── any_agent/
├── tests/
├── examples/
├── docs/
├── README.md
├── LICENSE
├── pyproject.toml
└── setup.py
```

### Release Checklist

- [ ] All tests passing
- [ ] Documentation complete
- [ ] Examples working
- [ ] Version bumped
- [ ] CHANGELOG updated
- [ ] PyPI credentials configured
- [ ] GitHub release created
- [ ] Announcement posted

### Post-Release

- Monitor issues and feedback
- Plan Phase 2 features
- Community engagement
- Documentation improvements

---

## Success Metrics

### Technical Metrics

- **Test Coverage**: > 90%
- **Performance**: < 50ms overhead per query
- **Compatibility**: Works with 10+ providers
- **Reliability**: < 1% error rate

### User Metrics

- **Ease of Use**: Simple agent in < 10 lines
- **Documentation**: Complete API reference
- **Examples**: 5+ working examples
- **Community**: GitHub stars and adoption

---

## Conclusion

This implementation plan provides a structured, phase-by-phase approach to building the Any-Agent SDK. Each phase builds on the previous one, ensuring steady progress toward a production-ready package.

The modular design allows for parallel development of features and easy adaptation based on feedback. By the completion of Phase 6, we'll have a fully functional agent SDK ready for release and community adoption.