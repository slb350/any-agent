# Building an Open‑Source, OpenAI‑Compatible Agent Framework (LM Studio)

This document analyzes the existing `.agents/` system (with a focus on the Copy Editor Agent) and proposes a provider‑agnostic Python framework that can run on an OpenAI‑compatible endpoint (e.g., LM Studio) while preserving current workflows, outputs, and data tracking.

Sections
- Context
- Current Usage of the Claude Agent SDK
- Problem Statement & Goals
- Proposed Architecture (Provider‑agnostic)
- Component Design
- Prompting & Output Contracts
- Streaming, Token Budget, and Chunking
- Error Handling & Reliability
- Quality Considerations (Models, Parameters, Guardrails)
- Migration Plan for This Repository
- Example Code (LLM adapter + agent changes)
- Configuration & Environment
- Local Dev & Run Instructions (LM Studio)
- Validation & Testing Plan
- Risks & Mitigations
- Future Enhancements
- Decision Log

---

## Context

The repository ships a multi‑agent system under `.agents/` with shared configuration and per‑agent code. The Copy Editor Agent is a concrete, production‑ready agent that:
- Reads chapters from each novel’s `chapters/` directory.
- Sends prompts to a language model for copy‑editing analysis.
- Parses structured findings (line‑keyed issues) from the model’s response.
- Generates per‑chapter reports and a summary index.
- Persists results to SQLite for historical trend analysis.

Key references:
- Copy editor runner: `.agents/run_copy_editor.py:1`
- Copy editor agent: `.agents/copy_editor/agent.py:67` (main analysis pipeline)
- Claude SDK usage: `.agents/copy_editor/agent.py:147-160`
- Response accumulation: `.agents/copy_editor/agent.py:162-166`
- Issue parsing: `.agents/copy_editor/agent.py:252-311`
- DB utilities: `.agents/copy_editor/database.py:1`
- Shared config loader: `.agents/utils/config.py:1`
- Novels config: `.agents/config/novels_metadata.yaml:200` (analysis defaults)

Summary of the current flow (Copy Editor Agent):
1) Iterate chapters → build line‑numbered content prompt
2) Send system + user prompts via Claude Agent SDK `query()`
3) Stream response → concat text → parse issue blocks → write reports
4) Update stats (SQLite) → generate index → archive reports


## Current Usage of the Claude Agent SDK

In `.agents/copy_editor/agent.py:147-160`, the agent prepares `ClaudeAgentOptions` and calls `query()`:
- `system_prompt`: copy‑editing role and strict output format
- `permission_mode='bypassPermissions'`: no tool prompts
- `cwd` / `add_dirs`: working directory context

Response handling:
- The agent consumes an async iterator of messages and extracts `block.text` from `msg.content` blocks (`.agents/copy_editor/agent.py:176-184`).

What’s not used:
- Tool execution, MCP servers, or file edit permissions (the agent reads files directly via Python).

Implication:
- We can replace the SDK call with a provider‑agnostic LLM client that produces a single response string (via streaming or non‑streaming) without touching prompts, parsing, DB, or outputs.


## Problem Statement & Goals

We want an equivalent, portable agent framework that:
- Runs against an OpenAI‑compatible endpoint (e.g., LM Studio’s `/v1/chat/completions`).
- Keeps existing agent logic (prompts, parsing, DB, reports) intact.
- Is configurable per provider/model (`claude` vs `openai‑compatible`).
- Provides streaming and good error handling.

Non‑goals:
- Rewriting agents’ business logic or report formats.
- Deep integration with provider‑specific tools/permissions (not needed here).


## Proposed Architecture (Provider‑agnostic)

Introduce a slim LLM abstraction with pluggable provider implementations:

- `LLMClient` interface: a small class with one async streaming method
  - `async stream(system_prompt: str, user_prompt: str) -> AsyncIterator[str]`
  - Yields text chunks; the agent concatenates into a single response string.

- Provider implementations:
  - `LMStudioClient` using the `openai` Python SDK with `base_url` and `api_key` to target LM Studio.
  - Optional `ClaudeClient` that wraps existing SDK (feature parity and fallback).

- Configuration‑driven selection:
  - Extend `.agents/config/novels_metadata.yaml` `analysis_defaults` with `provider`, `base_url`, and `temperature`.
  - Allow per‑novel or global defaults.

This architecture keeps the Copy Editor Agent intact while swapping only the dependency used to obtain model text.


## Component Design

1) `LLMClient` interface
- Clearly defines inputs/outputs independent of provider specifics.
- Keeps agents decoupled from HTTP/SDK details.

2) `LMStudioClient`
- Uses `openai` SDK v1.x (`AsyncOpenAI`) with `base_url` (e.g., `http://localhost:1234/v1`) and a dummy `api_key`.
- Supports streaming tokens and standard chat schema (system + user messages).
- Configurable `model`, `temperature`, and `max_tokens`.

3) (Optional) `ClaudeClient`
- Wraps `claude_agent_sdk.query()` behind the same interface for drop‑in compatibility.
- Useful if you want to freely switch between Claude Code and LM Studio.

4) Agent wiring
- `CopyEditorAgent` constructs an `LLMClient` from config (provider switch).
- `_analyze_chapter()` streams chunks, concatenates, then parses exactly as today.


## Prompting & Output Contracts

The Copy Editor Agent already enforces a strict format that’s friendly for parsing:
- System prompt with clear scope (copy editing only, high precision, low false positives)
- User prompt: numbered lines and the “issue block” template

References:
- System framing: `.agents/copy_editor/agent.py:186-236`
- User prompt with numbered lines: `.agents/copy_editor/agent.py:238-250`
- Parser: `.agents/copy_editor/agent.py:252-311`

Recommendation (optional enhancement):
- Keep the existing block format for now. If formatting drift appears under some OS models, consider migrating to JSON output with a post‑validator to eliminate regex fragility.


## Streaming, Token Budget, and Chunking

Streaming
- LM Studio’s OpenAI‑compatible streaming works similarly to OpenAI’s; we yield deltas from the chat completion stream and concatenate.

Token budget
- `analysis_defaults.max_tokens` controls the maximum generated tokens (not the prompt tokens). Ensure selected models support the needed context window (many OS models support 8k–32k+ tokens). Chapters with very large input might need chunking.

Chunking strategy (if needed later)
- Split large chapters into logical segments (e.g., ~2–4k tokens each), request issues per segment, then merge.
- Preserve line numbers by keeping a running offset; this keeps the final report consistent with the original file.


## Error Handling & Reliability

Implement standard resilience patterns in the client layer:
- Timeouts, retries with backoff for transient connection errors
- Fallback from streaming to non‑streaming if stream breaks mid‑response
- Post‑response validation: if the output cannot be parsed into at least one well‑formed block, optionally re‑ask with a shorter, stricter reminder prompt

The agent already treats exceptions per chapter and continues (`_analyze_chapter()` returns `[]` on error), so a client that raises clear exceptions is sufficient.


## Quality Considerations (Models, Parameters, Guardrails)

Model suggestions for local LM Studio:
- Strong general‑purpose: Llama 3.1 70B Instruct, Qwen 2.5 32B/72B Instruct (if available)
- Lighter options: Mistral / Mixtral Instruct variants (may require stricter prompting)

Parameters
- `temperature`: 0.0–0.2 for deterministic, format‑following output
- `top_p`: optionally tighten to 0.9 or lower for consistency

Guardrails in prompt
- Keep current strict instructions (accuracy over completeness; include context; exact format).
- If drift persists, add a trailing reminder: “Only output issue blocks in the specified format; do not include commentary.”


## Migration Plan for This Repository

Minimal, low‑risk steps:

1) Add an LLM adapter module
- New file: `.agents/llm/client.py`
- Provides `LLMClient` and `LMStudioClient` (see examples below).

2) Extend configuration
- In `.agents/config/novels_metadata.yaml`, within `analysis_defaults`, add:
  ```yaml
  analysis_defaults:
    provider: "openai"                 # or "claude"
    model: "qwen2.5-32b-instruct"     # example; match your LM Studio model
    base_url: "http://localhost:1234/v1"
    temperature: 0.1
    max_tokens: 8000
  ```

3) Update requirements
- `.agents/requirements.txt`: add `openai>=1.51.0` (or latest 1.x)

4) Patch `CopyEditorAgent` to use the adapter
- Replace direct `claude_agent_sdk` calls with an `LLMClient` constructed from config.
- Keep Claude support behind a conditional if you want to preserve the old path.

5) Verify end‑to‑end
- Run against a single small chapter; confirm reports, index, and DB are generated as before.


## Example Code

### 1) LLM adapter: `.agents/llm/client.py`

```python
# .agents/llm/client.py
# pip install openai>=1.51

import os
from typing import AsyncIterator
from openai import AsyncOpenAI


class LLMClient:
    async def stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        raise NotImplementedError


class LMStudioClient(LLMClient):
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 temperature: float = 0.1, max_tokens: int = 8000):
        self.model = model
        self.client = AsyncOpenAI(
            base_url=base_url or os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
            api_key=api_key or os.getenv("OPENAI_API_KEY", "lm-studio"),  # LM Studio ignores or accepts any key
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
```

Optionally add a `ClaudeClient` adapter if you want to preserve Claude Code usage in the same interface.


### 2) Copy Editor agent patch: `_analyze_chapter` changes

References for the current code:
- `.agents/copy_editor/agent.py:147-160` (Claude SDK call)
- `.agents/copy_editor/agent.py:162-166` (message accumulation)

Proposed integration snippet (illustrative):

```python
# .agents/copy_editor/agent.py (inside CopyEditorAgent.__init__)
from llm.client import LMStudioClient  # new

defaults = config.get_analysis_defaults()
self.model = defaults.get("model", "qwen2.5-32b-instruct")
self.max_tokens = defaults.get("max_tokens", 8000)
self.temperature = defaults.get("temperature", 0.1)
self.base_url = defaults.get("base_url")
self.provider = defaults.get("provider", "openai")

if self.provider == "openai":
    self.client = LMStudioClient(
        model=self.model,
        base_url=self.base_url,
        temperature=self.temperature,
        max_tokens=self.max_tokens,
    )
else:
    # Optionally: keep Claude path via a ClaudeClient implementing LLMClient
    from claude_agent_sdk import query  # fallback if desired
    self.client = None  # use legacy flow in _analyze_chapter


# .agents/copy_editor/agent.py (inside _analyze_chapter)
system_prompt = self._build_system_prompt()
user_prompt = self._build_chapter_prompt(chapter_file.name, content)

try:
    if self.provider == "openai":
        response_text = ""
        async for chunk in self.client.stream(system_prompt, user_prompt):
            response_text += chunk
    else:
        # Legacy Claude path (unchanged)
        from claude_agent_sdk import query
        from claude_agent_sdk.types import ClaudeAgentOptions
        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            max_turns=1,
            permission_mode='bypassPermissions',
            cwd=str(self.novel.base_path),
            add_dirs=[str(self.novel.base_path)],
        )
        result = query(prompt=user_prompt, options=options)
        response_text = ""
        async for msg in result:
            if hasattr(msg, 'content'):
                for block in msg.content:
                    if hasattr(block, 'text'):
                        response_text += block.text

    chapter_issues = self._parse_issues(chapter_file.name, response_text, lines)
    return chapter_issues
except Exception as e:
    print(f"\n   ⚠️  Error analyzing chapter: {e}")
    return []
```

This preserves all parsing, report generation, and DB logic.


## Configuration & Environment

Update `.agents/config/novels_metadata.yaml` analysis defaults:

```yaml
analysis_defaults:
  provider: "openai"                 # or "claude"
  model: "qwen2.5-32b-instruct"
  base_url: "http://localhost:1234/v1"
  temperature: 0.1
  max_tokens: 8000
```

Dependencies (`.agents/requirements.txt`):
- Add `openai>=1.51.0`
- Keep existing packages (`pyyaml`, etc.).

Environment variables (optional convenience):
- `OPENAI_BASE_URL=http://localhost:1234/v1`
- `OPENAI_API_KEY=lm-studio` (LM Studio accepts any token by default)


## Local Dev & Run Instructions (LM Studio)

1) Start LM Studio and load your preferred instruct model.
2) Enable the OpenAI‑compatible server (default: `http://localhost:1234/v1`).
3) In `.agents/` virtualenv:
   ```bash
   source .agents/venv/bin/activate
   pip install -r .agents/requirements.txt
   export OPENAI_BASE_URL="http://localhost:1234/v1"
   export OPENAI_API_KEY="lm-studio"
   ```
4) Ensure `analysis_defaults.provider: openai` and set your `model` name.
5) Run the copy editor on a single novel key for a quick test:
   ```bash
   python .agents/run_copy_editor.py bloomfall
   ```


## Validation & Testing Plan

Unit‑level
- Parser tests: feed a known, correct issue block string and verify parsed fields.
- Negative tests: incomplete/malformed blocks should be ignored safely.

Adapter‑level
- A tiny harness that calls `LMStudioClient.stream()` with a trivial system/user prompt and verifies chunks are received and concatenated.

Integration
- Use a short chapter file (~100–200 lines) to validate full flow: report files, index, and DB rows are produced as before.

Regression
- Re‑run on the same chapter after a fix and confirm reduced issue counts in the index; check SQLite trends.


## Risks & Mitigations

1) Output format drift (OS models)
- Mitigation: low temperature; keep explicit, fenced template; add a post‑validator and a retry with a more explicit “format‑only” instruction if needed.

2) Context window limitations
- Mitigation: Chapter chunking with line offset mapping; or use a larger‑context model.

3) Performance variability on local hardware
- Mitigation: Batch per chapter, background runs, or switch to a lighter model during drafting.

4) Dependency differences vs Claude SDK
- Mitigation: The agent no longer relies on CLI permissions/tools; file I/O is already done in Python, so functionality remains intact.


## Future Enhancements

- Structured JSON output + schema validation to eliminate regex parsing risk.
- A retry controller: detect parse failure and automatically re‑ask with simplified instructions.
- Provider matrix: add `vLLM`, `llama.cpp` server, or hosted OpenAI/Azure OpenAI with the same adapter.
- Quality evals: curate a small benchmark of chapters with expected issues to measure precision/recall across models.
- Multi‑agent orchestration: a coordinator that runs copy edit → style check → market analysis in a chain.


## Decision Log

- Chosen to introduce a minimal `LLMClient` abstraction to decouple agent logic from provider SDKs. This isolates changes and maximizes reuse.
- Kept the strict block‑based output format for continuity with current parsing and DB schemas.
- Recommended LM Studio via OpenAI‑compatible API for portability; maintained optional Claude fallback.


## Appendix: Key File References

- `.agents/copy_editor/agent.py:67` — Main analysis entry
- `.agents/copy_editor/agent.py:147-160` — Claude SDK call site (to be abstracted)
- `.agents/copy_editor/agent.py:252-311` — Issue parsing (regex)
- `.agents/copy_editor/database.py:1` — SQLite schema and helpers
- `.agents/run_copy_editor.py:1` — Runner; selects novel and invokes agent
- `.agents/utils/config.py:1` — Configuration loader and novel metadata
- `.agents/config/novels_metadata.yaml:200` — Analysis defaults (extend with provider/base_url)

---

If desired, I can implement the adapter module, config changes, and a minimal patch to the Copy Editor Agent so you can switch between Claude and LM Studio by changing `analysis_defaults.provider`.

