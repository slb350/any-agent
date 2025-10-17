"""
Hooks Example

Demonstrates using hooks to monitor and control agent behavior at lifecycle points:
- PreToolUse: Intercept and control tool execution before it happens
- PostToolUse: Monitor tool results after execution
- UserPromptSubmit: Sanitize or modify user input before processing

Run this example:
    python examples/hooks_example.py
"""

import asyncio
import logging

from open_agent import (
    AgentOptions,
    Client,
    PreToolUseEvent,
    PostToolUseEvent,
    UserPromptSubmitEvent,
    HookDecision,
    HOOK_PRE_TOOL_USE,
    HOOK_POST_TOOL_USE,
    HOOK_USER_PROMPT_SUBMIT,
    tool,
)

# Set up logging to see hook execution
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Define some example tools
# ============================================================================

@tool(
    name="file_writer",
    description="Write content to a file",
    input_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    },
)
async def file_writer(arguments: dict) -> str:
    """Simulate writing to a file"""
    path = arguments.get("path", "")
    content = arguments.get("content", "")
    logger.info(f"✓ Writing to {path}: {content[:50]}...")
    return f"Successfully wrote to {path}"


@tool(
    name="web_search",
    description="Search the web",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
        },
        "required": ["query"],
    },
)
async def web_search(arguments: dict) -> str:
    """Simulate web search"""
    query = arguments.get("query", "")
    logger.info(f"✓ Searching web for: {query}")
    return f"Found results for: {query}"


# ============================================================================
# Example 1: PreToolUse Hook - Security Gates
# ============================================================================

async def security_gate(event: PreToolUseEvent) -> HookDecision | None:
    """Block dangerous file operations"""
    if event.tool_name == "file_writer":
        path = event.tool_input.get("path", "")

        # Block writes to sensitive paths
        if "/etc/" in path or "/sys/" in path:
            logger.warning(f"🛑 PreToolUse: Blocked write to {path} (security policy)")
            return HookDecision(
                continue_=False,
                reason=f"Cannot write to system path: {path}"
            )

        # Redirect writes to safe directory
        if not path.startswith("/tmp/"):
            safe_path = f"/tmp/{path.lstrip('/')}"
            logger.info(f"🔀 PreToolUse: Redirecting {path} -> {safe_path}")
            return HookDecision(
                modified_input={"path": safe_path, "content": event.tool_input.get("content", "")},
                reason="Redirected to safe directory"
            )

    return None  # Allow by default


async def example_1_security_gate():
    """Demonstrate PreToolUse hook for security"""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 1: PreToolUse Hook - Security Gates")
    logger.info("=" * 70)

    options = AgentOptions(
        system_prompt="You are a helpful assistant",
        model="gpt-3.5-turbo",  # Change to your local model
        base_url="http://localhost:1234/v1",
        tools=[file_writer, web_search],
        hooks={
            HOOK_PRE_TOOL_USE: [security_gate],
        },
    )

    async with Client(options) as client:
        # Test 1: Try to write to system path (will be blocked)
        await client.query("Write 'hello' to /etc/config.txt")
        async for block in client.receive_messages():
            logger.info(f"Response: {block}")

        # Test 2: Write to relative path (will be redirected)
        await client.query("Write 'test' to data/file.txt")
        async for block in client.receive_messages():
            logger.info(f"Response: {block}")


# ============================================================================
# Example 2: PostToolUse Hook - Audit Logging
# ============================================================================

audit_log = []


async def audit_logger(event: PostToolUseEvent) -> HookDecision:
    """Log all tool executions for compliance"""
    audit_log.append({
        "tool": event.tool_name,
        "input": event.tool_input,
        "result": str(event.tool_result)[:50],  # Truncate for display
    })
    logger.info(f"📝 PostToolUse: Logged {event.tool_name} execution (audit entry #{len(audit_log)})")
    return HookDecision(reason="Logged for audit")


async def example_2_audit_logging():
    """Demonstrate PostToolUse hook for audit logging"""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 2: PostToolUse Hook - Audit Logging")
    logger.info("=" * 70)

    global audit_log
    audit_log = []  # Reset

    options = AgentOptions(
        system_prompt="You are a helpful assistant",
        model="gpt-3.5-turbo",
        base_url="http://localhost:1234/v1",
        tools=[file_writer, web_search],
        hooks={
            HOOK_POST_TOOL_USE: [audit_logger],
        },
    )

    async with Client(options) as client:
        # Execute some tools
        await client.query("Search for 'python async' and write results to /tmp/results.txt")

        # Consume response
        blocks = []
        async for block in client.receive_messages():
            blocks.append(block)

        # Simulate tool execution (in real scenario, you'd execute the tools)
        # For demo purposes, we'll just show the audit log

    logger.info("\n📋 Audit Log:")
    for i, entry in enumerate(audit_log, 1):
        logger.info(f"  {i}. {entry['tool']}: {entry['input']} -> {entry['result']}")


# ============================================================================
# Example 3: UserPromptSubmit Hook - Input Sanitization
# ============================================================================

async def input_sanitizer(event: UserPromptSubmitEvent) -> HookDecision | None:
    """Sanitize and validate user input"""
    prompt = event.prompt

    # Block prompts with offensive content
    if "DELETE" in prompt.upper() or "DROP TABLE" in prompt.upper():
        logger.warning(f"🚫 UserPromptSubmit: Blocked dangerous prompt")
        return HookDecision(
            continue_=False,
            reason="Prompt contains potentially dangerous keywords"
        )

    # Add safety instructions
    if "write" in prompt.lower() or "delete" in prompt.lower():
        safe_prompt = prompt + " (Please confirm this is a safe operation before proceeding)"
        logger.info(f"⚠️  UserPromptSubmit: Added safety warning to prompt")
        return HookDecision(
            modified_prompt=safe_prompt,
            reason="Added safety instructions"
        )

    return None


async def example_3_input_sanitization():
    """Demonstrate UserPromptSubmit hook for input sanitization"""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 3: UserPromptSubmit Hook - Input Sanitization")
    logger.info("=" * 70)

    options = AgentOptions(
        system_prompt="You are a helpful assistant",
        model="gpt-3.5-turbo",
        base_url="http://localhost:1234/v1",
        tools=[file_writer, web_search],
        hooks={
            HOOK_USER_PROMPT_SUBMIT: [input_sanitizer],
        },
    )

    async with Client(options) as client:
        # Test 1: Dangerous prompt (will be blocked)
        try:
            await client.query("DELETE all files in my system")
        except RuntimeError as e:
            logger.info(f"❌ Query blocked: {e}")

        # Test 2: Risky prompt (will be modified)
        await client.query("Write config to file")
        async for block in client.receive_messages():
            logger.info(f"Response: {block}")


# ============================================================================
# Example 4: Combining Multiple Hooks
# ============================================================================

async def example_4_combined_hooks():
    """Demonstrate using multiple hooks together"""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE 4: Combining Multiple Hooks")
    logger.info("=" * 70)

    options = AgentOptions(
        system_prompt="You are a helpful assistant",
        model="gpt-3.5-turbo",
        base_url="http://localhost:1234/v1",
        tools=[file_writer, web_search],
        hooks={
            HOOK_USER_PROMPT_SUBMIT: [input_sanitizer],
            HOOK_PRE_TOOL_USE: [security_gate],
            HOOK_POST_TOOL_USE: [audit_logger],
        },
    )

    async with Client(options) as client:
        logger.info("\nSending: 'Write hello to config.txt'")
        logger.info("  → UserPromptSubmit will add safety warning")
        logger.info("  → PreToolUse will redirect to /tmp/")
        logger.info("  → PostToolUse will log execution\n")

        await client.query("Write hello to config.txt")
        async for block in client.receive_messages():
            logger.info(f"Response: {block}")


# ============================================================================
# Main
# ============================================================================

async def main():
    """Run all hook examples"""
    logger.info("\n🎣 Open Agent SDK - Hooks Examples")
    logger.info("=" * 70)
    logger.info("These examples use fake tools and won't make real API calls.")
    logger.info("Update the base_url to your local LLM server to see it in action.")
    logger.info("=" * 70)

    # Run examples (comment out to run individually)
    # await example_1_security_gate()
    # await example_2_audit_logging()
    # await example_3_input_sanitization()
    # await example_4_combined_hooks()

    logger.info("\n✅ Examples complete!")
    logger.info("\nKey Takeaways:")
    logger.info("  • PreToolUse: Control and modify tool execution before it happens")
    logger.info("  • PostToolUse: Monitor and log tool results after execution")
    logger.info("  • UserPromptSubmit: Sanitize and validate user input")
    logger.info("  • Hooks run inline on the event loop - spawn tasks for heavy work")
    logger.info("  • Multiple hooks can be combined for layered control")


if __name__ == "__main__":
    asyncio.run(main())
