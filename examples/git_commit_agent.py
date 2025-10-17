#!/usr/bin/env python3
"""
Git Commit Agent - Analyzes staged changes and writes professional commit messages

This agent examines your staged git changes and generates well-structured commit
messages following conventional commit format (feat/fix/docs/chore/etc).

Usage:
    # Stage your changes first
    git add .

    # Run the agent
    python examples/git_commit_agent.py

    # Agent will analyze changes and suggest a commit message
    # You can accept, edit, or regenerate

Features:
- Analyzes file changes to determine commit type
- Writes clear, descriptive commit messages
- Follows conventional commit format
- Includes breaking change detection
- Lists affected files in the body
"""

import asyncio
import subprocess
import sys
import json
import re
from typing import Dict, List, Tuple
from any_agent import Client, AgentOptions, TextBlock
from any_agent.config import get_model, get_base_url


class GitCommitAgent:
    """Agent that analyzes git changes and writes commit messages."""

    COMMIT_TYPES = {
        "feat": "A new feature",
        "fix": "A bug fix",
        "docs": "Documentation only changes",
        "style": "Changes that don't affect code meaning (whitespace, formatting)",
        "refactor": "Code change that neither fixes a bug nor adds a feature",
        "perf": "Code change that improves performance",
        "test": "Adding missing tests or correcting existing tests",
        "chore": "Changes to build process or auxiliary tools",
        "ci": "Changes to CI configuration files and scripts",
        "build": "Changes that affect the build system or dependencies"
    }

    def __init__(self, options: AgentOptions):
        self.options = options

    def run_git_command(self, *args) -> str:
        """Execute a git command and return output."""
        try:
            result = subprocess.run(
                ["git"] + list(args),
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Git command failed: {e}")
            print(f"Error output: {e.stderr}")
            return ""

    def get_staged_changes(self) -> Dict[str, str]:
        """Get staged changes with file paths and diff content."""
        # Get list of staged files
        staged_files = self.run_git_command("diff", "--cached", "--name-only")
        if not staged_files:
            return {}

        changes = {}
        for file in staged_files.split('\n'):
            if file:
                # Get the diff for this specific file
                diff = self.run_git_command("diff", "--cached", file)
                changes[file] = diff

        return changes

    def get_diff_summary(self) -> str:
        """Get a summary of staged changes."""
        return self.run_git_command("diff", "--cached", "--stat")

    def analyze_changes(self, changes: Dict[str, str]) -> str:
        """Create a structured analysis of the changes for the LLM."""
        if not changes:
            return "No staged changes found."

        analysis = f"Staged changes in {len(changes)} file(s):\n\n"
        analysis += f"Summary:\n{self.get_diff_summary()}\n\n"

        # Include sample diffs (limit size for context)
        analysis += "Detailed changes:\n"
        total_chars = 0
        max_chars = 3000  # Limit context size

        for file, diff in changes.items():
            if total_chars > max_chars:
                analysis += f"\n... and {len(changes) - len(analysis.split('---')[1:])} more files"
                break

            # Truncate large diffs
            if len(diff) > 500:
                diff = diff[:500] + "\n... (truncated)"

            analysis += f"\n--- {file} ---\n{diff}\n"
            total_chars += len(diff)

        return analysis

    async def generate_commit_message(self, changes: Dict[str, str]) -> str:
        """Use LLM to generate a commit message based on changes."""
        analysis = self.analyze_changes(changes)

        if "No staged changes" in analysis:
            return ""

        prompt = f"""Analyze these git changes and write a professional commit message.

{analysis}

Based on these changes, write a commit message following these rules:

1. Use conventional commit format: type(scope): description
2. Types to choose from: {', '.join(self.COMMIT_TYPES.keys())}
3. Keep the first line under 72 characters
4. Add a blank line after the first line
5. Include a bullet list of key changes in the body
6. If there are breaking changes, add "BREAKING CHANGE:" section

Format your response as JSON with these fields:
{{
  "type": "feat|fix|docs|etc",
  "scope": "optional scope like 'auth' or 'api'",
  "subject": "imperative mood description without type prefix",
  "body": "detailed bullet points of changes",
  "breaking": "any breaking changes or empty string"
}}

Focus on WHAT changed and WHY, not just restating the diff."""

        async with Client(self.options) as client:
            await client.query(prompt)

            response = ""
            async for block in client.receive_messages():
                if isinstance(block, TextBlock):
                    response += block.text

            # Try to parse JSON response
            try:
                cleaned = response.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.strip("`")

                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1 and end > start:
                    commit_data = json.loads(cleaned[start:end + 1])
                    return self.format_commit_message(commit_data)
            except (json.JSONDecodeError, AttributeError):
                pass

            # Fallback: return raw response if not JSON
            return response

    def format_commit_message(self, data: Dict[str, str]) -> str:
        """Format the commit message from structured data."""
        # Build the commit message
        type_part = data.get('type', 'chore')
        scope_part = f"({data['scope']})" if data.get('scope') else ""
        subject = data.get('subject', 'Update code')

        message = f"{type_part}{scope_part}: {subject}"

        if data.get('body'):
            # Ensure body is formatted as bullet points
            body = data['body'].strip()
            if not body.startswith('-'):
                # Convert to bullet points if not already
                body_lines = [f"- {line.strip()}" for line in body.split('\n') if line.strip()]
                body = '\n'.join(body_lines)
            message += f"\n\n{body}"

        if data.get('breaking'):
            message += f"\n\nBREAKING CHANGE: {data['breaking']}"

        return message

    async def run(self):
        """Main agent flow."""
        print("🔍 Git Commit Agent")
        print("=" * 50)

        # Check for staged changes
        changes = self.get_staged_changes()

        if not changes:
            print("❌ No staged changes found!")
            print("\nPlease stage your changes first:")
            print("  git add <files>")
            print("  git add .")
            return

        print(f"✓ Found staged changes in {len(changes)} file(s)")
        print("\n📊 Change summary:")
        print(self.get_diff_summary())

        print("\n🤖 Analyzing changes and generating commit message...")

        commit_message = await self.generate_commit_message(changes)

        if not commit_message:
            print("❌ Failed to generate commit message")
            return

        print("\n📝 Suggested commit message:")
        print("-" * 50)
        print(commit_message)
        print("-" * 50)

        # Interactive options
        while True:
            print("\nOptions:")
            print("  [a] Accept and commit")
            print("  [e] Edit message")
            print("  [r] Regenerate")
            print("  [c] Cancel")

            choice = input("\nYour choice: ").strip().lower()

            if choice == 'a':
                # Commit with the message
                result = subprocess.run(
                    ["git", "commit", "-m", commit_message],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print("✅ Successfully committed!")
                    print(result.stdout)
                else:
                    print(f"❌ Commit failed: {result.stderr}")
                break

            elif choice == 'e':
                print("\nEdit the message (Ctrl+D when done):")
                lines = []
                while True:
                    try:
                        lines.append(input())
                    except EOFError:
                        break

                edited_message = '\n'.join(lines)
                if edited_message.strip():
                    result = subprocess.run(
                        ["git", "commit", "-m", edited_message],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        print("✅ Successfully committed with edited message!")
                    else:
                        print(f"❌ Commit failed: {result.stderr}")
                break

            elif choice == 'r':
                print("\n🤖 Regenerating commit message...")
                commit_message = await self.generate_commit_message(changes)
                print("\n📝 New commit message:")
                print("-" * 50)
                print(commit_message)
                print("-" * 50)

            elif choice == 'c':
                print("❌ Cancelled")
                break
            else:
                print("Invalid choice. Please select a, e, r, or c.")


async def main():
    """Run the Git Commit Agent."""
    # Configuration - uses env vars with fallbacks
    options = AgentOptions(
        system_prompt="""You are a git commit message expert. You write clear,
        professional commit messages that follow conventional commit standards.
        You understand code changes and can identify the type and scope of changes.
        Always be concise but descriptive.""",
        model=get_model("qwen2.5-32b-instruct"),
        base_url=get_base_url(provider="lmstudio"),
        temperature=0.3,  # Lower temperature for consistent formatting
        max_tokens=500
    )

    agent = GitCommitAgent(options)
    await agent.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
