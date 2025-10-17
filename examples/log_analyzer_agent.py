#!/usr/bin/env python3
"""
Log Analyzer Agent - Intelligent log file analysis with pattern detection

This agent analyzes application logs to identify errors, patterns, and anomalies.
It provides actionable insights and can help debug production issues quickly.

Usage:
    # Analyze a log file
    python examples/log_analyzer_agent.py /path/to/app.log

    # Analyze with custom time range
    python examples/log_analyzer_agent.py app.log --since "2025-10-15T00:00:00" --until "2025-10-16"

    # Interactive mode for drilling down
    python examples/log_analyzer_agent.py app.log --interactive

Features:
- Automatic error detection and categorization
- Pattern recognition across log entries
- Time-based analysis (error rates, peak times)
- Root cause suggestions
- Time range filters (--since/--until)
- Interactive drill-down into specific issues
"""

import asyncio
import argparse
import sys
import re
import json
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from open_agent import Client, AgentOptions, TextBlock
from open_agent.config import get_model, get_base_url


class LogEntry:
    """Represents a parsed log entry."""

    def __init__(self, raw: str, timestamp: Optional[datetime] = None,
                 level: str = "INFO", message: str = "", **kwargs):
        self.raw = raw
        self.timestamp = timestamp
        self.level = level.upper()
        self.message = message
        self.metadata = kwargs

    def __str__(self):
        ts = self.timestamp.isoformat() if self.timestamp else "No timestamp"
        return f"[{ts}] {self.level}: {self.message[:100]}"


class LogParser:
    """Parses various log formats."""

    # Common log patterns
    PATTERNS = [
        # ISO timestamp with level
        re.compile(
            r'(?P<timestamp>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[^\s]*)\s*'
            r'(?:\[(?P<level>\w+)\]|\s+(?P<level2>ERROR|WARN|INFO|DEBUG))\s*'
            r'(?P<message>.*)', re.IGNORECASE
        ),
        # Apache/Nginx style
        re.compile(
            r'(?P<ip>[\d\.]+)\s+.*?\[(?P<timestamp>[^\]]+)\]\s+'
            r'"(?P<method>\w+)\s+(?P<path>[^\s]+)[^"]*"\s+'
            r'(?P<status>\d+)\s+(?P<size>\d+)'
        ),
        # Syslog style
        re.compile(
            r'(?P<timestamp>\w+\s+\d+\s+\d{2}:\d{2}:\d{2})\s+'
            r'(?P<host>\S+)\s+(?P<process>[^\[:]+)(?:\[(?P<pid>\d+)\])?\s*:\s*'
            r'(?P<message>.*)'
        ),
        # Generic with just timestamp
        re.compile(
            r'^(?P<timestamp>\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+'
            r'(?P<message>.*)'
        )
    ]

    @classmethod
    def parse_line(cls, line: str) -> LogEntry:
        """Parse a single log line."""
        line = line.strip()
        if not line:
            return None

        # JSON log lines
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                pass
            else:
                timestamp = None
                for key in ("timestamp", "time", "ts", "datetime"):
                    value = payload.get(key)
                    if value:
                        parsed = cls.parse_timestamp(str(value))
                        if parsed:
                            timestamp = parsed
                            break

                level = (
                    payload.get("level")
                    or payload.get("severity")
                    or payload.get("log_level")
                    or "INFO"
                )
                message = (
                    payload.get("message")
                    or payload.get("msg")
                    or payload.get("event")
                    or line
                )

                metadata = {
                    k: v for k, v in payload.items()
                    if k not in {"timestamp", "time", "ts", "datetime", "level", "severity", "log_level", "message", "msg", "event"}
                }

                return LogEntry(
                    raw=line,
                    timestamp=timestamp,
                    level=level,
                    message=str(message),
                    **metadata
                )

        # Try each pattern
        for pattern in cls.PATTERNS:
            match = pattern.match(line)
            if match:
                data = match.groupdict()

                # Parse timestamp
                timestamp = None
                if 'timestamp' in data and data['timestamp']:
                    timestamp = cls.parse_timestamp(data['timestamp'])

                # Determine level
                level = data.get('level') or data.get('level2') or 'INFO'

                # Get message
                message = data.get('message', line)

                # Create entry with all captured groups
                return LogEntry(
                    raw=line,
                    timestamp=timestamp,
                    level=level,
                    message=message,
                    **{k: v for k, v in data.items()
                       if k not in ['timestamp', 'level', 'level2', 'message']}
                )

        # Fallback: treat as info message
        return LogEntry(raw=line, message=line)

    @classmethod
    def parse_timestamp(cls, ts_str: str) -> Optional[datetime]:
        """Try to parse various timestamp formats."""
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%d %H:%M:%S.%f',
            '%d/%b/%Y:%H:%M:%S',
            '%b %d %H:%M:%S',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(ts_str.split('+')[0].split('Z')[0], fmt)
            except ValueError:
                continue
        return None


class LogAnalyzer:
    """Analyzes parsed logs for patterns and issues."""

    def __init__(self, entries: List[LogEntry]):
        self.entries = entries
        self.errors = [e for e in entries if e and e.level in ['ERROR', 'FATAL', 'CRITICAL']]
        self.warnings = [e for e in entries if e and e.level in ['WARN', 'WARNING']]

    def get_summary(self) -> Dict:
        """Get overall log summary."""
        total = len(self.entries)
        if total == 0:
            return {"total": 0, "errors": 0, "warnings": 0}

        # Time range
        timestamps = [e.timestamp for e in self.entries if e and e.timestamp]
        time_range = None
        if timestamps:
            time_range = {
                "start": min(timestamps).isoformat(),
                "end": max(timestamps).isoformat(),
                "duration": str(max(timestamps) - min(timestamps))
            }

        # Level distribution
        level_counts = Counter(e.level for e in self.entries if e)

        # Error patterns
        error_patterns = self.find_error_patterns()

        return {
            "total_entries": total,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "time_range": time_range,
            "level_distribution": dict(level_counts),
            "top_errors": error_patterns[:5],
            "error_rate": f"{len(self.errors) / total * 100:.1f}%"
        }

    def find_error_patterns(self) -> List[Tuple[str, int]]:
        """Find common error patterns."""
        if not self.errors:
            return []

        # Extract error types
        patterns = []
        for error in self.errors:
            msg = error.message

            # Look for exception names
            exception_match = re.search(r'(\w+(?:Exception|Error))', msg)
            if exception_match:
                patterns.append(exception_match.group(1))
                continue

            # Look for file paths in errors
            file_match = re.search(r'(?:file|at)\s+([^\s:]+\.\w+)', msg, re.IGNORECASE)
            if file_match:
                patterns.append(f"File: {file_match.group(1)}")
                continue

            # Generic classification
            if 'connection' in msg.lower():
                patterns.append("Connection Error")
            elif 'timeout' in msg.lower():
                patterns.append("Timeout Error")
            elif 'permission' in msg.lower() or 'denied' in msg.lower():
                patterns.append("Permission Error")
            elif 'not found' in msg.lower() or '404' in msg:
                patterns.append("Not Found Error")
            else:
                # Use first few words as pattern
                words = msg.split()[:5]
                patterns.append(' '.join(words))

        return Counter(patterns).most_common()

    def get_time_analysis(self) -> Dict:
        """Analyze errors over time."""
        if not self.errors:
            return {"message": "No errors found"}

        # Group errors by hour
        hourly = defaultdict(int)
        for error in self.errors:
            if error.timestamp:
                hour = error.timestamp.replace(minute=0, second=0, microsecond=0)
                hourly[hour] += 1

        if not hourly:
            return {"message": "No timestamps in error logs"}

        # Find peak times
        peak_hour = max(hourly.items(), key=lambda x: x[1])

        return {
            "peak_error_time": peak_hour[0].isoformat(),
            "peak_error_count": peak_hour[1],
            "hourly_distribution": {
                k.isoformat(): v for k, v in sorted(hourly.items())
            }
        }

    def find_correlated_events(self, target_error: str) -> List[LogEntry]:
        """Find log entries that might be related to a specific error."""
        related = []

        # Find the error entry
        target_entries = [e for e in self.errors if target_error.lower() in e.message.lower()]

        if not target_entries:
            return related

        for target in target_entries:
            if not target.timestamp:
                continue

            # Look for events within 30 seconds before the error
            window_start = target.timestamp - timedelta(seconds=30)

            for entry in self.entries:
                if entry and entry.timestamp:
                    if window_start <= entry.timestamp <= target.timestamp:
                        if entry != target:
                            related.append(entry)

        return related[:10]  # Limit to 10 most relevant


class LogAnalyzerAgent:
    """Agent that provides intelligent log analysis."""

    def __init__(self, options: AgentOptions):
        self.options = options

    async def analyze_logs(self, log_file: Path, analyzer: LogAnalyzer) -> str:
        """Use LLM to provide intelligent analysis."""
        summary = analyzer.get_summary()
        time_analysis = analyzer.get_time_analysis()

        # Prepare context for LLM
        context = f"""Analyze these application logs and provide actionable insights.

Log Summary:
{json.dumps(summary, indent=2)}

Time Analysis:
{json.dumps(time_analysis, indent=2)}

Sample Errors (first 10):
"""
        for i, error in enumerate(analyzer.errors[:10], 1):
            context += f"\n{i}. {error.raw[:200]}"

        context += """

Based on this analysis, provide:
1. Main issues identified (prioritized by severity)
2. Root cause analysis for the top errors
3. Specific recommendations to fix each issue
4. Monitoring suggestions to prevent future issues

Be specific and actionable. If you see patterns like connection errors,
timeout issues, or permission problems, provide concrete steps to diagnose
and resolve them."""

        async with Client(self.options) as client:
            await client.query(context)

            response = ""
            async for block in client.receive_messages():
                if isinstance(block, TextBlock):
                    response += block.text

            return response

    async def interactive_analysis(self, analyzer: LogAnalyzer) -> None:
        """Interactive mode for drilling down into specific issues."""
        async with Client(self.options) as client:
            print("\n🤖 Interactive Log Analysis Mode")
            print("Type 'quit' to exit, 'summary' for overview")
            print("-" * 50)

            # Initial context
            summary = analyzer.get_summary()
            initial_prompt = f"""You are analyzing application logs.
Here's the summary: {json.dumps(summary, indent=2)}

You can help me:
1. Investigate specific errors
2. Find patterns in the logs
3. Suggest fixes for issues
4. Analyze time-based patterns

What would you like to know about these logs?"""

            await client.query(initial_prompt)

            async for block in client.receive_messages():
                if isinstance(block, TextBlock):
                    print(block.text, end="", flush=True)
            print()

            # Interactive loop
            while True:
                user_input = input("\n📝 Your question: ").strip()

                if user_input.lower() == 'quit':
                    break

                if user_input.lower() == 'summary':
                    print(json.dumps(summary, indent=2))
                    continue

                # Process user query with context
                if "error" in user_input.lower():
                    # Include relevant error samples
                    context_addition = "\n\nRelevant errors:\n"
                    for error in analyzer.errors[:5]:
                        context_addition += f"- {error.raw[:150]}\n"
                    user_input += context_addition

                await client.query(user_input)

                print("\n🤖 ", end="", flush=True)
                async for block in client.receive_messages():
                    if isinstance(block, TextBlock):
                        print(block.text, end="", flush=True)
                print()

    async def run(
        self,
        log_file: Path,
        interactive: bool = False,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ):
        """Main agent flow."""
        print(f"📁 Analyzing log file: {log_file}")
        print("=" * 50)

        # Read and parse log file
        try:
            with open(log_file, 'r', errors='ignore') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"❌ Failed to read log file: {e}")
            return

        print(f"📊 Parsing {len(lines)} log lines...")

        # Parse entries
        entries = []
        for line in lines:
            entry = LogParser.parse_line(line)
            if entry:
                entries.append(entry)

        if not entries:
            print("❌ No valid log entries found")
            return

        print(f"✓ Parsed {len(entries)} log entries")

        # Apply time filtering if requested
        if since or until:
            filtered_entries: List[LogEntry] = []
            for entry in entries:
                if not entry.timestamp:
                    continue
                if since and entry.timestamp < since:
                    continue
                if until and entry.timestamp > until:
                    continue
                filtered_entries.append(entry)

            entries = filtered_entries

            range_desc = []
            if since:
                range_desc.append(f"since {since.isoformat()}")
            if until:
                range_desc.append(f"until {until.isoformat()}")
            print(f"🕑 Applied time filter ({', '.join(range_desc)})")

        if not entries:
            print("❌ No log entries matched the selected filters")
            return

        # Analyze logs
        analyzer = LogAnalyzer(entries)
        summary = analyzer.get_summary()

        # Print summary
        print("\n📈 Log Summary:")
        print(f"  Total entries: {summary['total_entries']}")
        print(f"  Errors: {summary['errors']} ({summary['error_rate']})")
        print(f"  Warnings: {summary['warnings']}")

        if summary['time_range']:
            print(f"  Time range: {summary['time_range']['duration']}")

        if summary['top_errors']:
            print("\n🔴 Top Error Patterns:")
            for pattern, count in summary['top_errors']:
                print(f"  - {pattern}: {count} occurrences")

        # Time analysis
        time_analysis = analyzer.get_time_analysis()
        if 'peak_error_time' in time_analysis:
            print(f"\n⏰ Peak error time: {time_analysis['peak_error_time']}")
            print(f"   Errors in that hour: {time_analysis['peak_error_count']}")

        if interactive:
            # Interactive mode
            await self.interactive_analysis(analyzer)
        else:
            # Generate analysis
            print("\n🤖 Generating intelligent analysis...")
            analysis = await self.analyze_logs(log_file, analyzer)

            print("\n" + "=" * 50)
            print("📋 ANALYSIS REPORT")
            print("=" * 50)
            print(analysis)
            print("=" * 50)


def _parse_time_arg(value: str) -> datetime:
    """Parse CLI datetime arguments into naive datetime objects."""
    normalized = value.strip()
    normalized = normalized.replace("Z", "")

    try:
        dt = datetime.fromisoformat(normalized)
        return dt.replace(tzinfo=None)
    except ValueError:
        pass

    candidates = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ]

    for fmt in candidates:
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue

    raise argparse.ArgumentTypeError(f"Invalid datetime: {value}")


def parse_arguments(argv: List[str]) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze application logs and generate summaries."
    )
    parser.add_argument("log_file", help="Path to the log file to analyze")
    parser.add_argument(
        "--since",
        type=_parse_time_arg,
        help="Analyze logs from this timestamp (ISO 8601 or YYYY-MM-DD)"
    )
    parser.add_argument(
        "--until",
        type=_parse_time_arg,
        help="Analyze logs up to this timestamp (ISO 8601 or YYYY-MM-DD)"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive analysis mode"
    )
    return parser.parse_args(argv)


async def main():
    """Run the Log Analyzer Agent."""
    args = parse_arguments(sys.argv[1:])

    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        sys.exit(1)

    interactive = args.interactive

    # Configuration
    options = AgentOptions(
        system_prompt="""You are an expert log analyzer and site reliability engineer.
        You excel at finding patterns in logs, identifying root causes of errors,
        and providing actionable recommendations. You understand various log formats
        and can correlate events to find issues. Always be specific and practical
        in your recommendations.""",
        model=get_model("qwen2.5-32b-instruct"),
        base_url=get_base_url(provider="lmstudio"),
        temperature=0.3,  # Lower for more consistent analysis
        max_tokens=2000,
        max_turns=10 if interactive else 1
    )

    agent = LogAnalyzerAgent(options)
    await agent.run(log_file, interactive, args.since, args.until)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
