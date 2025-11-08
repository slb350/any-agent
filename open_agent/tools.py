"""
Tool system for Open Agent SDK - OpenAI-compatible function calling

This module provides a simple, decorator-based tool definition system that
works with OpenAI-compatible function calling APIs. It's designed to be
straightforward and Pythonic, avoiding the complexity of protocol-based
approaches like MCP (Model Context Protocol).

Design Philosophy:
==================
1. SIMPLE DECORATOR PATTERN: Use @tool decorator on async functions
2. DIRECT TO OPENAI FORMAT: Convert directly to OpenAI's function calling schema
3. TYPE HINTS OPTIONAL: Support simple type dicts or full JSON Schema
4. ASYNC-FIRST: All tool handlers are async (for I/O operations)
5. NO MAGIC: Explicit schema definition, no introspection of function signatures

Why Not MCP (Model Context Protocol)?
- MCP adds protocol overhead (client/server, transports, stdio/HTTP)
- Local LLMs don't need protocol negotiation - direct API is simpler
- Easier to debug (no hidden protocol layer)
- Fewer moving parts = more reliable

How It Works:
=============
1. User defines tool with @tool decorator
2. Tool class wraps the handler function and schema
3. Tool.to_openai_format() converts to OpenAI function calling format
4. Client passes tools to LLM in API request
5. LLM generates ToolUseBlock with function call
6. SDK calls Tool.execute() with arguments
7. Result sent back to LLM as ToolResultBlock

Schema Formats Supported:
=========================
1. Simple Type Mapping (recommended for basic tools):
   {"location": str, "temperature_unit": str}

2. Full JSON Schema (for complex validation):
   {
       "type": "object",
       "properties": {
           "location": {"type": "string"},
           "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
       },
       "required": ["location"]
   }

Example Usage:
==============
>>> @tool("get_weather", "Get current weather", {"location": str})
... async def get_weather(args):
...     location = args["location"]
...     # Call weather API...
...     return {"temperature": 72, "condition": "sunny"}

>>> options = AgentOptions(
...     system_prompt="You are helpful",
...     model="qwen2.5-32b",
...     base_url="http://localhost:1234/v1",
...     tools=[get_weather]  # Register tool
... )

>>> async with Client(options) as client:
...     await client.query("What's the weather in NYC?")
...     async for block in client.receive_messages():
...         if isinstance(block, ToolUseBlock):
...             # In manual mode, execute and return result
...             # In auto mode (auto_execute_tools=True), automatic
...             pass

Implementation Notes:
=====================
- All tool handlers must be async (or will be wrapped in async)
- Tool names must be unique within an agent's tool list
- Arguments are passed as a single dict (args["param_name"])
- Tool execution errors are caught and sent to LLM as is_error=True
- Sync functions are automatically wrapped in async wrapper
"""

from dataclasses import dataclass
from functools import wraps
import inspect
from typing import Any, Callable, Awaitable


# Private helper function for type conversion
def _type_to_json_schema(python_type: type) -> dict[str, str]:
    """
    Convert Python built-in type to JSON Schema type string.

    Maps Python's native types to their JSON Schema equivalents following
    the JSON Schema specification. Used when tools provide simple type
    mappings instead of full JSON Schema definitions.

    Args:
        python_type: Python built-in type class (str, int, float, bool, list, dict)
            Example: str, int, float, bool, list, dict

    Returns:
        dict: JSON Schema property definition with 'type' field
            Example: {"type": "string"} for str

    Type Mapping:
        str   -> "string"   (text values)
        int   -> "integer"  (whole numbers)
        float -> "number"   (decimals, includes integers)
        bool  -> "boolean"  (true/false)
        list  -> "array"    (ordered sequences)
        dict  -> "object"   (key-value mappings)

    Unknown types default to "string" for maximum compatibility.

    Examples:
        >>> _type_to_json_schema(str)
        {'type': 'string'}

        >>> _type_to_json_schema(int)
        {'type': 'integer'}

        >>> _type_to_json_schema(float)
        {'type': 'number'}

        >>> _type_to_json_schema(bool)
        {'type': 'boolean'}

        >>> _type_to_json_schema(list)
        {'type': 'array'}

        >>> _type_to_json_schema(dict)
        {'type': 'object'}

    Note:
        This is a simple mapping for basic types. For complex validation
        (enums, patterns, ranges, nested schemas), use full JSON Schema
        format in input_schema instead of Python types.
    """
    # JSON Schema type mapping for Python built-ins
    type_mapping = {
        str: "string",    # Text strings
        int: "integer",   # Whole numbers (subset of number)
        float: "number",  # Floating point (includes integers in JSON Schema)
        bool: "boolean",  # True/False
        list: "array",    # Ordered sequences [1, 2, 3]
        dict: "object",   # Key-value pairs {"key": "value"}
    }

    # Look up JSON type, default to "string" for unknown types (lenient)
    json_type = type_mapping.get(python_type, "string")
    return {"type": json_type}


def _convert_schema_to_openai(input_schema: dict[str, type] | dict[str, Any]) -> dict[str, Any]:
    """
    Convert tool input schema to OpenAI function parameters format.

    Handles two input formats:
    1. Simple type mapping: {"location": str, "units": str}
    2. Full JSON Schema: {"type": "object", "properties": {...}}

    This function is the bridge between user-friendly tool definitions and
    OpenAI's strict function calling schema format.

    Args:
        input_schema: Tool parameter schema in one of two formats:
            - Simple: Dict mapping param names to Python types
              Example: {"city": str, "country": str}
            - Full: Complete JSON Schema dict with type/properties/required
              Example: {"type": "object", "properties": {...}, "required": [...]}

    Returns:
        dict: OpenAI-compatible JSON Schema with guaranteed structure:
            {
                "type": "object",
                "properties": {param_name: {schema}, ...},
                "required": [param_names...]
            }

    Examples:
        Simple type mapping (most common):
        >>> _convert_schema_to_openai({"location": str, "units": str})
        {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string"}
            },
            "required": ["location", "units"]
        }

        Already valid JSON Schema (pass-through):
        >>> _convert_schema_to_openai({
        ...     "type": "object",
        ...     "properties": {"name": {"type": "string"}},
        ...     "required": ["name"]
        ... })
        {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }

        Mixed format with optional parameters:
        >>> _convert_schema_to_openai({
        ...     "query": str,  # Required (simple type)
        ...     "limit": {"type": "integer", "default": 10}  # Optional (has default)
        ... })
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10}
            },
            "required": ["query"]  # limit not required (has default)
        }

    Schema Detection Logic:
        - If dict has "type" AND "properties" keys → assume valid JSON Schema
        - Otherwise → convert simple type mapping to JSON Schema

    Required vs Optional Parameters:
        For simple types (str, int, etc.):
            - Always marked as required

        For dict schemas (per-parameter config):
            - required=True → Add to required list
            - required=False → Not required
            - optional=True → Not required
            - Has "default" key → Not required (has default value)
            - Otherwise → Add to required list (required by default)

    Note:
        This function mutates the property_schema by popping "optional" and
        "required" keys (non-standard JSON Schema fields used for convenience).
        It makes a copy first to avoid mutating the caller's data.
    """
    # Fast path: If already in OpenAI/JSON Schema format, return as-is
    # Check for both "type" and "properties" to ensure it's a valid schema
    if "type" in input_schema and "properties" in input_schema:
        return input_schema

    # Slow path: Convert simple type mapping to full JSON Schema format
    properties: dict[str, Any] = {}  # Will hold {param_name: {type, ...}, ...}
    required_params: list[str] = []  # Will hold list of required parameter names

    # Process each parameter in the input schema
    for param_name, param_type in input_schema.items():
        # Case 1: Simple Python type (str, int, float, bool, list, dict)
        if isinstance(param_type, type):
            # Convert Python type to JSON Schema type
            properties[param_name] = _type_to_json_schema(param_type)
            # Simple types are always required (no way to specify optional)
            required_params.append(param_name)
            continue

        # Case 2: Dict schema (per-parameter JSON Schema configuration)
        if isinstance(param_type, dict):
            # Make a copy to avoid mutating caller's data when we pop keys
            property_schema = dict(param_type)

            # Pop non-standard convenience keys (not part of JSON Schema spec)
            # These are Open Agent SDK extensions for easier schema definition
            optional_flag = property_schema.pop("optional", False)  # optional=True means not required
            required_flag = property_schema.pop("required", None)  # required=True/False explicit control

            # Store the cleaned schema (without our custom keys)
            properties[param_name] = property_schema

            # Determine if this parameter should be required
            # Priority order:
            # 1. Explicit required=True → required
            # 2. Explicit required=False OR optional=True → not required
            # 3. Has "default" key → not required (default value available)
            # 4. Otherwise → required (default behavior)
            if required_flag is True:
                required_params.append(param_name)
            elif required_flag is False or optional_flag:
                # Explicitly marked as optional, don't add to required list
                continue
            elif "default" in property_schema:
                # Has default value, so not required
                continue
            else:
                # No explicit marking, assume required (safer default)
                required_params.append(param_name)
            continue

        # Case 3: Fallback for unexpected types (defensive programming)
        # If param_type is neither a type nor a dict, treat as string
        # This shouldn't happen in normal usage but prevents crashes
        properties[param_name] = {"type": "string"}
        required_params.append(param_name)

    # Return OpenAI-compatible JSON Schema format
    return {
        "type": "object",  # Function parameters are always objects in OpenAI format
        "properties": properties,  # Parameter schemas
        "required": required_params,  # List of required parameter names
    }


@dataclass
class Tool:
    """
    Tool definition for OpenAI-compatible function calling.

    Attributes:
        name: Unique tool identifier (used by model in function calls)
        description: Human-readable description (helps model understand when to use)
        input_schema: Parameter schema (simple type mapping or JSON Schema)
        handler: Async function that executes the tool

    The handler receives a dict of arguments matching the input_schema and
    should return any JSON-serializable value.
    """

    name: str
    description: str
    input_schema: dict[str, type] | dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[Any]]

    def to_openai_format(self) -> dict[str, Any]:
        """
        Convert tool definition to OpenAI function calling format.

        Returns:
            Dict matching OpenAI's tool schema:
            {
                "type": "function",
                "function": {
                    "name": "tool_name",
                    "description": "Tool description",
                    "parameters": {...}
                }
            }

        Example:
            >>> tool = Tool("add", "Add numbers", {"a": float, "b": float}, handler)
            >>> tool.to_openai_format()
            {
                "type": "function",
                "function": {
                    "name": "add",
                    "description": "Add numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"}
                        },
                        "required": ["a", "b"]
                    }
                }
            }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _convert_schema_to_openai(self.input_schema),
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> Any:
        """
        Execute the tool with given arguments.

        Args:
            arguments: Dict of arguments matching input_schema

        Returns:
            Tool execution result (any JSON-serializable value)

        Raises:
            Any exceptions raised by the handler
        """
        return await self.handler(arguments)


def tool(
    name: str,
    description: str,
    input_schema: dict[str, type] | dict[str, Any],
) -> Callable[[Callable[[dict[str, Any]], Awaitable[Any]]], Tool]:
    """
    Decorator for defining tools with OpenAI-compatible function calling.

    Creates a Tool instance that can be passed to AgentOptions for use with
    local LLMs via OpenAI-compatible endpoints.

    Args:
        name: Unique identifier for the tool. This is what the model will use
            to reference the tool in function calls.
        description: Human-readable description of what the tool does.
            This helps the model understand when to use the tool.
        input_schema: Schema defining the tool's input parameters.
            Can be either:
            - A dict mapping parameter names to Python types (e.g., {"text": str})
            - A full JSON Schema dict for complex validation

    Returns:
        Decorator function that wraps the tool handler and returns a Tool instance

    Examples:
        Simple tool with basic types:
        >>> @tool("get_weather", "Get current weather", {"location": str, "units": str})
        ... async def get_weather(args):
        ...     location = args["location"]
        ...     units = args["units"]
        ...     return {"temp": 72, "conditions": "sunny", "units": units}

        Tool with numeric types:
        >>> @tool("calculate", "Add two numbers", {"a": float, "b": float})
        ... async def add_numbers(args):
        ...     return {"result": args["a"] + args["b"]}

        Tool with error handling:
        >>> @tool("divide", "Divide numbers", {"a": float, "b": float})
        ... async def divide(args):
        ...     if args["b"] == 0:
        ...         return {"error": "Division by zero"}
        ...     return {"result": args["a"] / args["b"]}

        Tool with full JSON Schema:
        >>> @tool("search", "Search items", {
        ...     "type": "object",
        ...     "properties": {
        ...         "query": {"type": "string"},
        ...         "limit": {"type": "integer", "default": 10}
        ...     },
        ...     "required": ["query"]
        ... })
        ... async def search(args):
        ...     return {"results": []}

    Usage with AgentOptions:
        >>> options = AgentOptions(
        ...     system_prompt="You are a helpful assistant",
        ...     model="qwen2.5-32b",
        ...     base_url="http://localhost:1234/v1",
        ...     tools=[get_weather, calculate]
        ... )

    Notes:
        - The handler function must be async (defined with async def)
        - The handler receives a single dict argument with the parameters
        - The handler can return any JSON-serializable value
        - All parameters in simple schemas are required by default
        - For optional parameters, use full JSON Schema format
    """

    def decorator(handler: Callable[[dict[str, Any]], Awaitable[Any]]) -> Tool:
        # Type annotation for the async handler (will be assigned below)
        async_handler: Callable[[dict[str, Any]], Awaitable[Any]]

        # Check if the handler is already async (coroutine function)
        if inspect.iscoroutinefunction(handler):
            # Handler is already async, use it directly
            async_handler = handler
        else:
            # Handler is sync function - wrap it in an async wrapper
            # This allows users to define simple sync tools without async/await
            # The wrapper ensures all tools have consistent async interface
            @wraps(handler)  # Preserve original function metadata (__name__, __doc__, etc.)
            async def async_wrapper(arguments: dict[str, Any]) -> Any:
                # Call the sync handler directly (no await needed)
                # This runs synchronously but is callable with await
                return handler(arguments)

            async_handler = async_wrapper

        # Create and return Tool instance with the async handler
        return Tool(
            name=name,  # Tool identifier from decorator args
            description=description,  # Human-readable description from decorator args
            input_schema=input_schema,  # Parameter schema from decorator args
            handler=async_handler,  # Wrapped or original handler (guaranteed async)
        )

    # Return the decorator function (will be called when decorating a function)
    # Decorator pattern: @tool(...) returns decorator, which is called with the function
    return decorator
