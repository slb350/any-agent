"""Tool system for Open Agent SDK.

Provides decorator-based tool definition compatible with OpenAI's function calling API.
Simplified compared to MCP-based approaches - goes direct to OpenAI tools format.
"""

from dataclasses import dataclass
from functools import wraps
import inspect
from typing import Any, Callable, Awaitable


def _type_to_json_schema(python_type: type) -> dict[str, str]:
    """
    Convert Python type to JSON Schema type string.

    Args:
        python_type: Python type (str, int, float, bool, list, dict)

    Returns:
        JSON Schema property dict with 'type' field

    Examples:
        >>> _type_to_json_schema(str)
        {'type': 'string'}
        >>> _type_to_json_schema(int)
        {'type': 'integer'}
    """
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    json_type = type_mapping.get(python_type, "string")  # Default to string
    return {"type": json_type}


def _convert_schema_to_openai(input_schema: dict[str, type] | dict[str, Any]) -> dict[str, Any]:
    """
    Convert input schema to OpenAI function parameters format.

    Handles both simple type mappings ({"name": str}) and full JSON schemas.

    Args:
        input_schema: Either simple dict mapping param names to Python types,
                     or a complete JSON Schema dict

    Returns:
        OpenAI-compatible parameters schema

    Examples:
        Simple schema:
        >>> _convert_schema_to_openai({"location": str, "units": str})
        {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string"}
            },
            "required": ["location", "units"]
        }

        Already JSON Schema:
        >>> _convert_schema_to_openai({
        ...     "type": "object",
        ...     "properties": {"name": {"type": "string"}}
        ... })
        {
            "type": "object",
            "properties": {"name": {"type": "string"}}
        }
    """
    # Check if already a JSON Schema (has 'type' and 'properties' keys)
    if "type" in input_schema and "properties" in input_schema:
        return input_schema

    # Convert simple type mapping to JSON Schema
    properties: dict[str, Any] = {}
    required_params: list[str] = []

    for param_name, param_type in input_schema.items():
        if isinstance(param_type, type):
            properties[param_name] = _type_to_json_schema(param_type)
            required_params.append(param_name)
            continue

        if isinstance(param_type, dict):
            # Copy so we don't mutate caller data
            property_schema = dict(param_type)
            optional_flag = property_schema.pop("optional", False)
            required_flag = property_schema.pop("required", None)
            properties[param_name] = property_schema

            if required_flag is True:
                required_params.append(param_name)
            elif required_flag is False or optional_flag:
                continue
            elif "default" in property_schema:
                continue
            else:
                required_params.append(param_name)
            continue

        # Fallback: treat as string and mark required
        properties[param_name] = {"type": "string"}
        required_params.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required_params,
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
        async_handler: Callable[[dict[str, Any]], Awaitable[Any]]

        if inspect.iscoroutinefunction(handler):
            async_handler = handler
        else:
            @wraps(handler)
            async def async_wrapper(arguments: dict[str, Any]) -> Any:
                return handler(arguments)

            async_handler = async_wrapper

        return Tool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=async_handler,
        )

    return decorator
