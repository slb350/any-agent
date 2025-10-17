"""Tests for tool system"""
import pytest
from open_agent import tool, Tool
from open_agent.tools import _type_to_json_schema, _convert_schema_to_openai


class TestTypeConversion:
    """Test Python type to JSON Schema conversion"""

    def test_basic_types(self):
        """Test conversion of basic Python types"""
        assert _type_to_json_schema(str) == {"type": "string"}
        assert _type_to_json_schema(int) == {"type": "integer"}
        assert _type_to_json_schema(float) == {"type": "number"}
        assert _type_to_json_schema(bool) == {"type": "boolean"}
        assert _type_to_json_schema(list) == {"type": "array"}
        assert _type_to_json_schema(dict) == {"type": "object"}

    def test_unknown_type_defaults_to_string(self):
        """Unknown types should default to string"""
        class CustomType:
            pass

        assert _type_to_json_schema(CustomType) == {"type": "string"}


class TestSchemaConversion:
    """Test schema conversion to OpenAI format"""

    def test_simple_schema_conversion(self):
        """Test conversion of simple type mapping"""
        schema = {"name": str, "age": int}
        result = _convert_schema_to_openai(schema)

        assert result == {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

    def test_mixed_types_schema(self):
        """Test schema with various types"""
        schema = {
            "query": str,
            "limit": int,
            "threshold": float,
            "enabled": bool,
        }
        result = _convert_schema_to_openai(schema)

        assert result["type"] == "object"
        assert result["properties"]["query"] == {"type": "string"}
        assert result["properties"]["limit"] == {"type": "integer"}
        assert result["properties"]["threshold"] == {"type": "number"}
        assert result["properties"]["enabled"] == {"type": "boolean"}
        assert set(result["required"]) == {"query", "limit", "threshold", "enabled"}

    def test_already_json_schema_passthrough(self):
        """Test that existing JSON Schema is passed through unchanged"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0},
            },
            "required": ["name"],
        }
        result = _convert_schema_to_openai(schema)

        assert result == schema

    def test_empty_schema(self):
        """Test empty schema conversion"""
        schema = {}
        result = _convert_schema_to_openai(schema)

        assert result == {
            "type": "object",
            "properties": {},
            "required": [],
        }


class TestToolDecorator:
    """Test @tool decorator"""

    @pytest.mark.asyncio
    async def test_basic_tool_creation(self):
        """Test creating a simple tool with decorator"""

        @tool("greet", "Greet a user", {"name": str})
        async def greet_tool(args):
            return f"Hello, {args['name']}!"

        assert isinstance(greet_tool, Tool)
        assert greet_tool.name == "greet"
        assert greet_tool.description == "Greet a user"
        assert greet_tool.input_schema == {"name": str}

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test tool can be executed"""

        @tool("add", "Add numbers", {"a": float, "b": float})
        async def add_tool(args):
            return {"result": args["a"] + args["b"]}

        result = await add_tool.execute({"a": 5, "b": 3})
        assert result == {"result": 8}

    @pytest.mark.asyncio
    async def test_tool_with_complex_schema(self):
        """Test tool with full JSON Schema"""

        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        }

        @tool("search", "Search items", schema)
        async def search_tool(args):
            return {"query": args["query"], "count": args.get("limit", 10)}

        assert search_tool.input_schema == schema
        result = await search_tool.execute({"query": "test", "limit": 5})
        assert result == {"query": "test", "count": 5}

    @pytest.mark.asyncio
    async def test_sync_handler_is_supported(self):
        """Sync handlers should be wrapped so execute still awaits correctly"""

        @tool("shout", "Uppercase text", {"text": str})
        def shout_tool(args):
            return args["text"].upper()

        assert isinstance(shout_tool, Tool)
        result = await shout_tool.execute({"text": "hello"})
        assert result == "HELLO"


class TestToolOpenAIFormat:
    """Test conversion to OpenAI format"""

    @pytest.mark.asyncio
    async def test_simple_tool_openai_format(self):
        """Test tool conversion to OpenAI format"""

        @tool("get_weather", "Get current weather", {"location": str, "units": str})
        async def weather_tool(args):
            return {"temp": 72, "units": args["units"]}

        openai_format = weather_tool.to_openai_format()

        assert openai_format == {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "units": {"type": "string"},
                    },
                    "required": ["location", "units"],
                },
            },
        }

    @pytest.mark.asyncio
    async def test_numeric_tool_openai_format(self):
        """Test tool with numeric parameters"""

        @tool("calculate", "Perform calculation", {"a": float, "b": float, "op": str})
        async def calc_tool(args):
            return {}

        openai_format = calc_tool.to_openai_format()

        assert openai_format["function"]["parameters"]["properties"]["a"] == {
            "type": "number"
        }
        assert openai_format["function"]["parameters"]["properties"]["b"] == {
            "type": "number"
        }
        assert openai_format["function"]["parameters"]["properties"]["op"] == {
            "type": "string"
        }

    @pytest.mark.asyncio
    async def test_complex_schema_openai_format(self):
        """Test tool with pre-defined JSON Schema"""

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer", "minimum": 0, "maximum": 120},
            },
            "required": ["name"],
        }

        @tool("create_user", "Create a user", schema)
        async def user_tool(args):
            return {}

        openai_format = user_tool.to_openai_format()

        # Schema should be passed through unchanged
        assert openai_format["function"]["parameters"] == schema


class TestToolErrorHandling:
    """Test error handling in tools"""

    @pytest.mark.asyncio
    async def test_tool_execution_error_propagates(self):
        """Test that errors in tool execution propagate"""

        @tool("failing_tool", "A tool that fails", {"input": str})
        async def failing_tool(args):
            raise ValueError("Tool execution failed")

        with pytest.raises(ValueError, match="Tool execution failed"):
            await failing_tool.execute({"input": "test"})

    @pytest.mark.asyncio
    async def test_tool_can_return_error_indication(self):
        """Test that tools can return error indication in result"""

        @tool("divide", "Divide numbers", {"a": float, "b": float})
        async def divide_tool(args):
            if args["b"] == 0:
                return {"error": "Division by zero"}
            return {"result": args["a"] / args["b"]}

        # Normal execution
        result = await divide_tool.execute({"a": 10, "b": 2})
        assert result == {"result": 5.0}

        # Error case
        result = await divide_tool.execute({"a": 10, "b": 0})
        assert result == {"error": "Division by zero"}


class TestSchemaOptionality:
    """Tests for optional parameter handling in schema conversion"""

    def test_optional_param_via_default(self):
        """Parameters with defaults should not be marked required"""
        schema = {
            "query": str,
            "limit": {"type": "integer", "default": 10}
        }

        converted = _convert_schema_to_openai(schema)

        assert converted["type"] == "object"
        assert set(converted["properties"].keys()) == {"query", "limit"}
        assert converted["required"] == ["query"]
