"""
Unit tests for the centralized Tool Registry with schema generation.

Tests Story 1 acceptance criteria:
- Decorator Implementation
- Schema Extraction
- Registry Class methods
- Error Handling for missing type hints/docstrings
"""

import pytest
from core.tool_registry import (
    ToolRegistry,
    ToolDefinitionError,
    tool_schema,
    get_global_registry,
    register_tool
)


class TestToolSchemaDecorator:
    """Tests for the @tool_schema decorator."""
    
    def test_basic_function_with_type_hints(self):
        """Verify a standard Python function converts to valid JSON schema."""
        @tool_schema
        def calculate_sum(a: int, b: int) -> int:
            """Calculates the sum of two integers.
            
            Args:
                a: First integer to add
                b: Second integer to add
            
            Returns:
                The sum of a and b
            """
            return a + b
        
        schema = calculate_sum.__tool_schema__
        
        assert schema["name"] == "calculate_sum"
        assert "sum of two integers" in schema["description"]
        assert schema["parameters"]["type"] == "object"
        assert schema["parameters"]["properties"]["a"]["type"] == "integer"
        assert schema["parameters"]["properties"]["b"]["type"] == "integer"
        assert "a" in schema["parameters"]["required"]
        assert "b" in schema["parameters"]["required"]
    
    def test_function_with_string_params(self):
        """Test schema extraction for string parameters."""
        @tool_schema
        def greet(name: str, greeting: str = "Hello") -> str:
            """Greets a person.
            
            Args:
                name: The person's name
                greeting: The greeting phrase
            """
            return f"{greeting}, {name}!"
        
        schema = greet.__tool_schema__
        
        assert schema["parameters"]["properties"]["name"]["type"] == "string"
        assert schema["parameters"]["properties"]["greeting"]["type"] == "string"
        assert "name" in schema["parameters"]["required"]
        assert "greeting" not in schema["parameters"]["required"]  # Has default
    
    def test_function_with_optional_params(self):
        """Test that optional parameters are not in required list."""
        from typing import Optional
        
        @tool_schema
        def search(query: str, max_results: int = 10) -> str:
            """Searches for something.
            
            Args:
                query: The search query
                max_results: Maximum results to return
            """
            return f"Searching for {query}"
        
        schema = search.__tool_schema__
        assert "query" in schema["parameters"]["required"]
        assert "max_results" not in schema["parameters"]["required"]
    
    def test_missing_type_hints_raises_error(self):
        """Verify that functions without type hints raise ToolDefinitionError."""
        with pytest.raises(ToolDefinitionError) as exc_info:
            @tool_schema
            def bad_function(a, b):
                """This function has no type hints."""
                return a + b
        
        assert "missing type hints" in str(exc_info.value).lower()
    
    def test_missing_docstring_raises_error(self):
        """Verify that functions without docstrings raise ToolDefinitionError."""
        with pytest.raises(ToolDefinitionError) as exc_info:
            @tool_schema
            def no_docs(a: int, b: int) -> int:
                return a + b
        
        assert "docstring" in str(exc_info.value).lower()
    
    def test_partial_type_hints_raises_error(self):
        """Verify that partial type hints raise ToolDefinitionError."""
        with pytest.raises(ToolDefinitionError) as exc_info:
            @tool_schema
            def partial_hints(a: int, b):
                """Only one parameter has a type hint."""
                return a + b
        
        assert "missing type hints" in str(exc_info.value).lower()
        assert "b" in str(exc_info.value)
    
    def test_decorated_function_still_callable(self):
        """Verify the decorated function still executes correctly."""
        @tool_schema
        def multiply(x: int, y: int) -> int:
            """Multiplies two numbers.
            
            Args:
                x: First number
                y: Second number
            """
            return x * y
        
        assert multiply(3, 4) == 12
        assert multiply(0, 100) == 0


class TestToolRegistry:
    """Tests for the ToolRegistry class."""
    
    def test_register_decorated_function(self):
        """Test registering a @tool_schema decorated function."""
        registry = ToolRegistry()
        
        @tool_schema
        def my_tool(x: int) -> str:
            """Converts an integer to string.
            
            Args:
                x: The integer to convert
            """
            return str(x)
        
        registry.register(my_tool)
        
        assert "my_tool" in registry.list_tools()
        assert len(registry) == 1
    
    def test_get_tool_returns_callable(self):
        """Test that get_tool returns an executable function."""
        registry = ToolRegistry()
        
        @tool_schema
        def add_one(n: int) -> int:
            """Adds one to a number.
            
            Args:
                n: Input number
            """
            return n + 1
        
        registry.register(add_one)
        tool = registry.get_tool("add_one")
        
        assert callable(tool)
        assert tool(5) == 6
    
    def test_get_schemas_returns_list(self):
        """Test that get_schemas returns proper OpenAI format."""
        registry = ToolRegistry()
        
        @tool_schema
        def tool_a(x: int) -> int:
            """Tool A.
            
            Args:
                x: Parameter
            """
            return x
        
        @tool_schema
        def tool_b(y: str) -> str:
            """Tool B.
            
            Args:
                y: Parameter
            """
            return y
        
        registry.register(tool_a)
        registry.register(tool_b)
        
        schemas = registry.get_schemas()
        
        assert isinstance(schemas, list)
        assert len(schemas) == 2
        
        names = [s["name"] for s in schemas]
        assert "tool_a" in names
        assert "tool_b" in names
    
    def test_get_tool_not_found_raises_keyerror(self):
        """Test that requesting non-existent tool raises KeyError."""
        registry = ToolRegistry()
        
        with pytest.raises(KeyError) as exc_info:
            registry.get_tool("nonexistent")
        
        assert "nonexistent" in str(exc_info.value)
    
    def test_register_with_custom_name(self):
        """Test registering a tool with a custom name."""
        registry = ToolRegistry()
        
        @tool_schema
        def internal_name(x: int) -> int:
            """Does something.
            
            Args:
                x: Input
            """
            return x
        
        registry.register(internal_name, name="custom_name")
        
        assert "custom_name" in registry.list_tools()
        assert "internal_name" not in registry.list_tools()
    
    def test_get_formatted_metadata(self):
        """Test the formatted metadata output for prompts."""
        registry = ToolRegistry()
        
        @tool_schema
        def sample_tool(query: str) -> str:
            """Searches for information.
            
            Args:
                query: The search query
            """
            return query
        
        registry.register(sample_tool)
        
        metadata = registry.get_formatted_tool_metadata()
        
        assert "sample_tool" in metadata
        assert "Searches for information" in metadata
        assert "query" in metadata
    
    def test_empty_registry_formatted_metadata(self):
        """Test formatted metadata for empty registry."""
        registry = ToolRegistry()
        
        metadata = registry.get_formatted_tool_metadata()
        
        assert "No tools are available" in metadata
    
    def test_contains_operator(self):
        """Test the 'in' operator for checking tool existence."""
        registry = ToolRegistry()
        
        @tool_schema
        def exists(x: int) -> int:
            """Exists.
            
            Args:
                x: Input
            """
            return x
        
        registry.register(exists)
        
        assert "exists" in registry
        assert "not_exists" not in registry


class TestCalculateSumSchema:
    """
    Specific test case from acceptance criteria:
    Verify that calculate_sum(a: int, b: int) converts to valid JSON schema.
    """
    
    def test_calculate_sum_full_schema_validation(self):
        """Complete validation of calculate_sum schema structure."""
        @tool_schema
        def calculate_sum(a: int, b: int) -> int:
            """Calculates the sum of two integers.
            
            Args:
                a: The first integer operand
                b: The second integer operand
            
            Returns:
                int: The sum of a and b
            """
            return a + b
        
        schema = calculate_sum.__tool_schema__
        
        # Validate top-level structure
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
        
        # Validate name
        assert schema["name"] == "calculate_sum"
        
        # Validate description
        assert len(schema["description"]) > 0
        
        # Validate parameters schema
        params = schema["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params
        
        # Validate property 'a'
        assert "a" in params["properties"]
        assert params["properties"]["a"]["type"] == "integer"
        assert "description" in params["properties"]["a"]
        
        # Validate property 'b'
        assert "b" in params["properties"]
        assert params["properties"]["b"]["type"] == "integer"
        assert "description" in params["properties"]["b"]
        
        # Validate required
        assert set(params["required"]) == {"a", "b"}
    
    def test_calculate_sum_in_registry(self):
        """Test registering and using calculate_sum via registry."""
        registry = ToolRegistry()
        
        @tool_schema
        def calculate_sum(a: int, b: int) -> int:
            """Calculates the sum of two integers.
            
            Args:
                a: First integer
                b: Second integer
            """
            return a + b
        
        registry.register(calculate_sum)
        
        # Get and execute
        tool = registry.get_tool("calculate_sum")
        result = tool(3, 7)
        
        assert result == 10
        
        # Verify schema is accessible
        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "calculate_sum"


class TestGlobalRegistry:
    """Tests for the global registry singleton."""
    
    def test_global_registry_singleton(self):
        """Verify get_global_registry returns same instance."""
        reg1 = get_global_registry()
        reg2 = get_global_registry()
        
        assert reg1 is reg2
    
    def test_register_tool_decorator(self):
        """Test the @register_tool decorator adds to global registry."""
        # Clear any previous state
        global_reg = get_global_registry()
        initial_count = len(global_reg)
        
        @register_tool
        def global_tool(x: int) -> int:
            """A globally registered tool.
            
            Args:
                x: Input value
            """
            return x * 2
        
        assert len(global_reg) > initial_count
        assert "global_tool" in global_reg


class TestComplexTypes:
    """Tests for handling complex Python types."""
    
    def test_list_parameter(self):
        """Test schema extraction for List type."""
        from typing import List
        
        @tool_schema
        def process_items(items: List[str]) -> int:
            """Processes a list of items.
            
            Args:
                items: List of strings to process
            """
            return len(items)
        
        schema = process_items.__tool_schema__
        assert schema["parameters"]["properties"]["items"]["type"] == "array"
    
    def test_dict_parameter(self):
        """Test schema extraction for Dict type."""
        from typing import Dict
        
        @tool_schema
        def process_config(config: Dict[str, int]) -> str:
            """Processes a configuration dict.
            
            Args:
                config: Configuration dictionary
            """
            return str(config)
        
        schema = process_config.__tool_schema__
        assert schema["parameters"]["properties"]["config"]["type"] == "object"
    
    def test_bool_parameter(self):
        """Test schema extraction for bool type."""
        @tool_schema
        def toggle(enabled: bool) -> str:
            """Toggles a setting.
            
            Args:
                enabled: Whether to enable
            """
            return "on" if enabled else "off"
        
        schema = toggle.__tool_schema__
        assert schema["parameters"]["properties"]["enabled"]["type"] == "boolean"
    
    def test_float_parameter(self):
        """Test schema extraction for float type."""
        @tool_schema
        def scale(value: float, factor: float) -> float:
            """Scales a value.
            
            Args:
                value: The value to scale
                factor: The scaling factor
            """
            return value * factor
        
        schema = scale.__tool_schema__
        assert schema["parameters"]["properties"]["value"]["type"] == "number"
        assert schema["parameters"]["properties"]["factor"]["type"] == "number"
