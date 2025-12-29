"""
Unit tests for Story 1.1: Registry-Retrieval Handshake.

Verifies:
- ToolRegistry.get_all_tool_schemas() returns JSON-serializable definitions
- ToolRetriever logs warnings for tools not in registry
"""
import pytest
import json
import logging
from unittest.mock import patch, MagicMock

from core.tool_registry import ToolRegistry, tool_schema, get_global_registry


class TestRegistrySchemaExport:
    """Tests for ToolRegistry.get_all_tool_schemas()"""
    
    def test_get_all_tool_schemas_returns_list(self):
        """Verify get_all_tool_schemas returns a list of dicts."""
        registry = ToolRegistry()
        
        @tool_schema
        def sample_tool(x: int) -> str:
            """Sample tool for testing.
            
            Args:
                x: Input value
            """
            return str(x)
        
        registry.register(sample_tool)
        schemas = registry.get_all_tool_schemas()
        
        assert isinstance(schemas, list)
        assert len(schemas) == 1
        assert schemas[0]["name"] == "sample_tool"
        assert "parameters" in schemas[0]
        assert "description" in schemas[0]
    
    def test_schemas_are_json_serializable(self):
        """Verify schemas can be serialized to JSON."""
        registry = ToolRegistry()
        
        @tool_schema
        def test_tool(query: str, count: int = 5) -> str:
            """Test tool for JSON serialization.
            
            Args:
                query: Search query
                count: Result count
            """
            return query
        
        registry.register(test_tool)
        schemas = registry.get_all_tool_schemas()
        
        # Should not raise
        json_str = json.dumps(schemas)
        assert json_str is not None
        
        # Verify round-trip
        parsed = json.loads(json_str)
        assert parsed[0]["name"] == "test_tool"
    
    def test_empty_registry_returns_empty_list(self):
        """Verify empty registry returns empty list."""
        registry = ToolRegistry()
        schemas = registry.get_all_tool_schemas()
        
        assert isinstance(schemas, list)
        assert len(schemas) == 0
    
    def test_multiple_tools_schema_export(self):
        """Verify multiple tools are exported correctly."""
        registry = ToolRegistry()
        
        @tool_schema
        def tool_a(x: int) -> int:
            """Tool A does something.
            
            Args:
                x: Input value
            """
            return x
        
        @tool_schema
        def tool_b(name: str) -> str:
            """Tool B does something else.
            
            Args:
                name: A name string
            """
            return name
        
        registry.register(tool_a)
        registry.register(tool_b)
        
        schemas = registry.get_all_tool_schemas()
        
        assert len(schemas) == 2
        names = {s["name"] for s in schemas}
        assert "tool_a" in names
        assert "tool_b" in names
    
    def test_schema_contains_parameters(self):
        """Verify schema contains proper parameter definitions."""
        registry = ToolRegistry()
        
        @tool_schema
        def parameterized_tool(query: str, max_results: int = 10, verbose: bool = False) -> str:
            """A tool with multiple parameters.
            
            Args:
                query: The search query
                max_results: Maximum results
                verbose: Enable verbose output
            """
            return query
        
        registry.register(parameterized_tool)
        schemas = registry.get_all_tool_schemas()
        
        params = schemas[0]["parameters"]
        assert "properties" in params
        assert "query" in params["properties"]
        assert "max_results" in params["properties"]
        assert "verbose" in params["properties"]


class TestRetrieverRegistryValidation:
    """Tests for ToolRetriever registry validation."""
    
    def test_retriever_logs_warning_for_unregistered_tool(self, caplog):
        """Verify warning is logged for tools not in registry."""
        from core.tool_retriever import ToolRetriever
        
        # Create a mock tool not in registry
        mock_tool = MagicMock()
        mock_tool.name = "ghost_tool_that_does_not_exist"
        mock_tool.description = "A tool that doesn't exist in registry"
        
        with caplog.at_level(logging.WARNING):
            retriever = ToolRetriever(
                tools=[mock_tool],
                validate_against_registry=True
            )
        
        assert "ghost_tool_that_does_not_exist" in caplog.text
        assert "not found in ToolRegistry" in caplog.text
    
    def test_retriever_no_warning_for_registered_tool(self, caplog):
        """Verify no warning for tools that are in the registry."""
        from core.tool_retriever import ToolRetriever
        
        # Register a tool in global registry
        global_reg = get_global_registry()
        
        @tool_schema
        def registered_test_tool(x: int) -> str:
            """A properly registered tool.
            
            Args:
                x: Input value
            """
            return str(x)
        
        global_reg.register(registered_test_tool)
        
        # Create a mock tool with matching name
        mock_tool = MagicMock()
        mock_tool.name = "registered_test_tool"
        mock_tool.description = "A registered tool"
        
        with caplog.at_level(logging.WARNING):
            retriever = ToolRetriever(
                tools=[mock_tool],
                validate_against_registry=True
            )
        
        # Should not log a warning for this tool
        assert "registered_test_tool" not in caplog.text or "not found" not in caplog.text
    
    def test_retriever_validation_can_be_disabled(self, caplog):
        """Verify validation can be disabled."""
        from core.tool_retriever import ToolRetriever
        
        mock_tool = MagicMock()
        mock_tool.name = "another_ghost_tool"
        mock_tool.description = "Another ghost tool"
        
        with caplog.at_level(logging.WARNING):
            retriever = ToolRetriever(
                tools=[mock_tool],
                validate_against_registry=False
            )
        
        # Should not log any warning when validation is disabled
        assert "another_ghost_tool" not in caplog.text


class TestRegistryRetrieverHandshake:
    """Integration tests for the registry-retriever handshake."""
    
    def test_get_all_tool_schemas_matches_get_schemas(self):
        """Verify get_all_tool_schemas is consistent with get_schemas."""
        registry = ToolRegistry()
        
        @tool_schema
        def consistency_tool(value: str) -> str:
            """A tool for consistency testing.
            
            Args:
                value: Input value
            """
            return value
        
        registry.register(consistency_tool)
        
        all_schemas = registry.get_all_tool_schemas()
        schemas = registry.get_schemas()
        
        # Both should have one entry
        assert len(all_schemas) == len(schemas)
        assert all_schemas[0]["name"] == schemas[0]["name"]
    
    def test_global_registry_get_all_tool_schemas(self):
        """Verify get_all_tool_schemas works on global registry."""
        global_reg = get_global_registry()
        
        # Get current count
        initial_schemas = global_reg.get_all_tool_schemas()
        initial_count = len(initial_schemas)
        
        @tool_schema
        def global_handshake_tool(x: int) -> int:
            """Global handshake tool.
            
            Args:
                x: Input
            """
            return x
        
        global_reg.register(global_handshake_tool)
        
        new_schemas = global_reg.get_all_tool_schemas()
        assert len(new_schemas) == initial_count + 1
