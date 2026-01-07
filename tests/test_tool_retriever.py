"""
DeepAgent Protocol - Story 2.3: Tool Retriever Tests

Tests for the dynamic tool retrieval system using semantic similarity.
"""

import pytest
from typing import Dict, Any


class MockSchema:
    """Mock tool schema for testing."""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description


class MockToolRegistry:
    """Mock tool registry for testing without full system dependencies."""
    
    def __init__(self):
        self._tools = {
            "generate_image": {
                "name": "generate_image",
                "description": "Generate an image using AI. Creates visual artwork from text prompts.",
                "parameters": {
                    "properties": {
                        "prompt": {"description": "Visual description of the image"},
                        "width": {"description": "Width in pixels"},
                        "height": {"description": "Height in pixels"}
                    }
                }
            },
            "post_to_bluesky": {
                "name": "post_to_bluesky",
                "description": "Post a message to Bluesky social media platform with optional image.",
                "parameters": {
                    "properties": {
                        "text": {"description": "The post content"},
                        "image": {"description": "Optional image to attach"}
                    }
                }
            },
            "get_notifications": {
                "name": "get_notifications",
                "description": "Fetch recent notifications from Bluesky.",
                "parameters": {
                    "properties": {
                        "limit": {"description": "Maximum notifications to fetch"}
                    }
                }
            },
            "read_file": {
                "name": "read_file",
                "description": "Read contents of a file from the filesystem.",
                "parameters": {
                    "properties": {
                        "filepath": {"description": "Path to the file"}
                    }
                }
            },
            "analyze_code": {
                "name": "analyze_code",
                "description": "Analyze Python code for quality and issues.",
                "parameters": {
                    "properties": {
                        "code": {"description": "Code to analyze"}
                    }
                }
            },
            "search_memories": {
                "name": "search_memories",
                "description": "Search through stored memories and past interactions.",
                "parameters": {
                    "properties": {
                        "query": {"description": "Search query"}
                    }
                }
            }
        }
    
    def list_tools(self):
        return list(self._tools.keys())
    
    def get_schema(self, name: str) -> Dict[str, Any]:
        return self._tools.get(name)


class TestToolRetriever:
    """Test suite for ToolRetriever class."""
    
    @pytest.fixture
    def retriever(self):
        """Create a ToolRetriever instance."""
        from core.tool_retriever import ToolRetriever
        return ToolRetriever(similarity_threshold=0.2)
    
    @pytest.fixture
    def mock_registry(self):
        """Create a mock registry."""
        return MockToolRegistry()
    
    def test_tool_indexing(self, retriever, mock_registry):
        """Test that tools are properly indexed."""
        retriever.index_tools(mock_registry)
        
        assert len(retriever._tool_cache) == 6
        assert "generate_image" in retriever._tool_cache
        assert "post_to_bluesky" in retriever._tool_cache
    
    def test_retrieve_image_tools(self, retriever, mock_registry):
        """Test retrieval of image-related tools."""
        retriever.index_tools(mock_registry)
        
        matches = retriever.retrieve("Generate an image of a sunset", max_tools=3)
        
        # Should find generate_image with high relevance
        tool_names = [m.name for m in matches]
        assert "generate_image" in tool_names
    
    def test_retrieve_social_tools(self, retriever, mock_registry):
        """Test retrieval of social media tools."""
        retriever.index_tools(mock_registry)
        
        matches = retriever.retrieve("Post a message to Bluesky", max_tools=3)
        
        tool_names = [m.name for m in matches]
        # Should find bluesky-related tools
        assert "post_to_bluesky" in tool_names or "get_notifications" in tool_names
    
    def test_retrieve_file_tools(self, retriever, mock_registry):
        """Test retrieval of file-related tools."""
        retriever.index_tools(mock_registry)
        
        matches = retriever.retrieve("Read the configuration file", max_tools=3)
        
        tool_names = [m.name for m in matches]
        assert "read_file" in tool_names
    
    def test_retrieve_memory_tools(self, retriever, mock_registry):
        """Test retrieval of memory-related tools."""
        retriever.index_tools(mock_registry)
        
        matches = retriever.retrieve("Remember what we discussed earlier", max_tools=3)
        
        tool_names = [m.name for m in matches]
        assert "search_memories" in tool_names
    
    def test_max_tools_limit(self, retriever, mock_registry):
        """Test that max_tools limit is respected."""
        retriever.index_tools(mock_registry)
        
        matches = retriever.retrieve("Do something with files and images", max_tools=2)
        
        assert len(matches) <= 2
    
    def test_empty_query(self, retriever, mock_registry):
        """Test handling of empty query."""
        retriever.index_tools(mock_registry)
        
        matches = retriever.retrieve("", max_tools=3)
        
        # Should return empty or very few results
        assert len(matches) <= 3
    
    def test_format_tools_subset(self, retriever, mock_registry):
        """Test formatting of tool subset for prompts."""
        retriever.index_tools(mock_registry)
        
        matches = retriever.retrieve("Generate an image", max_tools=2)
        formatted = retriever.get_tool_subset_metadata(matches)
        
        assert "generate_image" in formatted or "selected" in formatted
        assert "relevance:" in formatted


class TestToolCategories:
    """Test the category-based pre-filtering."""
    
    def test_image_category_keywords(self):
        """Verify image category keywords are correct."""
        from core.tool_retriever import ToolRetriever
        
        retriever = ToolRetriever()
        
        assert "image" in retriever.TOOL_CATEGORIES
        assert "generate" in retriever.TOOL_CATEGORIES["image"]
        assert "visual" in retriever.TOOL_CATEGORIES["image"]
    
    def test_social_category_keywords(self):
        """Verify social category keywords are correct."""
        from core.tool_retriever import ToolRetriever
        
        retriever = ToolRetriever()
        
        assert "social" in retriever.TOOL_CATEGORIES
        assert "post" in retriever.TOOL_CATEGORIES["social"]
        assert "bluesky" in retriever.TOOL_CATEGORIES["social"]


class TestConvenienceFunctions:
    """Test the module-level convenience functions."""
    
    def test_get_tool_retriever_singleton(self):
        """Test that get_tool_retriever returns a singleton."""
        from core.tool_retriever import get_tool_retriever
        
        r1 = get_tool_retriever()
        r2 = get_tool_retriever()
        
        assert r1 is r2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
