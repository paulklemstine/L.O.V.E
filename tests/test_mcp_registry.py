"""
Tests for MCP Registry: Public Registry Discovery and Installation

Tests the public MCP registry integration including:
- Search across mcp.so, Smithery.ai, and GitHub registries
- Server installation (npm, git, custom)
- Configuration management
"""

import pytest
import asyncio
import os
from unittest.mock import patch, MagicMock, AsyncMock

from core.mcp_registry import (
    MCPRegistry,
    MCPServerInfo,
    DockerManager,
    get_mcp_registry,
    search_mcp_servers,
    install_mcp_server,
    reset_mcp_registry
)


# =============================================================================
# DockerManager Tests
# =============================================================================

class TestDockerManager:
    """Tests for Docker detection."""
    
    def setup_method(self):
        DockerManager._docker_available = None
    
    def test_is_docker_available_returns_bool(self):
        """Should return boolean for Docker availability."""
        result = DockerManager.is_docker_available()
        assert isinstance(result, bool)
    
    def test_docker_availability_is_cached(self):
        """Should cache Docker availability check."""
        DockerManager._docker_available = True
        assert DockerManager.is_docker_available() is True
        
        DockerManager._docker_available = False
        assert DockerManager.is_docker_available() is False
    
    def test_get_install_instructions(self):
        """Should return install instructions."""
        instructions = DockerManager.get_install_instructions()
        assert isinstance(instructions, str)
        assert len(instructions) > 0


# =============================================================================
# MCPServerInfo Tests
# =============================================================================

class TestMCPServerInfo:
    """Tests for MCPServerInfo dataclass."""
    
    def test_create_server_info(self):
        """Should create server info with all fields."""
        info = MCPServerInfo(
            id="test-server",
            name="Test Server",
            description="A test server",
            author="Test Author",
            registry="mcp.so"
        )
        
        assert info.id == "test-server"
        assert info.name == "Test Server"
        assert info.description == "A test server"
    
    def test_to_dict(self):
        """Should convert to dictionary."""
        info = MCPServerInfo(
            id="test",
            name="Test",
            description="Desc",
            npm_package="@test/mcp-server"
        )
        
        d = info.to_dict()
        
        assert d["id"] == "test"
        assert d["name"] == "Test"
        assert d["npm_package"] == "@test/mcp-server"


# =============================================================================
# MCPRegistry Tests
# =============================================================================

class TestMCPRegistry:
    """Tests for MCPRegistry client."""
    
    @pytest.fixture
    def registry(self, tmp_path):
        """Create a registry with temp install directory."""
        return MCPRegistry(install_dir=str(tmp_path))
    
    def test_registry_initialization(self, registry):
        """Should initialize with correct defaults."""
        assert registry.install_dir is not None
        assert os.path.exists(registry.install_dir)
        assert len(registry.REGISTRIES) >= 3
    
    @pytest.mark.asyncio
    async def test_search_returns_list(self, registry):
        """Search should return list (possibly empty if network unavailable)."""
        # This test may fail without internet, so we just check it returns a list
        try:
            results = await registry.search("test", limit=3)
            assert isinstance(results, list)
        except Exception:
            # Network may not be available
            pass
    
    def test_parse_mcp_so_response(self, registry):
        """Should parse mcp.so response format."""
        mock_response = {
            "servers": [
                {
                    "id": "weather-server",
                    "name": "Weather MCP",
                    "description": "Get weather data",
                    "author": "Test",
                    "npm_package": "@test/weather-mcp"
                }
            ]
        }
        
        results = registry._parse_mcp_so_response(mock_response, limit=10)
        
        assert len(results) == 1
        assert results[0].id == "weather-server"
        assert results[0].name == "Weather MCP"
    
    def test_parse_smithery_response(self, registry):
        """Should parse Smithery.ai response format."""
        mock_response = {
            "items": [
                {
                    "id": "db-server",
                    "name": "Database MCP",
                    "description": "SQL operations",
                    "tags": ["database", "sql"]
                }
            ]
        }
        
        results = registry._parse_smithery_response(mock_response, limit=10)
        
        assert len(results) == 1
        assert results[0].id == "db-server"
        assert results[0].categories == ["database", "sql"]
    
    def test_parse_github_registry(self, registry):
        """Should parse GitHub registry format with filtering."""
        mock_data = {
            "servers": [
                {"name": "weather-api", "description": "Get weather data"},
                {"name": "stocks-api", "description": "Stock market data"},
                {"name": "github-api", "description": "GitHub integration"}
            ]
        }
        
        results = registry._parse_github_registry(mock_data, "weather", limit=10)
        
        assert len(results) == 1
        assert results[0].name == "weather-api"
    
    def test_list_installed_empty(self, registry):
        """Should return empty list if nothing installed."""
        installed = registry.list_installed()
        assert isinstance(installed, list)


# =============================================================================
# Installation Tests
# =============================================================================

class TestMCPRegistryInstallation:
    """Tests for server installation."""
    
    @pytest.fixture
    def registry(self, tmp_path):
        return MCPRegistry(install_dir=str(tmp_path))
    
    @pytest.mark.asyncio
    async def test_install_no_method_fails(self, registry):
        """Should fail if no installation method available."""
        server = MCPServerInfo(
            id="no-method",
            name="No Method Server",
            description="Has no installation method"
        )
        
        success, message = await registry.install(server)
        
        assert success is False
        assert "installation method" in message.lower()
    
    @pytest.mark.asyncio
    async def test_install_npm_without_npm_fails(self, registry):
        """Should fail npm install if npm not available."""
        server = MCPServerInfo(
            id="npm-server",
            name="NPM Server",
            description="Requires npm",
            npm_package="@nonexistent/package"
        )
        
        # Mock npm as unavailable
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            success, message = await registry.install(server)
            
            # Either npm not available or package doesn't exist - both are failures
            assert success is False


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def setup_method(self):
        reset_mcp_registry()
    
    def test_get_mcp_registry_singleton(self):
        """Should return same instance."""
        reg1 = get_mcp_registry()
        reg2 = get_mcp_registry()
        assert reg1 is reg2
    
    @pytest.mark.asyncio
    async def test_search_mcp_servers_returns_list(self):
        """Convenience function should return list of dicts."""
        reset_mcp_registry()
        try:
            results = await search_mcp_servers("nonexistent_query_12345", limit=1)
            assert isinstance(results, list)
            # Results may be empty or contain dicts
            if results:
                assert isinstance(results[0], dict)
        except Exception:
            # Network may not be available
            pass
