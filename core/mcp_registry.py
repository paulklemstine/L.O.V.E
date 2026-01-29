"""
MCP Registry Discovery and Public Integration.

Enables agents to discover and install MCP servers from public registries,
implementing the "Open Agentic Web" vision where agents can dynamically
extend their capabilities by finding tools they weren't originally programmed with.

Key Features:
1. search_registry(query) - Search public MCP registries for servers
2. install_server(server_info) - Install a discovered server
3. list_installed() - List locally installed MCP servers
4. capability_negotiation() - Handle MCP listChanged notifications

Supported Registries:
- mcp.so - Anthropic's official MCP server registry
- Smithery.ai - Community MCP server registry
- GitHub MCP Registry - Curated list of MCP servers

References:
- Design doc: "The Architectural Evolution of Open Source AI"
- MCP Specification: https://modelcontextprotocol.io
"""

import os
import sys
import subprocess
import tempfile
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


# =============================================================================
# Docker Detection and Management
# =============================================================================

class DockerManager:
    """Manages Docker availability detection and installation."""
    
    _docker_available: Optional[bool] = None
    
    @classmethod
    def is_docker_available(cls) -> bool:
        """Check if Docker is available on the system."""
        if cls._docker_available is not None:
            return cls._docker_available
            
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10
            )
            cls._docker_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            cls._docker_available = False
            
        logger.info(f"Docker available: {cls._docker_available}")
        return cls._docker_available
    
    @classmethod
    def get_install_instructions(cls) -> str:
        """Get platform-specific Docker installation instructions."""
        if sys.platform == "linux":
            return "curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER"
        elif sys.platform == "darwin":
            return "brew install --cask docker"
        elif sys.platform == "win32":
            return "winget install Docker.DockerDesktop"
        return "Visit https://docs.docker.com/get-docker/"
    
    @classmethod
    async def attempt_install(cls) -> Tuple[bool, str]:
        """Attempt to install Docker automatically (Linux only)."""
        if sys.platform != "linux":
            return False, f"Auto-install only on Linux. Manual: {cls.get_install_instructions()}"
        
        try:
            # Download and run Docker install script
            result = subprocess.run(
                ["bash", "-c", "curl -fsSL https://get.docker.com | sh"],
                capture_output=True,
                timeout=300
            )
            if result.returncode == 0:
                cls._docker_available = None  # Reset cache
                return True, "Docker installed successfully. Please log out/in for group permissions."
            return False, f"Installation failed: {result.stderr.decode()}"
        except Exception as e:
            return False, f"Installation failed: {e}"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MCPServerInfo:
    """Information about an MCP server from a registry."""
    id: str
    name: str
    description: str
    author: str = ""
    registry: str = ""  # "mcp.so", "smithery.ai", etc.
    install_command: Optional[str] = None
    npm_package: Optional[str] = None
    github_url: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    stars: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "author": self.author,
            "registry": self.registry,
            "install_command": self.install_command,
            "npm_package": self.npm_package,
            "github_url": self.github_url,
            "categories": self.categories,
            "tools": self.tools,
            "stars": self.stars
        }


# =============================================================================
# MCP Registry Client
# =============================================================================

class MCPRegistry:
    """
    Public MCP Registry Discovery and Installation.
    
    Enables agents to discover and install MCP servers from public registries.
    
    Usage:
        registry = MCPRegistry()
        servers = await registry.search("weather")
        success, msg = await registry.install(servers[0])
    """
    
    # Registry endpoints
    REGISTRIES = {
        "mcp.so": {
            "search_url": "https://mcp.so/api/servers/search",
            "detail_url": "https://mcp.so/api/servers/{id}",
            "enabled": True
        },
        "smithery.ai": {
            "search_url": "https://registry.smithery.ai/servers",
            "detail_url": "https://registry.smithery.ai/servers/{id}",
            "enabled": True
        },
        "github": {
            "search_url": "https://raw.githubusercontent.com/modelcontextprotocol/servers/main/registry.json",
            "detail_url": None,
            "enabled": True
        }
    }
    
    def __init__(self, install_dir: Optional[str] = None):
        """
        Initialize the MCP Registry client.
        
        Args:
            install_dir: Directory for installed servers (default: ./mcp_servers)
        """
        self.install_dir = install_dir or os.path.join(
            os.path.dirname(__file__), "..", "mcp_servers"
        )
        os.makedirs(self.install_dir, exist_ok=True)
        
        self._cache: Dict[str, List[MCPServerInfo]] = {}
        self._config_path = os.path.join(os.path.dirname(__file__), "mcp_servers.json")
    
    async def search(
        self,
        query: str,
        registries: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[MCPServerInfo]:
        """
        Search public MCP registries for servers matching a capability.
        
        Args:
            query: Search query (e.g., "weather", "database", "github")
            registries: Optional list of registries to search (default: all enabled)
            limit: Maximum results per registry
            
        Returns:
            List of MCPServerInfo objects sorted by relevance
        """
        registries = registries or [k for k, v in self.REGISTRIES.items() if v["enabled"]]
        results = []
        
        for registry_name in registries:
            try:
                registry_results = await self._search_registry(registry_name, query, limit)
                results.extend(registry_results)
            except Exception as e:
                logger.warning(f"Failed to search {registry_name}: {e}")
        
        # Sort by stars/popularity
        results.sort(key=lambda x: x.stars, reverse=True)
        
        logger.info(f"Found {len(results)} servers for query '{query}'")
        return results[:limit * 2]
    
    async def _search_registry(
        self,
        registry_name: str,
        query: str,
        limit: int
    ) -> List[MCPServerInfo]:
        """Search a specific registry."""
        config = self.REGISTRIES.get(registry_name)
        if not config or not config.get("enabled"):
            return []
        
        # Use urllib for sync HTTP (aiohttp may not be installed)
        import urllib.request
        import urllib.parse
        
        try:
            search_url = config["search_url"]
            
            if registry_name == "github":
                req = urllib.request.Request(
                    search_url,
                    headers={"User-Agent": "L.O.V.E-Agent/2.0"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                    return self._parse_github_registry(data, query, limit)
            else:
                params = urllib.parse.urlencode({"q": query, "limit": limit})
                full_url = f"{search_url}?{params}"
                req = urllib.request.Request(
                    full_url,
                    headers={"User-Agent": "L.O.V.E-Agent/2.0"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    data = json.loads(resp.read().decode())
                    
                    if registry_name == "mcp.so":
                        return self._parse_mcp_so_response(data, limit)
                    elif registry_name == "smithery.ai":
                        return self._parse_smithery_response(data, limit)
                        
        except Exception as e:
            logger.warning(f"Search failed for {registry_name}: {e}")
        
        return []
    
    def _parse_mcp_so_response(self, data: Any, limit: int) -> List[MCPServerInfo]:
        """Parse response from mcp.so registry."""
        servers = []
        items = data.get("servers", data) if isinstance(data, dict) else data
        
        for item in (items or [])[:limit]:
            if isinstance(item, dict):
                servers.append(MCPServerInfo(
                    id=item.get("id", item.get("name", "")),
                    name=item.get("name", ""),
                    description=item.get("description", "")[:200],
                    author=item.get("author", item.get("publisher", "")),
                    registry="mcp.so",
                    npm_package=item.get("npm_package"),
                    github_url=item.get("github_url", item.get("repository")),
                    categories=item.get("categories", []),
                    tools=item.get("tools", []),
                    stars=item.get("stars", 0)
                ))
        
        return servers
    
    def _parse_smithery_response(self, data: Any, limit: int) -> List[MCPServerInfo]:
        """Parse response from Smithery.ai registry."""
        servers = []
        items = data.get("items", data) if isinstance(data, dict) else data
        
        for item in (items or [])[:limit]:
            if isinstance(item, dict):
                servers.append(MCPServerInfo(
                    id=item.get("id", item.get("slug", "")),
                    name=item.get("name", item.get("title", "")),
                    description=item.get("description", "")[:200],
                    author=item.get("author", ""),
                    registry="smithery.ai",
                    install_command=item.get("install_command"),
                    github_url=item.get("github"),
                    categories=item.get("tags", []),
                    tools=item.get("tools", []),
                    stars=item.get("downloads", 0)
                ))
        
        return servers
    
    def _parse_github_registry(
        self,
        data: Any,
        query: str,
        limit: int
    ) -> List[MCPServerInfo]:
        """Parse response from GitHub MCP registry."""
        servers = []
        items = data.get("servers", data) if isinstance(data, dict) else data
        query_lower = query.lower()
        
        for item in (items or []):
            if isinstance(item, dict):
                name = item.get("name", "")
                desc = item.get("description", "")
                
                # Filter by query
                if query_lower not in name.lower() and query_lower not in desc.lower():
                    continue
                
                servers.append(MCPServerInfo(
                    id=item.get("id", name),
                    name=name,
                    description=desc[:200],
                    author=item.get("author", ""),
                    registry="github",
                    npm_package=item.get("npm"),
                    github_url=item.get("github"),
                    categories=item.get("categories", []),
                    tools=[],
                    stars=item.get("stars", 0)
                ))
                
                if len(servers) >= limit:
                    break
        
        return servers
    
    async def install(
        self,
        server: MCPServerInfo,
        auto_configure: bool = True
    ) -> Tuple[bool, str]:
        """
        Install an MCP server from a registry.
        
        Args:
            server: MCPServerInfo object from search results
            auto_configure: Whether to add to local MCP configuration
            
        Returns:
            Tuple of (success, message)
        """
        logger.info(f"Installing MCP server: {server.name}")
        
        # Determine installation method
        if server.npm_package:
            return await self._install_npm(server, auto_configure)
        elif server.github_url:
            return await self._install_git(server, auto_configure)
        elif server.install_command:
            return await self._install_custom(server, auto_configure)
        else:
            return False, f"No installation method available for {server.name}"
    
    async def _install_npm(
        self,
        server: MCPServerInfo,
        auto_configure: bool
    ) -> Tuple[bool, str]:
        """Install an MCP server from npm."""
        try:
            # Check npm availability
            npm_check = subprocess.run(["npm", "--version"], capture_output=True, timeout=5)
            if npm_check.returncode != 0:
                return False, "npm not available. Install Node.js first."
            
            # Install globally
            result = subprocess.run(
                ["npm", "install", "-g", server.npm_package],
                capture_output=True,
                timeout=120
            )
            
            if result.returncode != 0:
                return False, f"npm install failed: {result.stderr.decode()}"
            
            if auto_configure:
                self._add_to_config(server)
            
            return True, f"Successfully installed {server.npm_package}"
            
        except subprocess.TimeoutExpired:
            return False, "Installation timed out"
        except Exception as e:
            return False, f"Installation failed: {e}"
    
    async def _install_git(
        self,
        server: MCPServerInfo,
        auto_configure: bool
    ) -> Tuple[bool, str]:
        """Install an MCP server from GitHub."""
        try:
            server_dir = os.path.join(self.install_dir, server.id)
            
            if os.path.exists(server_dir):
                result = subprocess.run(
                    ["git", "pull"],
                    cwd=server_dir,
                    capture_output=True,
                    timeout=60
                )
            else:
                result = subprocess.run(
                    ["git", "clone", server.github_url, server_dir],
                    capture_output=True,
                    timeout=120
                )
            
            if result.returncode != 0:
                return False, f"git operation failed: {result.stderr.decode()}"
            
            # Install npm dependencies if present
            package_json = os.path.join(server_dir, "package.json")
            if os.path.exists(package_json):
                subprocess.run(
                    ["npm", "install"],
                    cwd=server_dir,
                    capture_output=True,
                    timeout=120
                )
            
            if auto_configure:
                self._add_to_config(server, server_dir)
            
            return True, f"Successfully cloned {server.name}"
            
        except Exception as e:
            return False, f"Installation failed: {e}"
    
    async def _install_custom(
        self,
        server: MCPServerInfo,
        auto_configure: bool
    ) -> Tuple[bool, str]:
        """Install using custom install command."""
        try:
            result = subprocess.run(
                server.install_command.split(),
                capture_output=True,
                timeout=120
            )
            
            if result.returncode != 0:
                return False, f"Install command failed: {result.stderr.decode()}"
            
            if auto_configure:
                self._add_to_config(server)
            
            return True, f"Successfully installed {server.name}"
            
        except Exception as e:
            return False, f"Installation failed: {e}"
    
    def _add_to_config(
        self,
        server: MCPServerInfo,
        server_dir: Optional[str] = None
    ):
        """Add installed server to local MCP configuration."""
        try:
            # Load existing config
            if os.path.exists(self._config_path):
                with open(self._config_path, 'r') as f:
                    config = json.load(f)
            else:
                config = {}
            
            # Build server config
            server_config = {
                "description": server.description,
                "type": "stdio",
                "tools": {t: f"Tool: {t}" for t in server.tools} if server.tools else {},
                "source": server.registry
            }
            
            if server.npm_package:
                server_config["command"] = "npx"
                server_config["args"] = [server.npm_package]
            elif server_dir:
                for entry in ["index.js", "server.js", "src/index.js"]:
                    entry_path = os.path.join(server_dir, entry)
                    if os.path.exists(entry_path):
                        server_config["command"] = "node"
                        server_config["args"] = [entry_path]
                        break
            
            config[server.id] = server_config
            
            # Save config
            with open(self._config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Added {server.name} to MCP config")
            
        except Exception as e:
            logger.warning(f"Failed to add to config: {e}")
    
    def list_installed(self) -> List[str]:
        """List locally installed MCP servers."""
        installed = []
        
        # From install directory
        if os.path.exists(self.install_dir):
            for name in os.listdir(self.install_dir):
                path = os.path.join(self.install_dir, name)
                if os.path.isdir(path):
                    installed.append(name)
        
        # From config file
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, 'r') as f:
                    config = json.load(f)
                    for server_id in config.keys():
                        if server_id not in installed:
                            installed.append(server_id)
            except:
                pass
        
        return installed


# =============================================================================
# Global Instance & Convenience Functions
# =============================================================================

_registry_instance: Optional[MCPRegistry] = None


def get_mcp_registry() -> MCPRegistry:
    """Get or create the global MCPRegistry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = MCPRegistry()
    return _registry_instance


async def search_mcp_servers(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Convenience function to search MCP registries.
    
    Args:
        query: Search query (e.g., "weather", "database")
        limit: Maximum results
        
    Returns:
        List of server info dicts
    """
    registry = get_mcp_registry()
    results = await registry.search(query, limit=limit)
    return [r.to_dict() for r in results]


async def install_mcp_server(server_id: str, registry_name: str = "mcp.so") -> Tuple[bool, str]:
    """
    Convenience function to install an MCP server.
    
    Args:
        server_id: Server ID from search results
        registry_name: Registry to install from
        
    Returns:
        Tuple of (success, message)
    """
    registry = get_mcp_registry()
    results = await registry.search(server_id, registries=[registry_name], limit=5)
    
    for server in results:
        if server.id == server_id:
            return await registry.install(server)
    
    return False, f"Server '{server_id}' not found in {registry_name}"


def reset_mcp_registry():
    """Reset the global MCPRegistry instance (for testing)."""
    global _registry_instance
    _registry_instance = None
