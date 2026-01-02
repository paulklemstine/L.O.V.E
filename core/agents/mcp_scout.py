# core/agents/mcp_scout.py
"""
MCPScout Agent - The "Tool Hunter" for Epic 3: Autonomous Tool Acquisition

This agent searches GitHub for MCP servers, validates them, and generates
configuration snippets for mcp_servers.json.
"""

import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from core.logging import log_event


@dataclass
class MCPServerCandidate:
    """Represents a potential MCP server found on GitHub."""
    repo_full_name: str  # e.g., "owner/repo-name"
    description: str
    stars: int = 0
    has_dockerfile: bool = False
    has_python_deps: bool = False
    has_mcp_mentions: bool = False
    detected_tools: List[str] = field(default_factory=list)
    install_command: str = ""
    install_args: List[str] = field(default_factory=list)
    required_env_vars: List[str] = field(default_factory=list)
    validation_score: float = 0.0
    
    def to_config(self) -> Dict[str, Any]:
        """Generate mcp_servers.json config snippet."""
        # Extract server name from repo
        server_name = self.repo_full_name.split("/")[-1].replace("-", "_")
        server_name = server_name.replace("mcp_server_", "").replace("mcp_", "")
        
        config = {
            server_name: {
                "command": self.install_command or "docker",
                "args": self.install_args or ["run", "-i", "--rm", f"ghcr.io/{self.repo_full_name}"],
                "requires_env": self.required_env_vars,
                "tools": {tool: f"Tool from {self.repo_full_name}" for tool in self.detected_tools}
            }
        }
        return config


class MCPScout:
    """
    The "Tool Hunter" agent that searches GitHub for MCP servers.
    
    Story 3.1: Uses existing GitHub MCP tools to find, validate, and
    generate configurations for new MCP servers.
    """
    
    # Search queries for finding MCP servers
    SEARCH_QUERIES = [
        "mcp-server",
        "model context protocol server",
        "mcp server python",
        "mcp server docker",
    ]
    
    # Files that indicate a valid MCP server
    VALIDATION_FILES = {
        "dockerfile": ["Dockerfile", "dockerfile", "Dockerfile.dev"],
        "python_deps": ["pyproject.toml", "requirements.txt", "setup.py"],
        "mcp_indicators": ["README.md", "README", "readme.md"],
    }
    
    # Keywords that indicate MCP compatibility
    MCP_KEYWORDS = [
        "model context protocol",
        "mcp server",
        "mcp-server",
        "jsonrpc",
        "tools/call",
        "langchain",
    ]
    
    def __init__(self, mcp_manager=None, tool_registry=None):
        """
        Initialize MCPScout.
        
        Args:
            mcp_manager: MCPManager instance for calling GitHub MCP tools
            tool_registry: ToolRegistry for accessing github tools
        """
        self.mcp_manager = mcp_manager
        self.tool_registry = tool_registry
        self._candidates: List[MCPServerCandidate] = []
    
    async def search_mcp_servers(self, capability_query: str = "") -> List[MCPServerCandidate]:
        """
        Search GitHub for MCP servers matching a capability.
        
        Args:
            capability_query: Specific capability to search for (e.g., "postgres", "slack")
            
        Returns:
            List of MCPServerCandidate objects
        """
        log_event(f"MCPScout: Searching for MCP servers with query: {capability_query}", "INFO")
        
        candidates = []
        
        # Build search queries
        queries = []
        if capability_query:
            queries.append(f"mcp server {capability_query}")
            queries.append(f"model context protocol {capability_query}")
        queries.extend(self.SEARCH_QUERIES)
        
        for query in queries[:3]:  # Limit to 3 queries to avoid rate limits
            try:
                results = await self._github_search(query)
                for repo in results:
                    candidate = MCPServerCandidate(
                        repo_full_name=repo.get("full_name", ""),
                        description=repo.get("description", ""),
                        stars=repo.get("stargazers_count", 0),
                    )
                    candidates.append(candidate)
            except Exception as e:
                log_event(f"MCPScout: Search error for '{query}': {e}", "WARNING")
        
        # Deduplicate by repo name
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c.repo_full_name not in seen:
                seen.add(c.repo_full_name)
                unique_candidates.append(c)
        
        self._candidates = unique_candidates
        log_event(f"MCPScout: Found {len(unique_candidates)} unique candidates", "INFO")
        
        return unique_candidates
    
    async def validate_mcp_repo(self, candidate: MCPServerCandidate) -> MCPServerCandidate:
        """
        Validate a repository to determine if it's a usable MCP server.
        
        Checks for:
        - Dockerfile or container definition
        - Python dependencies (pyproject.toml, requirements.txt)
        - MCP/Model Context Protocol mentions in README
        
        Args:
            candidate: The MCPServerCandidate to validate
            
        Returns:
            Updated candidate with validation results
        """
        log_event(f"MCPScout: Validating {candidate.repo_full_name}", "DEBUG")
        
        score = 0.0
        
        # Check for Dockerfile
        try:
            dockerfile = await self._get_file_contents(candidate.repo_full_name, "Dockerfile")
            if dockerfile:
                candidate.has_dockerfile = True
                score += 0.3
                
                # Try to extract install command from Dockerfile
                if "ENTRYPOINT" in dockerfile or "CMD" in dockerfile:
                    candidate.install_command = "docker"
                    candidate.install_args = ["run", "-i", "--rm"]
        except Exception:
            pass
        
        # Check for Python dependencies
        for dep_file in self.VALIDATION_FILES["python_deps"]:
            try:
                deps = await self._get_file_contents(candidate.repo_full_name, dep_file)
                if deps:
                    candidate.has_python_deps = True
                    score += 0.2
                    
                    # Check for MCP library dependency
                    if "mcp" in deps.lower() or "model-context-protocol" in deps.lower():
                        score += 0.2
                    
                    # Extract env var requirements
                    env_vars = self._extract_env_vars(deps)
                    candidate.required_env_vars.extend(env_vars)
                    break
            except Exception:
                pass
        
        # Check README for MCP mentions
        for readme in self.VALIDATION_FILES["mcp_indicators"]:
            try:
                content = await self._get_file_contents(candidate.repo_full_name, readme)
                if content:
                    content_lower = content.lower()
                    mcp_mentions = sum(1 for kw in self.MCP_KEYWORDS if kw in content_lower)
                    if mcp_mentions > 0:
                        candidate.has_mcp_mentions = True
                        score += min(0.3, mcp_mentions * 0.1)
                    
                    # Try to extract tool names from README
                    tools = self._extract_tools_from_readme(content)
                    candidate.detected_tools.extend(tools)
                    break
            except Exception:
                pass
        
        # Bonus for stars (popularity indicator)
        if candidate.stars > 100:
            score += 0.1
        elif candidate.stars > 10:
            score += 0.05
        
        candidate.validation_score = min(1.0, score)
        
        log_event(
            f"MCPScout: {candidate.repo_full_name} validated with score {candidate.validation_score:.2f}",
            "DEBUG"
        )
        
        return candidate
    
    def generate_config(self, candidate: MCPServerCandidate) -> Dict[str, Any]:
        """
        Generate a mcp_servers.json configuration snippet.
        
        Args:
            candidate: Validated MCPServerCandidate
            
        Returns:
            Configuration dictionary ready for mcp_servers.json
        """
        if candidate.validation_score < 0.3:
            log_event(
                f"MCPScout: Low validation score for {candidate.repo_full_name}, config may be unreliable",
                "WARNING"
            )
        
        return candidate.to_config()
    
    async def hunt_capability(self, capability: str) -> Optional[Dict[str, Any]]:
        """
        Complete workflow: Search, validate, and generate config for a capability.
        
        Args:
            capability: The capability to hunt for (e.g., "postgres", "slack")
            
        Returns:
            Best matching server config, or None if none found
        """
        log_event(f"MCPScout: Hunting for capability: {capability}", "INFO")
        
        # Search
        candidates = await self.search_mcp_servers(capability)
        if not candidates:
            log_event(f"MCPScout: No candidates found for '{capability}'", "WARNING")
            return None
        
        # Validate top candidates (limit to avoid rate limits)
        validated = []
        for candidate in candidates[:5]:
            try:
                validated_candidate = await self.validate_mcp_repo(candidate)
                validated.append(validated_candidate)
            except Exception as e:
                log_event(f"MCPScout: Validation error for {candidate.repo_full_name}: {e}", "WARNING")
        
        if not validated:
            return None
        
        # Sort by validation score
        validated.sort(key=lambda c: c.validation_score, reverse=True)
        
        best = validated[0]
        if best.validation_score < 0.3:
            log_event(f"MCPScout: Best candidate score too low ({best.validation_score:.2f})", "WARNING")
            return None
        
        config = self.generate_config(best)
        log_event(f"MCPScout: Generated config for {best.repo_full_name}", "INFO")
        
        return config
    
    # Private helper methods
    
    async def _github_search(self, query: str) -> List[Dict[str, Any]]:
        """Execute GitHub repository search via MCP tool."""
        if not self.mcp_manager:
            log_event("MCPScout: No MCP manager available, using mock search", "DEBUG")
            return []
        
        try:
            # Call the github MCP tool
            request_id = self.mcp_manager.call_tool(
                "github",
                "repos.search_repositories",
                {"query": query, "params": {"per_page": 10}}
            )
            response = self.mcp_manager.get_response("github", request_id, timeout=30)
            
            if "error" in response:
                log_event(f"MCPScout: GitHub search error: {response['error']}", "WARNING")
                return []
            
            result = response.get("result", {})
            return result.get("items", [])
            
        except Exception as e:
            log_event(f"MCPScout: GitHub search exception: {e}", "ERROR")
            return []
    
    async def _get_file_contents(self, repo: str, path: str) -> Optional[str]:
        """Get file contents from GitHub via MCP tool."""
        if not self.mcp_manager:
            return None
        
        try:
            request_id = self.mcp_manager.call_tool(
                "github",
                "repos.get_file_contents",
                {"query": f"{repo}/{path}", "params": {}}
            )
            response = self.mcp_manager.get_response("github", request_id, timeout=30)
            
            if "error" in response:
                return None
            
            result = response.get("result", {})
            content = result.get("content", "")
            
            # GitHub returns base64 encoded content
            if result.get("encoding") == "base64":
                import base64
                content = base64.b64decode(content).decode("utf-8", errors="ignore")
            
            return content
            
        except Exception:
            return None
    
    def _extract_env_vars(self, deps_content: str) -> List[str]:
        """Extract environment variable names from dependency/config files."""
        env_vars = []
        
        # Common patterns for env vars
        patterns = [
            r'\bAPI_KEY\b',
            r'\b[A-Z_]+_TOKEN\b',
            r'\b[A-Z_]+_SECRET\b',
            r'\b[A-Z_]+_PASSWORD\b',
            r'os\.environ\[[\'"]([\w_]+)[\'"]\]',
            r'os\.getenv\([\'"]([\w_]+)[\'"]',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, deps_content)
            env_vars.extend(matches if isinstance(matches[0] if matches else "", str) else [m for m in matches])
        
        return list(set(env_vars))
    
    def _extract_tools_from_readme(self, readme_content: str) -> List[str]:
        """Extract tool names from README documentation."""
        tools = []
        
        # Look for tool definitions in common formats
        patterns = [
            r'`(\w+\.\w+)`',  # e.g., `repos.search_repositories`
            r'tools?:\s*\n((?:[-*]\s*.+\n)+)',  # List of tools
        ]
        
        for pattern in patterns[:1]:  # Just the first pattern for now
            matches = re.findall(pattern, readme_content)
            for match in matches:
                if "." in match and len(match) < 50:  # Reasonable tool name
                    tools.append(match)
        
        return tools[:10]  # Limit to 10 tools
