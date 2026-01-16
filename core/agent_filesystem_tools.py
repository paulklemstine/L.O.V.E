"""
Agent Filesystem Tools - Safe filesystem access for agents.

Provides tools for agents to create and manage working documents
in a sandboxed agent workspace. Supports markdown, JSON, YAML, and text files.
"""

import json
import os
import time
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from core.logging import log_event


# Agent workspace root - all agent file operations are confined here
AGENT_WORKSPACE = Path(".agent_workspace")


@dataclass
class FileOperationResult:
    """Result of a file operation."""
    success: bool
    path: str
    message: str
    content: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "path": self.path,
            "message": self.message,
            "content": self.content,
        }


class AgentFilesystemTools:
    """
    Safe filesystem tools for agents to create and manage working documents.
    
    All operations are confined to the .agent_workspace directory.
    Supports:
    - Markdown files (.md)
    - JSON files (.json)
    - YAML files (.yaml, .yml)
    - Text files (.txt)
    - Any other text-based files
    """
    
    # Maximum file size for read/write operations (1MB)
    MAX_FILE_SIZE = 1024 * 1024
    
    # Allowed file extensions (for safety)
    ALLOWED_EXTENSIONS = {
        ".md", ".json", ".yaml", ".yml", ".txt", 
        ".py", ".js", ".html", ".css", ".log"
    }
    
    def __init__(self, workspace_path: Path = None):
        """
        Initialize the filesystem tools.
        
        Args:
            workspace_path: Optional custom workspace path
        """
        self.workspace = workspace_path or AGENT_WORKSPACE
        self._ensure_workspace()
    
    def _ensure_workspace(self):
        """Ensure the workspace directory exists."""
        self.workspace.mkdir(parents=True, exist_ok=True)
        log_event(f"[AgentFS] Workspace initialized at: {self.workspace.absolute()}", "DEBUG")
    
    def _safe_path(self, filename: str) -> Optional[Path]:
        """
        Get a safe path within the workspace.
        Returns None if path escapes workspace.
        """
        # Normalize and resolve path
        clean_name = filename.replace("\\", "/").lstrip("/")
        full_path = (self.workspace / clean_name).resolve()
        
        # Ensure path is within workspace
        try:
            full_path.relative_to(self.workspace.resolve())
        except ValueError:
            log_event(f"[AgentFS] Path escape attempt blocked: {filename}", "WARNING")
            return None
        
        return full_path
    
    def _check_extension(self, path: Path) -> bool:
        """Check if file extension is allowed."""
        return path.suffix.lower() in self.ALLOWED_EXTENSIONS or path.suffix == ""
    
    # ==================== CORE FILE OPERATIONS ====================
    
    async def write_file(
        self, 
        filename: str, 
        content: str,
        overwrite: bool = True
    ) -> FileOperationResult:
        """
        Write content to a file in the workspace.
        
        Args:
            filename: Name/path of file relative to workspace
            content: Content to write
            overwrite: Whether to overwrite existing file
            
        Returns:
            FileOperationResult
        """
        path = self._safe_path(filename)
        if not path:
            return FileOperationResult(
                success=False,
                path=filename,
                message="Invalid path - must be within workspace"
            )
        
        if not self._check_extension(path):
            return FileOperationResult(
                success=False,
                path=filename,
                message=f"Extension not allowed. Allowed: {self.ALLOWED_EXTENSIONS}"
            )
        
        if len(content) > self.MAX_FILE_SIZE:
            return FileOperationResult(
                success=False,
                path=filename,
                message=f"Content too large. Max size: {self.MAX_FILE_SIZE} bytes"
            )
        
        if path.exists() and not overwrite:
            return FileOperationResult(
                success=False,
                path=str(path),
                message="File exists and overwrite=False"
            )
        
        try:
            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            
            log_event(f"[AgentFS] Wrote file: {path}", "INFO")
            return FileOperationResult(
                success=True,
                path=str(path),
                message=f"Successfully wrote {len(content)} bytes"
            )
        except Exception as e:
            log_event(f"[AgentFS] Write failed: {e}", "ERROR")
            return FileOperationResult(
                success=False,
                path=str(path),
                message=f"Write failed: {e}"
            )
    
    async def read_file(self, filename: str) -> FileOperationResult:
        """
        Read content from a file in the workspace.
        
        Args:
            filename: Name/path of file relative to workspace
            
        Returns:
            FileOperationResult with content
        """
        path = self._safe_path(filename)
        if not path:
            return FileOperationResult(
                success=False,
                path=filename,
                message="Invalid path"
            )
        
        if not path.exists():
            return FileOperationResult(
                success=False,
                path=str(path),
                message="File not found"
            )
        
        try:
            content = path.read_text(encoding="utf-8")
            return FileOperationResult(
                success=True,
                path=str(path),
                message=f"Read {len(content)} bytes",
                content=content
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                path=str(path),
                message=f"Read failed: {e}"
            )
    
    async def append_file(self, filename: str, content: str) -> FileOperationResult:
        """Append content to a file."""
        path = self._safe_path(filename)
        if not path:
            return FileOperationResult(
                success=False, path=filename, message="Invalid path"
            )
        
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(content)
            return FileOperationResult(
                success=True, path=str(path), message=f"Appended {len(content)} bytes"
            )
        except Exception as e:
            return FileOperationResult(
                success=False, path=str(path), message=f"Append failed: {e}"
            )
    
    async def delete_file(self, filename: str) -> FileOperationResult:
        """Delete a file from the workspace."""
        path = self._safe_path(filename)
        if not path:
            return FileOperationResult(
                success=False, path=filename, message="Invalid path"
            )
        
        if not path.exists():
            return FileOperationResult(
                success=True, path=str(path), message="File already deleted"
            )
        
        try:
            path.unlink()
            log_event(f"[AgentFS] Deleted file: {path}", "INFO")
            return FileOperationResult(
                success=True, path=str(path), message="File deleted"
            )
        except Exception as e:
            return FileOperationResult(
                success=False, path=str(path), message=f"Delete failed: {e}"
            )
    
    async def list_files(self, subdirectory: str = "") -> FileOperationResult:
        """List files in the workspace or a subdirectory."""
        path = self._safe_path(subdirectory) if subdirectory else self.workspace
        if not path:
            return FileOperationResult(
                success=False, path=subdirectory, message="Invalid path"
            )
        
        if not path.exists():
            return FileOperationResult(
                success=False, path=str(path), message="Directory not found"
            )
        
        try:
            files = []
            for item in path.iterdir():
                rel_path = item.relative_to(self.workspace)
                files.append({
                    "name": item.name,
                    "path": str(rel_path),
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else 0,
                })
            
            return FileOperationResult(
                success=True,
                path=str(path),
                message=f"Found {len(files)} items",
                content=json.dumps(files, indent=2)
            )
        except Exception as e:
            return FileOperationResult(
                success=False, path=str(path), message=f"List failed: {e}"
            )
    
    # ==================== SPECIALIZED FILE OPERATIONS ====================
    
    async def write_json(
        self, 
        filename: str, 
        data: Union[Dict, List],
        pretty: bool = True
    ) -> FileOperationResult:
        """Write JSON data to a file."""
        if not filename.endswith(".json"):
            filename += ".json"
        
        try:
            content = json.dumps(data, indent=2 if pretty else None, default=str)
            return await self.write_file(filename, content)
        except Exception as e:
            return FileOperationResult(
                success=False, path=filename, message=f"JSON encode failed: {e}"
            )
    
    async def read_json(self, filename: str) -> FileOperationResult:
        """Read and parse JSON from a file."""
        result = await self.read_file(filename)
        if not result.success:
            return result
        
        try:
            data = json.loads(result.content)
            result.content = data  # Replace string with parsed data
            return result
        except Exception as e:
            return FileOperationResult(
                success=False, path=result.path, message=f"JSON parse failed: {e}"
            )
    
    async def write_yaml(
        self, 
        filename: str, 
        data: Union[Dict, List]
    ) -> FileOperationResult:
        """Write YAML data to a file."""
        if not filename.endswith((".yaml", ".yml")):
            filename += ".yaml"
        
        try:
            content = yaml.safe_dump(data, default_flow_style=False, allow_unicode=True)
            return await self.write_file(filename, content)
        except Exception as e:
            return FileOperationResult(
                success=False, path=filename, message=f"YAML encode failed: {e}"
            )
    
    async def read_yaml(self, filename: str) -> FileOperationResult:
        """Read and parse YAML from a file."""
        result = await self.read_file(filename)
        if not result.success:
            return result
        
        try:
            data = yaml.safe_load(result.content)
            result.content = data
            return result
        except Exception as e:
            return FileOperationResult(
                success=False, path=result.path, message=f"YAML parse failed: {e}"
            )
    
    # ==================== STANDARD AGENT DOCUMENTS ====================
    
    async def write_agents_md(self, agents: List[Dict]) -> FileOperationResult:
        """
        Write active agents status to agents.md.
        
        Args:
            agents: List of agent info dicts with keys: name, role, status, last_active
        """
        lines = [
            "# Active Agents",
            "",
            f"*Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "| Name | Role | Status | Last Active |",
            "|------|------|--------|-------------|",
        ]
        
        for agent in agents:
            name = agent.get("name", "Unknown")
            role = agent.get("role", "N/A")
            status = agent.get("status", "unknown")
            last_active = agent.get("last_active", "N/A")
            lines.append(f"| {name} | {role} | {status} | {last_active} |")
        
        content = "\n".join(lines)
        return await self.write_file("agents.md", content)
    
    async def write_tasks_md(self, tasks: List[Dict]) -> FileOperationResult:
        """
        Write current task queue to tasks.md.
        
        Args:
            tasks: List of task dicts with keys: id, description, status, priority
        """
        lines = [
            "# Task Queue",
            "",
            f"*Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
        ]
        
        # Group by status
        pending = [t for t in tasks if t.get("status") == "pending"]
        in_progress = [t for t in tasks if t.get("status") == "in_progress"]
        completed = [t for t in tasks if t.get("status") == "completed"]
        
        if in_progress:
            lines.append("## 🔄 In Progress")
            lines.append("")
            for task in in_progress:
                lines.append(f"- **{task.get('description', 'Unknown')}**")
                lines.append(f"  - ID: `{task.get('id', 'N/A')}`")
                lines.append(f"  - Priority: {task.get('priority', 'N/A')}")
            lines.append("")
        
        if pending:
            lines.append("## ⏳ Pending")
            lines.append("")
            for task in pending[:20]:  # Limit to 20
                lines.append(f"- {task.get('description', 'Unknown')}")
            if len(pending) > 20:
                lines.append(f"- *... and {len(pending) - 20} more*")
            lines.append("")
        
        if completed:
            lines.append("## ✅ Recently Completed")
            lines.append("")
            for task in completed[-10:]:  # Last 10
                lines.append(f"- ~~{task.get('description', 'Unknown')}~~")
            lines.append("")
        
        content = "\n".join(lines)
        return await self.write_file("tasks.md", content)
    
    async def write_overview_md(self, state: Dict) -> FileOperationResult:
        """
        Write system overview to overview.md.
        
        Args:
            state: System state dict with relevant information
        """
        lines = [
            "# L.O.V.E. System Overview",
            "",
            f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*",
            "",
            "## System Status",
            "",
            f"- **Mode**: {state.get('mode', 'autopilot')}",
            f"- **Uptime**: {state.get('uptime', 'N/A')}",
            f"- **Active Threads**: {state.get('active_threads', 'N/A')}",
            "",
            "## Current Goal",
            "",
            f"> {state.get('current_goal', 'No goal set')}",
            "",
            "## Recent Activity",
            "",
        ]
        
        recent_activity = state.get("recent_activity", [])
        for activity in recent_activity[-10:]:
            lines.append(f"- {activity}")
        
        if not recent_activity:
            lines.append("- No recent activity logged")
        
        lines.append("")
        lines.append("## Memory Stats")
        lines.append("")
        
        memory_stats = state.get("memory_stats", {})
        lines.append(f"- Episodes: {memory_stats.get('episodes', 'N/A')}")
        lines.append(f"- Wisdom entries: {memory_stats.get('wisdom', 'N/A')}")
        lines.append(f"- Knowledge nodes: {memory_stats.get('knowledge_nodes', 'N/A')}")
        
        content = "\n".join(lines)
        return await self.write_file("overview.md", content)
    
    async def write_scratch_file(
        self, 
        name: str, 
        content: str
    ) -> FileOperationResult:
        """
        Write a scratch file for intermediate work.
        
        Args:
            name: Name of the scratch file (will be prefixed with 'scratch_')
            content: Content to write
        """
        filename = f"scratch/{name}" if "/" not in name else name
        if not any(filename.endswith(ext) for ext in self.ALLOWED_EXTENSIONS):
            filename += ".md"
        return await self.write_file(filename, content)
    
    async def read_scratch_file(self, name: str) -> FileOperationResult:
        """Read a scratch file."""
        filename = f"scratch/{name}" if "/" not in name else name
        if not any(filename.endswith(ext) for ext in self.ALLOWED_EXTENSIONS):
            filename += ".md"
        return await self.read_file(filename)


# ==================== LANGCHAIN TOOL WRAPPERS ====================

def create_filesystem_tools(workspace_path: Path = None) -> List[Any]:
    """
    Create LangChain-compatible tools for agent filesystem access.
    
    Returns list of StructuredTool instances.
    """
    try:
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
    except ImportError:
        log_event("[AgentFS] LangChain not available, tools not created", "WARNING")
        return []
    
    fs = AgentFilesystemTools(workspace_path)
    
    class WriteFileInput(BaseModel):
        filename: str = Field(description="Name/path of file to write")
        content: str = Field(description="Content to write to the file")
    
    class ReadFileInput(BaseModel):
        filename: str = Field(description="Name/path of file to read")
    
    class ListFilesInput(BaseModel):
        subdirectory: str = Field(default="", description="Subdirectory to list (optional)")
    
    async def write_file_impl(filename: str, content: str) -> str:
        result = await fs.write_file(filename, content)
        return json.dumps(result.to_dict())
    
    async def read_file_impl(filename: str) -> str:
        result = await fs.read_file(filename)
        return json.dumps(result.to_dict())
    
    async def list_files_impl(subdirectory: str = "") -> str:
        result = await fs.list_files(subdirectory)
        return json.dumps(result.to_dict())
    
    tools = [
        StructuredTool.from_function(
            coroutine=write_file_impl,
            name="write_workspace_file",
            description="Write content to a file in the agent workspace",
            args_schema=WriteFileInput,
        ),
        StructuredTool.from_function(
            coroutine=read_file_impl,
            name="read_workspace_file",
            description="Read content from a file in the agent workspace",
            args_schema=ReadFileInput,
        ),
        StructuredTool.from_function(
            coroutine=list_files_impl,
            name="list_workspace_files",
            description="List files in the agent workspace",
            args_schema=ListFilesInput,
        ),
    ]
    
    return tools
