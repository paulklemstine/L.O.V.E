"""
Virtual Filesystem Manager.

Routes filesystem operations to appropriate adapters based on path.
Provides a unified interface for the shell executor.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from core.fuse.base import (
    FilesystemAdapter,
    FileAttributes,
    FileType,
    FilesystemError,
    FileNotFoundError,
    NotADirectoryError,
    IsADirectoryError,
    ScratchFilesystem,
)

logger = logging.getLogger(__name__)


class VirtualFilesystem:
    """
    Main virtual filesystem manager.
    
    Routes filesystem operations to the appropriate adapter based on path.
    Manages multiple mount points for different domain adapters.
    
    Example:
        vfs = VirtualFilesystem()
        vfs.mount("/tools", ToolFilesystem(tool_registry))
        vfs.mount("/memories", MemoryFilesystem(memory_manager))
        
        # Now agents can use:
        vfs.readdir("/tools")  # Lists available tools
        vfs.read("/tools/execute/description.txt")  # Read tool description
        vfs.write("/tools/execute/invoke", '{"command": "ls -la"}')  # Invoke tool
    """
    
    def __init__(self, scratch_path: Optional[str] = None):
        self._mounts: Dict[str, FilesystemAdapter] = {}
        self._mount_order: List[str] = []  # For proper path resolution
        
        # Always include scratch filesystem for working files
        self._scratch = ScratchFilesystem("/scratch", root_path=scratch_path)
        self.mount("/scratch", self._scratch)
    
    def mount(self, mount_point: str, adapter: FilesystemAdapter) -> None:
        """
        Mount a filesystem adapter at the given path.
        
        Args:
            mount_point: Path where adapter will be mounted (e.g., "/tools")
            adapter: FilesystemAdapter instance
        """
        mount_point = "/" + mount_point.strip("/")
        self._mounts[mount_point] = adapter
        
        # Keep mounts sorted by depth (deepest first) for proper resolution
        self._mount_order = sorted(
            self._mounts.keys(),
            key=lambda x: x.count("/"),
            reverse=True
        )
        
        logger.info(f"Mounted filesystem adapter at {mount_point}")
    
    def unmount(self, mount_point: str) -> bool:
        """
        Unmount a filesystem adapter.
        
        Args:
            mount_point: Path to unmount
            
        Returns:
            True if adapter was unmounted
        """
        mount_point = "/" + mount_point.strip("/")
        if mount_point in self._mounts:
            del self._mounts[mount_point]
            self._mount_order = sorted(
                self._mounts.keys(),
                key=lambda x: x.count("/"),
                reverse=True
            )
            return True
        return False
    
    def _resolve_path(self, path: str) -> Tuple[FilesystemAdapter, str]:
        """
        Resolve a path to its adapter and relative path.
        
        Args:
            path: Absolute path in the virtual filesystem
            
        Returns:
            Tuple of (adapter, relative_path)
            
        Raises:
            FileNotFoundError: If no adapter handles the path
        """
        path = os.path.normpath("/" + path.lstrip("/")).replace("\\", "/")
        
        for mount_point in self._mount_order:
            if path == mount_point or path.startswith(mount_point + "/"):
                relative = path[len(mount_point):] or "/"
                return self._mounts[mount_point], relative
        
        raise FileNotFoundError(f"No filesystem mounted at {path}")
    
    def get_mount_points(self) -> List[str]:
        """Return list of all mount points."""
        return list(self._mounts.keys())
    
    def readdir(self, path: str) -> List[str]:
        """
        List directory contents.
        
        For the root path ("/"), lists all mount points.
        Otherwise delegates to the appropriate adapter.
        """
        path = os.path.normpath("/" + path.lstrip("/")).replace("\\", "/")
        
        # Root directory lists mount points
        if path == "/":
            return [mp.lstrip("/") for mp in self._mounts.keys()]
        
        adapter, relative_path = self._resolve_path(path)
        return adapter.readdir(relative_path)
    
    def read(self, path: str) -> str:
        """Read file contents."""
        adapter, relative_path = self._resolve_path(path)
        return adapter.read(relative_path)
    
    def write(self, path: str, content: str, append: bool = False) -> str:
        """
        Write content to a file.
        
        Returns:
            Result message (especially important for executable files)
        """
        adapter, relative_path = self._resolve_path(path)
        adapter.write(relative_path, content, append)
        
        # Check for action result (e.g., tool invocation)
        result = adapter.get_last_write_result()
        return result or "Write successful"
    
    def getattr(self, path: str) -> FileAttributes:
        """Get file/directory attributes."""
        path = os.path.normpath("/" + path.lstrip("/")).replace("\\", "/")
        
        # Root directory
        if path == "/":
            return FileAttributes(
                mode=0o755,
                file_type=FileType.DIRECTORY
            )
        
        adapter, relative_path = self._resolve_path(path)
        return adapter.getattr(relative_path)
    
    def exists(self, path: str) -> bool:
        """Check if path exists."""
        try:
            self.getattr(path)
            return True
        except (FileNotFoundError, FilesystemError):
            return False
    
    def isdir(self, path: str) -> bool:
        """Check if path is a directory."""
        try:
            attrs = self.getattr(path)
            return attrs.is_dir()
        except FileNotFoundError:
            return False
    
    def isfile(self, path: str) -> bool:
        """Check if path is a file."""
        try:
            attrs = self.getattr(path)
            return attrs.is_file()
        except FileNotFoundError:
            return False
    
    def mkdir(self, path: str) -> bool:
        """Create a directory."""
        adapter, relative_path = self._resolve_path(path)
        return adapter.mkdir(relative_path)
    
    def remove(self, path: str) -> bool:
        """Remove a file."""
        adapter, relative_path = self._resolve_path(path)
        return adapter.remove(relative_path)
    
    def rmdir(self, path: str) -> bool:
        """Remove a directory."""
        adapter, relative_path = self._resolve_path(path)
        return adapter.rmdir(relative_path)
    
    def rename(self, old_path: str, new_path: str) -> bool:
        """Rename/move a file or directory."""
        old_adapter, old_relative = self._resolve_path(old_path)
        new_adapter, new_relative = self._resolve_path(new_path)
        
        if old_adapter is not new_adapter:
            raise FilesystemError("Cannot move between different filesystems")
        
        return old_adapter.rename(old_relative, new_relative)
    
    def tree(self, path: str = "/", max_depth: int = 3, prefix: str = "") -> str:
        """
        Generate a tree representation of the filesystem.
        
        Useful for giving agents an overview of available resources.
        
        Args:
            path: Starting path
            max_depth: Maximum depth to recurse
            prefix: Prefix for formatting (internal use)
            
        Returns:
            Tree string representation
        """
        if max_depth < 0:
            return ""
        
        lines = []
        
        try:
            entries = self.readdir(path)
        except (FileNotFoundError, NotADirectoryError):
            return ""
        
        for i, entry in enumerate(sorted(entries)):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            
            full_path = os.path.join(path, entry).replace("\\", "/")
            
            if self.isdir(full_path):
                lines.append(f"{prefix}{connector}{entry}/")
                extension = "    " if is_last else "│   "
                subtree = self.tree(full_path, max_depth - 1, prefix + extension)
                if subtree:
                    lines.append(subtree)
            else:
                lines.append(f"{prefix}{connector}{entry}")
        
        return "\n".join(lines)
    
    def get_working_directory(self) -> str:
        """Get the current working directory (always /scratch for agents)."""
        return "/scratch"
    
    def write_scratch(self, filename: str, content: str) -> bool:
        """Convenience method to write to scratch space."""
        path = f"/scratch/{filename}"
        self.write(path, content)
        return True
    
    def read_scratch(self, filename: str) -> str:
        """Convenience method to read from scratch space."""
        return self.read(f"/scratch/{filename}")
