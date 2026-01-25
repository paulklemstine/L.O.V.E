"""
Base classes for the Virtual Filesystem abstraction.

This module provides the fundamental building blocks for implementing
a virtual filesystem that agents can interact with via shell commands.
"""

import os
import stat
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class FileType(Enum):
    """Types of virtual filesystem entries."""
    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    EXECUTABLE = "executable"  # Special files that execute actions when written to


class FilesystemError(Exception):
    """Base exception for filesystem operations."""
    pass


class FileNotFoundError(FilesystemError):
    """Raised when a file or directory is not found."""
    pass


class PermissionError(FilesystemError):
    """Raised when an operation is not permitted."""
    pass


class NotADirectoryError(FilesystemError):
    """Raised when directory operation attempted on non-directory."""
    pass


class IsADirectoryError(FilesystemError):
    """Raised when file operation attempted on directory."""
    pass


class FileExistsError(FilesystemError):
    """Raised when trying to create a file that already exists."""
    pass


@dataclass
class FileAttributes:
    """POSIX-like file attributes for virtual files."""
    mode: int = 0o644  # Default: readable file
    size: int = 0
    atime: float = field(default_factory=time.time)  # Access time
    mtime: float = field(default_factory=time.time)  # Modification time
    ctime: float = field(default_factory=time.time)  # Creation time
    file_type: FileType = FileType.FILE
    
    def is_dir(self) -> bool:
        return self.file_type == FileType.DIRECTORY
    
    def is_file(self) -> bool:
        return self.file_type in (FileType.FILE, FileType.EXECUTABLE)
    
    def is_executable(self) -> bool:
        return self.file_type == FileType.EXECUTABLE
    
    def to_stat_dict(self) -> Dict[str, Any]:
        """Convert to stat-like dictionary for shell commands."""
        mode = self.mode
        if self.file_type == FileType.DIRECTORY:
            mode = mode | stat.S_IFDIR
        else:
            mode = mode | stat.S_IFREG
        
        return {
            "st_mode": mode,
            "st_size": self.size,
            "st_atime": self.atime,
            "st_mtime": self.mtime,
            "st_ctime": self.ctime,
        }


@dataclass
class VirtualFile:
    """
    Represents a virtual file with content and metadata.
    
    Attributes:
        name: Filename (basename)
        content: File contents (string or bytes)
        attributes: File metadata (permissions, timestamps, etc.)
        metadata: Additional domain-specific metadata
    """
    name: str
    content: Union[str, bytes] = ""
    attributes: FileAttributes = field(default_factory=FileAttributes)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.attributes.size == 0 and self.content:
            content_bytes = self.content.encode() if isinstance(self.content, str) else self.content
            self.attributes.size = len(content_bytes)


@dataclass
class VirtualDirectory:
    """
    Represents a virtual directory.
    
    Directories are lazy-loaded through the adapter's readdir method.
    They don't store children directly; children are resolved on access.
    
    Attributes:
        name: Directory name (basename)
        attributes: Directory metadata
        metadata: Additional domain-specific metadata
    """
    name: str
    attributes: FileAttributes = field(default_factory=lambda: FileAttributes(
        mode=0o755,
        file_type=FileType.DIRECTORY
    ))
    metadata: Dict[str, Any] = field(default_factory=dict)


class FilesystemAdapter(ABC):
    """
    Base class for domain-specific filesystem adapters.
    
    Each adapter exposes a specific domain (tools, memories, etc.) as a 
    navigable filesystem. Adapters handle the translation between filesystem
    operations and domain-specific actions.
    
    Example structure for ToolFilesystem:
        /tools/
        ├── code_modifier/
        │   ├── schema.json
        │   ├── description.txt
        │   └── invoke           # Write arguments here to invoke
        └── execute/
            └── ...
    
    Subclasses must implement the core filesystem operations:
    - readdir: List directory contents
    - read: Read file contents
    - write: Write to a file (may trigger actions)
    - getattr: Get file/directory attributes
    """
    
    def __init__(self, mount_point: str):
        """
        Initialize the adapter with its mount point.
        
        Args:
            mount_point: The path where this adapter is mounted (e.g., "/tools")
        """
        self.mount_point = mount_point.rstrip("/")
        self._write_callbacks: Dict[str, callable] = {}
    
    def _normalize_path(self, path: str) -> str:
        """Normalize path to be relative to mount point."""
        if path.startswith(self.mount_point):
            path = path[len(self.mount_point):]
        path = "/" + path.lstrip("/")
        return os.path.normpath(path).replace("\\", "/")
    
    @abstractmethod
    def readdir(self, path: str) -> List[str]:
        """
        List the contents of a directory.
        
        Args:
            path: Path relative to mount point
            
        Returns:
            List of entry names (filenames and subdirectory names)
            
        Raises:
            FileNotFoundError: If path doesn't exist
            NotADirectoryError: If path is not a directory
        """
        pass
    
    @abstractmethod
    def read(self, path: str) -> str:
        """
        Read the contents of a file.
        
        Args:
            path: Path relative to mount point
            
        Returns:
            File contents as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IsADirectoryError: If path is a directory
        """
        pass
    
    @abstractmethod
    def write(self, path: str, content: str, append: bool = False) -> bool:
        """
        Write content to a file.
        
        For executable files (like /tools/*/invoke), this may trigger
        domain-specific actions and return the result.
        
        Args:
            path: Path relative to mount point
            content: Content to write
            append: If True, append to existing content
            
        Returns:
            True if write was successful
            
        Raises:
            FileNotFoundError: If file doesn't exist and can't be created
            PermissionError: If write is not allowed
            IsADirectoryError: If path is a directory
        """
        pass
    
    @abstractmethod
    def getattr(self, path: str) -> FileAttributes:
        """
        Get attributes of a file or directory.
        
        Args:
            path: Path relative to mount point
            
        Returns:
            FileAttributes object with metadata
            
        Raises:
            FileNotFoundError: If path doesn't exist
        """
        pass
    
    def exists(self, path: str) -> bool:
        """Check if a path exists."""
        try:
            self.getattr(path)
            return True
        except FileNotFoundError:
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
        """
        Create a directory.
        
        Override in subclasses that support directory creation.
        
        Args:
            path: Path for new directory
            
        Returns:
            True if directory was created
            
        Raises:
            FileExistsError: If path already exists
            PermissionError: If creation not allowed
        """
        raise PermissionError(f"Directory creation not supported at {path}")
    
    def remove(self, path: str) -> bool:
        """
        Remove a file.
        
        Override in subclasses that support file deletion.
        
        Args:
            path: Path to remove
            
        Returns:
            True if file was removed
            
        Raises:
            FileNotFoundError: If path doesn't exist
            PermissionError: If removal not allowed
            IsADirectoryError: If path is a directory
        """
        raise PermissionError(f"File removal not supported at {path}")
    
    def rmdir(self, path: str) -> bool:
        """
        Remove a directory.
        
        Override in subclasses that support directory removal.
        
        Args:
            path: Directory path to remove
            
        Returns:
            True if directory was removed
            
        Raises:
            FileNotFoundError: If path doesn't exist
            PermissionError: If removal not allowed
            NotADirectoryError: If path is not a directory
        """
        raise PermissionError(f"Directory removal not supported at {path}")
    
    def rename(self, old_path: str, new_path: str) -> bool:
        """
        Rename/move a file or directory.
        
        Override in subclasses that support renaming.
        
        Args:
            old_path: Current path
            new_path: New path
            
        Returns:
            True if rename was successful
            
        Raises:
            FileNotFoundError: If old_path doesn't exist
            PermissionError: If rename not allowed
        """
        raise PermissionError(f"Rename not supported: {old_path} -> {new_path}")
    
    def get_last_write_result(self) -> Optional[str]:
        """
        Get the result of the last write operation.
        
        For executable files that trigger actions (like tool invocation),
        this returns the action's output.
        
        Returns:
            Result string or None if no special result
        """
        result = getattr(self, "_last_write_result", None)
        self._last_write_result = None
        return result
    
    def _set_write_result(self, result: str):
        """Store the result of a write operation."""
        self._last_write_result = result


class ScratchFilesystem(FilesystemAdapter):
    """
    A filesystem for scratch/working files, optionally backed by disk.
    
    If 'root_path' is provided, files are stored on disk (enabling Docker mounts).
    If not, it behaves as an in-memory filesystem.
    """
    
    def __init__(self, mount_point: str = "/scratch", root_path: Optional[str] = None):
        super().__init__(mount_point)
        self.root_path = root_path
        if self.root_path:
            os.makedirs(self.root_path, exist_ok=True)
            self._files = {} # Unused in disk mode
            self._dirs = {}  # Unused in disk mode
        else:
            self._files: Dict[str, VirtualFile] = {}
            self._dirs: Dict[str, VirtualDirectory] = {
                "/": VirtualDirectory(name="")
            }
    
    def _get_real_path(self, virtual_path: str) -> str:
        """Convert virtual path to real disk path."""
        if not self.root_path:
             raise FilesystemError("Not a disk-backed filesystem")
        # Remove leading slash/mount point relative logic if needed
        # virtual_path is relative to mount point (e.g. "plan.txt" or "subdir/file.txt")
        # FilesystemAdapter generic logic usually passes 'path' relative to mount point?
        # Wait, readdir/read receive relative paths in the adapter?
        # Let's check: VirtualFilesystem routes with: adapter.readdir(relative_path)
        # Yes.
        
        # Security check: prevent traversal
        safe_path = os.path.normpath(os.path.join(self.root_path, virtual_path.lstrip("/")))
        if not safe_path.startswith(os.path.abspath(self.root_path)):
             raise PermissionError(f"Path traversal denied: {virtual_path}")
        return safe_path

    def readdir(self, path: str) -> List[str]:
        if self.root_path:
            real_path = self._get_real_path(path)
            if not os.path.exists(real_path):
                raise FileNotFoundError(f"Directory not found: {path}")
            if not os.path.isdir(real_path):
                raise NotADirectoryError(f"Not a directory: {path}")
            return sorted(os.listdir(real_path))
        else:
            # Memory implementation
            path = self._normalize_path(path)
            if path not in self._dirs:
                raise FileNotFoundError(f"Directory not found: {path}")
            
            entries = set()
            prefix = path if path == "/" else path + "/"
            
            for file_path in self._files:
                if file_path.startswith(prefix):
                    remainder = file_path[len(prefix):]
                    if "/" not in remainder:
                        entries.add(remainder)
                    else:
                        entries.add(remainder.split("/")[0])
            
            for dir_path in self._dirs:
                if dir_path.startswith(prefix) and dir_path != path:
                    remainder = dir_path[len(prefix):]
                    if "/" not in remainder:
                        entries.add(remainder)
                    else:
                        entries.add(remainder.split("/")[0])
            
            return sorted(entries)
    
    def read(self, path: str) -> str:
        if self.root_path:
            real_path = self._get_real_path(path)
            if os.path.isdir(real_path):
                raise IsADirectoryError(f"Is a directory: {path}")
            if not os.path.exists(real_path):
                raise FileNotFoundError(f"File not found: {path}")
            try:
                with open(real_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                raise FilesystemError(f"Read error: {e}")
        else:
            path = self._normalize_path(path)
            if path in self._dirs:
                raise IsADirectoryError(f"Is a directory: {path}")
            if path not in self._files:
                raise FileNotFoundError(f"File not found: {path}")
            
            content = self._files[path].content
            return content if isinstance(content, str) else content.decode()
    
    def write(self, path: str, content: str, append: bool = False) -> bool:
        if self.root_path:
            real_path = self._get_real_path(path)
            if os.path.isdir(real_path):
                 raise IsADirectoryError(f"Is a directory: {path}")
            
            mode = 'a' if append else 'w'
            try:
                os.makedirs(os.path.dirname(real_path), exist_ok=True)
                with open(real_path, mode, encoding='utf-8') as f:
                    f.write(content)
                return True
            except Exception as e:
                raise FilesystemError(f"Write error: {e}")
        else:
            path = self._normalize_path(path)
            if path in self._dirs:
                raise IsADirectoryError(f"Is a directory: {path}")
            
            # Ensure parent directory exists
            parent = os.path.dirname(path)
            if parent and parent != "/" and parent not in self._dirs:
                self.mkdir(parent)
            
            if path in self._files and append:
                existing = self._files[path].content
                existing = existing if isinstance(existing, str) else existing.decode()
                content = existing + content
            
            self._files[path] = VirtualFile(
                name=os.path.basename(path),
                content=content
            )
            return True
    
    def getattr(self, path: str) -> FileAttributes:
        if self.root_path:
            real_path = self._get_real_path(path)
            if not os.path.exists(real_path):
                 if path == "" or path == "/": # Root always exists
                     return FileAttributes(mode=0o755, file_type=FileType.DIRECTORY)
                 raise FileNotFoundError(f"Path not found: {path}")
            
            stat = os.stat(real_path)
            ftype = FileType.DIRECTORY if os.path.isdir(real_path) else FileType.FILE
            mode = 0o755 if ftype == FileType.DIRECTORY else 0o644
            
            return FileAttributes(
                mode=mode,
                file_type=ftype,
                size=stat.st_size,
                mtime=stat.st_mtime
            )
        else:
            path = self._normalize_path(path)
            
            if path in self._dirs:
                return self._dirs[path].attributes
            if path in self._files:
                return self._files[path].attributes
            
            raise FileNotFoundError(f"Path not found: {path}")
    
    def mkdir(self, path: str) -> bool:
        if self.root_path:
            real_path = self._get_real_path(path)
            try:
                os.makedirs(real_path, exist_ok=True)
                return True
            except Exception as e:
                raise FilesystemError(f"Mkdir error: {e}")
        else:
            path = self._normalize_path(path)
            if path in self._dirs:
                raise FileExistsError(f"Directory exists: {path}")
            if path in self._files:
                raise FileExistsError(f"File exists at path: {path}")
            
            # Create parent directories recursively
            parent = os.path.dirname(path)
            if parent and parent != "/" and parent not in self._dirs:
                self.mkdir(parent)
            
            self._dirs[path] = VirtualDirectory(name=os.path.basename(path))
            return True
    
    def remove(self, path: str) -> bool:
        if self.root_path:
             real_path = self._get_real_path(path)
             if os.path.isdir(real_path):
                 raise IsADirectoryError(f"Is a directory: {path}")
             try:
                 os.remove(real_path)
                 return True
             except FileNotFoundError:
                 raise FileNotFoundError(f"File not found: {path}")
             except Exception as e:
                 raise FilesystemError(f"Remove error: {e}")
        else:
            path = self._normalize_path(path)
            if path in self._dirs:
                raise IsADirectoryError(f"Is a directory: {path}")
            if path not in self._files:
                raise FileNotFoundError(f"File not found: {path}")
            
            del self._files[path]
            return True
    
    def rmdir(self, path: str) -> bool:
        if self.root_path:
            real_path = self._get_real_path(path)
            if not os.path.isdir(real_path):
                raise NotADirectoryError(f"Not a directory: {path}")
            try:
                os.rmdir(real_path)
                return True
            except OSError:
                 raise PermissionError(f"Directory not empty: {path}")
        else:
            path = self._normalize_path(path)
            if path not in self._dirs:
                raise FileNotFoundError(f"Directory not found: {path}")
            if path in self._files:
                raise NotADirectoryError(f"Not a directory: {path}")
            
            # Check if empty
            if self.readdir(path):
                raise PermissionError(f"Directory not empty: {path}")
            
            del self._dirs[path]
            return True
