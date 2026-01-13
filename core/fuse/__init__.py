"""
FUSE-Based Agent Filesystem Module.

This module provides a virtual filesystem abstraction for agent interactions,
inspired by the "FUSE is All You Need" paradigm. Instead of discrete tools,
agents interact with the system through filesystem operations (ls, cat, mv, etc.)
executed via a unified shell interface.

Key Components:
- VirtualFilesystem: Main filesystem manager routing to domain adapters
- FilesystemAdapter: Base class for domain-specific adapters
- ShellExecutor: Executes shell commands against the virtual filesystem
- FuseAgentHarness: Wraps agents with filesystem-based interaction

Example:
    harness = FuseAgentHarness(
        tool_registry=tool_registry,
        memory_manager=memory_manager,
        knowledge_base=knowledge_base,
    )
    result = await harness.execute("Organize my Bluesky notifications")
"""

from core.fuse.base import (
    VirtualFile,
    VirtualDirectory,
    FilesystemAdapter,
    FilesystemError,
)
from core.fuse.virtual_filesystem import VirtualFilesystem
from core.fuse.shell_executor import ShellExecutor
from core.fuse.fuse_agent_harness import FuseAgentHarness

__all__ = [
    "VirtualFile",
    "VirtualDirectory",
    "FilesystemAdapter",
    "FilesystemError",
    "VirtualFilesystem",
    "ShellExecutor",
    "FuseAgentHarness",
]
