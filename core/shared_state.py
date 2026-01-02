from typing import Any, Dict, Optional
import queue

# Global State Variables
# These are initialized by love.py and accessed by other modules

# Core Logic Engines
memory_manager: Any = None
knowledge_base: Any = None
deep_agent_engine: Any = None

# MCP (Model Context Protocol) Server Manager
mcp_manager: Any = None

# Application State
love_state: Dict[str, Any] = {}

# UI / Display
ui_panel_queue: Optional[queue.Queue] = None

# Task Management
love_task_manager: Any = None

# Tool Registry (for REPL agent and other components)
tool_registry: Any = None

# Completed Tasks Registry (Epic 1: Hierarchical Task Decomposition)
completed_tasks: Dict[str, Dict[str, Any]] = {}
