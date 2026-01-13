"""
FUSE Agent Harness.

Wraps an AI agent with access to the virtual filesystem.
Initializes the VirtualFilesystem and all its adapters.
Provides the `bash` tool for agents to interact with the environment.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable

from core.fuse.virtual_filesystem import VirtualFilesystem
from core.fuse.shell_executor import ShellExecutor
from core.fuse.adapters.tool_filesystem import ToolFilesystem
from core.fuse.adapters.memory_filesystem import MemoryFilesystem
from core.fuse.adapters.knowledge_filesystem import KnowledgeFilesystem
from core.fuse.adapters.conversation_filesystem import ConversationFilesystem
from core.fuse.adapters.social_filesystem import SocialFilesystem

logger = logging.getLogger(__name__)


class FuseAgentHarness:
    """
    Harness that equips an agent with a Virtual Filesystem.
    
    This class manages the initialization of the VFS, its adapters,
    and the ShellExecutor. It exposes a `bash` function that can be
    registered as a tool for the agent.
    """
    
    def __init__(
        self,
        tool_registry=None,
        memory_manager=None,
        knowledge_base=None,
        social_manager=None,
        love_state: Dict = None,
    ):
        """
        Initialize the harness.
        
        Args:
            tool_registry: The L.O.V.E. tool registry
            memory_manager: The memory manager
            knowledge_base: The knowledge base
            social_manager: The social media manager (e.g. Bluesky)
            love_state: Global state dictionary
        """
        self.vfs = VirtualFilesystem()
        self.shell = ShellExecutor(self.vfs)
        
        # Mount adapters if components are available
        if tool_registry:
            self.vfs.mount("/tools", ToolFilesystem(tool_registry))
        
        if memory_manager:
            self.vfs.mount("/memories", MemoryFilesystem(memory_manager))
        
        if knowledge_base:
            self.vfs.mount("/knowledge", KnowledgeFilesystem(knowledge_base))
            
        if social_manager:
            self.vfs.mount("/social", SocialFilesystem(social_manager))
            
        # Conversation history needs either state or manager
        if love_state or (hasattr(memory_manager, 'conversation_history')):
            # Sometimes conversation history is in memory manager, sometimes separate
            # For now, pass what we have
            self.vfs.mount("/conversations", ConversationFilesystem(
                conversation_manager=None,  # Pass if available separately
                love_state=love_state
            ))
            
        logger.info("FuseAgentHarness initialized with VFS mounts: " + 
                    ", ".join(self.vfs.get_mount_points()))
    
    def execute_shell(self, command: str) -> str:
        """
        Execute a shell command.
        
        This is the main entry point for the agent's interaction.
        """
        try:
            exit_code, stdout, stderr = self.shell.execute(command)
            
            output = ""
            if stdout:
                output += stdout
            if stderr:
                if output:
                    output += "\nSTDERR:\n"
                output += stderr
            
            if exit_code != 0:
                return f"Command failed with exit code {exit_code}:\n{output}"
                
            return output if output else "Command executed successfully (no output)"
            
        except Exception as e:
            logger.exception(f"Harness shell execution error: {e}")
            return f"Error executing shell command: {e}"

    def get_tool_definition(self) -> Dict[str, Any]:
        """
        Get the JSON schema definition for the `bash` tool.
        """
        return {
            "name": "bash",
            "description": "Execute shell commands to interact with the system. You can list files (ls), read content (cat), search (grep/find), and invoke tools or actions by writing to files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute (e.g., 'ls /tools', 'cat /memories/current_goal.txt')"
                    }
                },
                "required": ["command"]
            }
        }
    
    def get_system_prompt_addition(self) -> str:
        """
        Get the prompt text to teach the agent about the filesystem.
        """
        return """
## System Interface: Virtual Filesystem (FUSE)

You have access to a virtual filesystem that exposes tools, memories, knowledge, and more.
Interact with this system using the `bash` tool to run commands like `ls`, `cat`, `grep`, `mv`, `echo`.

### Available Directories:
- `/tools/`: Registered tools. Read `description.txt` to understand them. Write JSON args to `invoke` to execute.
- `/memories/`: Your memory. `/memories/working/` contains your current goal and plan.
- `/knowledge/`: The knowledge graph and entities.
- `/social/`: Social media interfaces (e.g., `/social/bluesky/`).
- `/conversations/`: Past interaction history.
- `/scratch/`: A place for temporary files and notes.

### Examples:
- List tools: `bash("ls /tools")`
- Read tool help: `bash("cat /tools/generate_image/description.txt")`
- Invoke tool: `bash("echo '{\"prompt\": \"a cat\"}' > /tools/generate_image/invoke")`
- Read failure logs: `bash("cat /tools/generate_image/result")`
- Update plan: `bash("echo '- Step 1: Done' > /memories/working/plan.txt")`
- Search knowledge: `bash("echo 'DeepAgent' > /knowledge/search")` then `bash("cat /knowledge/search")` (or result file)
"""
