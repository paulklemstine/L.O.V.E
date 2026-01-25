"""
Filesystem Adapters for the FUSE Agent System.

Each adapter exposes a specific domain as a navigable filesystem:
- ToolFilesystem: /tools/ - Available tools
- MemoryFilesystem: /memories/ - Episodic and working memory
- KnowledgeFilesystem: /knowledge/ - Knowledge base entities
- ConversationFilesystem: /conversations/ - Past conversation summaries
- SocialFilesystem: /social/ - Social media interactions (Bluesky)
"""

from core.fuse.adapters.tool_filesystem import ToolFilesystem
from core.fuse.adapters.memory_filesystem import MemoryFilesystem
from core.fuse.adapters.knowledge_filesystem import KnowledgeFilesystem
from core.fuse.adapters.conversation_filesystem import ConversationFilesystem
from core.fuse.adapters.social_filesystem import SocialFilesystem

__all__ = [
    "ToolFilesystem",
    "MemoryFilesystem",
    "KnowledgeFilesystem",
    "ConversationFilesystem",
    "SocialFilesystem",
]
