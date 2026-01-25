"""
Conversation Filesystem Adapter.

Exposes past conversation summaries as files for context.
Enables agents to reference and learn from previous interactions.

Filesystem structure:
    /conversations/
    ├── recent/
    │   ├── pollinations_logos.txt
    │   ├── surfer_persona.txt
    │   └── ...
    ├── by_date/
    │   ├── 2026-01-13/
    │   │   └── session_001.txt
    │   └── ...
    └── stats.txt
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from core.fuse.base import (
    FilesystemAdapter,
    FileAttributes,
    FileType,
    FileNotFoundError,
    NotADirectoryError,
    IsADirectoryError,
    PermissionError,
)

logger = logging.getLogger(__name__)


class ConversationFilesystem(FilesystemAdapter):
    """
    Exposes conversation history as a virtual filesystem.
    
    Provides access to:
    - recent/: Recent conversation summaries
    - by_date/: Conversations organized by date
    """
    
    def __init__(self, conversation_manager=None, love_state: Dict = None, mount_point: str = "/conversations"):
        """
        Initialize with conversation data.
        
        Args:
            conversation_manager: Optional conversation manager
            love_state: The L.O.V.E. state dict (may contain conversation history)
            mount_point: Where to mount this filesystem
        """
        super().__init__(mount_point)
        self.conversation_manager = conversation_manager
        self.love_state = love_state or {}
    
    def _get_conversations(self) -> List[Dict]:
        """Get all available conversations."""
        conversations = []
        
        try:
            # Try conversation manager first
            if self.conversation_manager:
                if hasattr(self.conversation_manager, 'get_all'):
                    return self.conversation_manager.get_all()
                elif hasattr(self.conversation_manager, 'list_conversations'):
                    return self.conversation_manager.list_conversations()
            
            # Fall back to love_state
            if "conversations" in self.love_state:
                return self.love_state["conversations"]
            
            if "conversation_history" in self.love_state:
                return self.love_state["conversation_history"]
            
            # Try to load from file
            import os
            if os.path.exists("conversation_history.json"):
                with open("conversation_history.json") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
                    if "conversations" in data:
                        return data["conversations"]
        except Exception as e:
            logger.error(f"Error loading conversations: {e}")
        
        return conversations
    
    def _conversation_to_filename(self, conversation: Dict) -> str:
        """Convert a conversation to a filename."""
        title = conversation.get("title") or conversation.get("summary") or ""
        if not title:
            conv_id = conversation.get("id") or conversation.get("conversation_id") or "unknown"
            title = str(conv_id)[:20]
        
        # Sanitize for filename
        safe_title = "".join(c if c.isalnum() or c in "_- " else "_" for c in title)
        safe_title = safe_title.replace(" ", "_")[:40]
        return f"{safe_title}.txt"
    
    def _conversation_to_content(self, conversation: Dict) -> str:
        """Convert a conversation to readable content."""
        lines = []
        
        # Title/Summary
        title = conversation.get("title") or conversation.get("summary") or "Untitled Conversation"
        lines.append(f"# {title}")
        lines.append("")
        
        # Timestamp
        created = conversation.get("created") or conversation.get("timestamp") or conversation.get("created_at")
        if created:
            lines.append(f"**Created:** {created}")
        
        modified = conversation.get("modified") or conversation.get("last_modified")
        if modified:
            lines.append(f"**Last Modified:** {modified}")
        
        lines.append("")
        
        # Messages/Content
        messages = conversation.get("messages") or conversation.get("content") or []
        
        if isinstance(messages, list):
            lines.append("## Messages")
            lines.append("")
            for msg in messages[-20:]:  # Last 20 messages
                if isinstance(msg, dict):
                    role = msg.get("role") or msg.get("type") or "unknown"
                    content = msg.get("content") or msg.get("text") or ""
                    lines.append(f"### {role.capitalize()}")
                    lines.append(content[:500])  # Truncate long messages
                    lines.append("")
                else:
                    lines.append(str(msg))
                    lines.append("")
        elif isinstance(messages, str):
            lines.append("## Content")
            lines.append(messages)
        
        # Summary if available
        summary = conversation.get("summary")
        if summary:
            lines.append("## Summary")
            lines.append(summary)
        
        return "\n".join(lines)
    
    def _get_conversations_by_date(self) -> Dict[str, List[Dict]]:
        """Group conversations by date."""
        by_date: Dict[str, List[Dict]] = {}
        
        for conv in self._get_conversations():
            timestamp = conv.get("created") or conv.get("timestamp") or conv.get("created_at")
            
            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError:
                    dt = datetime.now()
            elif isinstance(timestamp, (int, float)):
                dt = datetime.fromtimestamp(timestamp)
            else:
                dt = datetime.now()
            
            date_str = dt.strftime("%Y-%m-%d")
            if date_str not in by_date:
                by_date[date_str] = []
            by_date[date_str].append(conv)
        
        return by_date
    
    def readdir(self, path: str) -> List[str]:
        """List directory contents."""
        path = self._normalize_path(path)
        
        if path == "/" or path == "":
            return ["recent", "by_date", "stats.txt"]
        
        parts = path.strip("/").split("/")
        
        if parts[0] == "recent":
            if len(parts) == 1:
                conversations = self._get_conversations()
                # Return the 10 most recent
                recent = sorted(
                    conversations,
                    key=lambda c: c.get("modified") or c.get("created") or "",
                    reverse=True
                )[:10]
                return [self._conversation_to_filename(c) for c in recent]
        
        elif parts[0] == "by_date":
            if len(parts) == 1:
                by_date = self._get_conversations_by_date()
                return sorted(by_date.keys(), reverse=True)
            elif len(parts) == 2:
                date = parts[1]
                by_date = self._get_conversations_by_date()
                if date not in by_date:
                    raise FileNotFoundError(f"No conversations for date: {date}")
                return [self._conversation_to_filename(c) for c in by_date[date]]
        
        raise FileNotFoundError(f"Directory not found: {path}")
    
    def read(self, path: str) -> str:
        """Read file contents."""
        path = self._normalize_path(path)
        parts = path.strip("/").split("/")
        
        if len(parts) == 1:
            if parts[0] == "stats.txt":
                return self._get_stats()
            raise IsADirectoryError(f"Is a directory: {path}")
        
        if parts[0] == "recent":
            if len(parts) == 2:
                filename = parts[1]
                conversations = self._get_conversations()
                recent = sorted(
                    conversations,
                    key=lambda c: c.get("modified") or c.get("created") or "",
                    reverse=True
                )[:10]
                
                for conv in recent:
                    if self._conversation_to_filename(conv) == filename:
                        return self._conversation_to_content(conv)
                raise FileNotFoundError(f"Conversation not found: {filename}")
        
        elif parts[0] == "by_date":
            if len(parts) == 2:
                raise IsADirectoryError(f"Is a directory: {path}")
            elif len(parts) == 3:
                date = parts[1]
                filename = parts[2]
                by_date = self._get_conversations_by_date()
                
                if date not in by_date:
                    raise FileNotFoundError(f"Date not found: {date}")
                
                for conv in by_date[date]:
                    if self._conversation_to_filename(conv) == filename:
                        return self._conversation_to_content(conv)
                raise FileNotFoundError(f"Conversation not found: {filename}")
        
        raise FileNotFoundError(f"File not found: {path}")
    
    def write(self, path: str, content: str, append: bool = False) -> bool:
        """Write to a file (not supported for conversations)."""
        raise PermissionError("Conversations are read-only")
    
    def _get_stats(self) -> str:
        """Get conversation statistics."""
        lines = ["# Conversation Statistics", ""]
        
        conversations = self._get_conversations()
        lines.append(f"Total conversations: {len(conversations)}")
        
        by_date = self._get_conversations_by_date()
        lines.append(f"Days with conversations: {len(by_date)}")
        
        if by_date:
            recent_date = max(by_date.keys())
            lines.append(f"Most recent: {recent_date}")
            lines.append(f"Conversations that day: {len(by_date[recent_date])}")
        
        # Message count
        total_messages = 0
        for conv in conversations:
            messages = conv.get("messages") or conv.get("content") or []
            if isinstance(messages, list):
                total_messages += len(messages)
        
        lines.append("")
        lines.append(f"Total messages: {total_messages}")
        
        return "\n".join(lines)
    
    def getattr(self, path: str) -> FileAttributes:
        """Get file/directory attributes."""
        path = self._normalize_path(path)
        
        if path == "/" or path == "":
            return FileAttributes(mode=0o755, file_type=FileType.DIRECTORY)
        
        parts = path.strip("/").split("/")
        
        if parts[0] in ["recent", "by_date"]:
            if len(parts) == 1:
                return FileAttributes(mode=0o755, file_type=FileType.DIRECTORY)
            
            if parts[0] == "by_date" and len(parts) == 2:
                by_date = self._get_conversations_by_date()
                if parts[1] in by_date:
                    return FileAttributes(mode=0o755, file_type=FileType.DIRECTORY)
                raise FileNotFoundError(f"Date not found: {parts[1]}")
        
        try:
            content = self.read(path)
            return FileAttributes(
                mode=0o444,  # Read-only
                file_type=FileType.FILE,
                size=len(content.encode())
            )
        except (FileNotFoundError, IsADirectoryError):
            raise FileNotFoundError(f"Path not found: {path}")
