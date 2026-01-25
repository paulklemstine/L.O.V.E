"""
Memory Filesystem Adapter.

Exposes L.O.V.E.'s memory systems (episodic, semantic, working) as a navigable filesystem.
Agents can browse memories, search through episodes, and manage working memory.

Filesystem structure:
    /memories/
    ├── episodes/
    │   ├── 2026-01-13/
    │   │   ├── 09-14_bluesky_post.txt
    │   │   └── 08-30_code_evolution.txt
    │   └── ...
    ├── working/
    │   ├── current_goal.txt        # Current goal/task
    │   ├── scratch.txt             # Scratch space for notes
    │   └── plan.txt                # Current plan
    ├── semantic/
    │   ├── recent.txt              # Recent semantic memories
    │   └── search                  # Write query to search
    └── stats.txt                   # Memory statistics
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


class MemoryFilesystem(FilesystemAdapter):
    """
    Exposes memory systems as a virtual filesystem.
    
    Organizess memories into:
    - episodes/: Episodic memories organized by date
    - working/: Working memory (current task, scratch, plan)
    - semantic/: Semantic memory search
    """
    
    def __init__(self, memory_manager, mount_point: str = "/memories"):
        """
        Initialize with a MemoryManager instance.
        
        Args:
            memory_manager: The L.O.V.E. memory manager
            mount_point: Where to mount this filesystem
        """
        super().__init__(mount_point)
        self.memory_manager = memory_manager
        
        # Working memory storage (in-memory for now)
        self._working: Dict[str, str] = {
            "current_goal.txt": "",
            "scratch.txt": "",
            "plan.txt": "",
        }
        self._search_results: str = ""
    
    def _get_episodes(self) -> List[Dict]:
        """Get all episodic memories."""
        try:
            if hasattr(self.memory_manager, 'get_recent_episodes'):
                return self.memory_manager.get_recent_episodes(100)
            elif hasattr(self.memory_manager, 'episodes'):
                return list(self.memory_manager.episodes)
            return []
        except Exception as e:
            logger.error(f"Error getting episodes: {e}")
            return []
    
    def _get_episodes_by_date(self) -> Dict[str, List[Dict]]:
        """Group episodes by date."""
        episodes = self._get_episodes()
        by_date: Dict[str, List[Dict]] = {}
        
        for ep in episodes:
            # Extract date from episode
            timestamp = ep.get("timestamp") or ep.get("created_at") or ep.get("time")
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
            by_date[date_str].append(ep)
        
        return by_date
    
    def _episode_to_filename(self, episode: Dict, index: int = 0) -> str:
        """Convert an episode to a filename."""
        timestamp = episode.get("timestamp") or episode.get("created_at") or ""
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                time_part = dt.strftime("%H-%M")
            except ValueError:
                time_part = f"{index:04d}"
        else:
            time_part = f"{index:04d}"
        
        # Get a summary from the episode
        content = episode.get("content") or episode.get("text") or episode.get("summary") or ""
        if isinstance(content, str):
            # Create a slug from first few words
            words = content.split()[:4]
            slug = "_".join(words)[:30]
            slug = "".join(c if c.isalnum() or c == "_" else "_" for c in slug)
        else:
            slug = "episode"
        
        return f"{time_part}_{slug}.txt"
    
    def _episode_to_content(self, episode: Dict) -> str:
        """Convert an episode to readable content."""
        lines = []
        
        # Timestamp
        timestamp = episode.get("timestamp") or episode.get("created_at") or ""
        if timestamp:
            lines.append(f"Timestamp: {timestamp}")
        
        # Type/Category
        ep_type = episode.get("type") or episode.get("category") or "general"
        lines.append(f"Type: {ep_type}")
        
        # Content
        content = episode.get("content") or episode.get("text") or ""
        if content:
            lines.append("")
            lines.append("Content:")
            lines.append(content)
        
        # Metadata
        metadata = {k: v for k, v in episode.items() 
                   if k not in ["timestamp", "created_at", "type", "category", "content", "text"]}
        if metadata:
            lines.append("")
            lines.append("Metadata:")
            lines.append(json.dumps(metadata, indent=2, default=str))
        
        return "\n".join(lines)
    
    def readdir(self, path: str) -> List[str]:
        """List directory contents."""
        path = self._normalize_path(path)
        
        if path == "/" or path == "":
            return ["episodes", "working", "semantic", "stats.txt"]
        
        parts = path.strip("/").split("/")
        
        if parts[0] == "episodes":
            if len(parts) == 1:
                # List dates
                by_date = self._get_episodes_by_date()
                return sorted(by_date.keys(), reverse=True)
            elif len(parts) == 2:
                # List episodes for a date
                date = parts[1]
                by_date = self._get_episodes_by_date()
                if date not in by_date:
                    raise FileNotFoundError(f"No episodes for date: {date}")
                return [self._episode_to_filename(ep, i) for i, ep in enumerate(by_date[date])]
        
        elif parts[0] == "working":
            if len(parts) == 1:
                return list(self._working.keys())
        
        elif parts[0] == "semantic":
            if len(parts) == 1:
                return ["recent.txt", "search", "results.txt"]
        
        raise FileNotFoundError(f"Directory not found: {path}")
    
    def read(self, path: str) -> str:
        """Read file contents."""
        path = self._normalize_path(path)
        parts = path.strip("/").split("/")
        
        if len(parts) == 1:
            if parts[0] == "stats.txt":
                return self._get_stats()
            raise IsADirectoryError(f"Is a directory: {path}")
        
        if parts[0] == "episodes":
            if len(parts) == 2:
                raise IsADirectoryError(f"Is a directory: {path}")
            elif len(parts) == 3:
                date = parts[1]
                filename = parts[2]
                by_date = self._get_episodes_by_date()
                if date not in by_date:
                    raise FileNotFoundError(f"Date not found: {date}")
                
                # Find episode by filename
                for i, ep in enumerate(by_date[date]):
                    if self._episode_to_filename(ep, i) == filename:
                        return self._episode_to_content(ep)
                raise FileNotFoundError(f"Episode not found: {filename}")
        
        elif parts[0] == "working":
            if len(parts) == 2:
                filename = parts[1]
                if filename in self._working:
                    return self._working[filename]
                # Also check memory manager for working memory
                if hasattr(self.memory_manager, 'get_working_memory'):
                    wm = self.memory_manager.get_working_memory()
                    if isinstance(wm, dict):
                        if filename == "current_goal.txt":
                            return wm.get("current_goal", "")
                        elif filename == "plan.txt":
                            plan = wm.get("plan", [])
                            if isinstance(plan, list):
                                return "\n".join(f"- {step}" for step in plan)
                            return str(plan)
                raise FileNotFoundError(f"File not found: {filename}")
        
        elif parts[0] == "semantic":
            if len(parts) == 2:
                filename = parts[1]
                if filename == "recent.txt":
                    return self._get_recent_semantic()
                elif filename == "results.txt":
                    return self._search_results
                elif filename == "search":
                    return "Write a query to this file to search semantic memory.\nResults will appear in results.txt"
        
        raise FileNotFoundError(f"File not found: {path}")
    
    def write(self, path: str, content: str, append: bool = False) -> bool:
        """Write to a file."""
        path = self._normalize_path(path)
        parts = path.strip("/").split("/")
        
        if len(parts) < 2:
            raise IsADirectoryError(f"Is a directory: {path}")
        
        if parts[0] == "working":
            filename = parts[1]
            if filename in self._working:
                if append:
                    self._working[filename] += content
                else:
                    self._working[filename] = content
                
                # Also update memory manager if available
                if hasattr(self.memory_manager, 'update_working_memory'):
                    try:
                        if filename == "current_goal.txt":
                            self.memory_manager.update_working_memory(current_goal=content)
                        elif filename == "plan.txt":
                            steps = [line.lstrip("- ") for line in content.strip().split("\n") if line.strip()]
                            self.memory_manager.update_working_memory(plan=steps)
                    except Exception as e:
                        logger.warning(f"Failed to update memory manager: {e}")
                
                return True
            raise FileNotFoundError(f"File not found: {filename}")
        
        elif parts[0] == "semantic" and len(parts) == 2 and parts[1] == "search":
            # Perform semantic search
            self._search_results = self._search_semantic(content.strip())
            self._set_write_result(self._search_results)
            return True
        
        elif parts[0] == "episodes":
            # Create new episode
            if len(parts) >= 2:
                try:
                    if hasattr(self.memory_manager, 'add_episode'):
                        self.memory_manager.add_episode(content)
                    self._set_write_result("Episode added successfully")
                    return True
                except Exception as e:
                    raise PermissionError(f"Failed to add episode: {e}")
        
        raise PermissionError(f"Cannot write to: {path}")
    
    def _get_stats(self) -> str:
        """Get memory statistics."""
        lines = ["# Memory Statistics", ""]
        
        episodes = self._get_episodes()
        lines.append(f"Total episodes: {len(episodes)}")
        
        by_date = self._get_episodes_by_date()
        lines.append(f"Days with episodes: {len(by_date)}")
        
        if by_date:
            recent_date = max(by_date.keys())
            lines.append(f"Most recent: {recent_date}")
        
        # Working memory
        lines.append("")
        lines.append("## Working Memory")
        for filename, content in self._working.items():
            size = len(content)
            lines.append(f"- {filename}: {size} bytes")
        
        return "\n".join(lines)
    
    def _get_recent_semantic(self) -> str:
        """Get recent semantic memories."""
        try:
            if hasattr(self.memory_manager, 'get_recent_memories'):
                memories = self.memory_manager.get_recent_memories(10)
                if memories:
                    return "\n\n---\n\n".join(str(m) for m in memories)
            return "No recent semantic memories available."
        except Exception as e:
            return f"Error retrieving semantic memories: {e}"
    
    def _search_semantic(self, query: str) -> str:
        """Search semantic memory."""
        try:
            if hasattr(self.memory_manager, 'search') or hasattr(self.memory_manager, 'query'):
                method = getattr(self.memory_manager, 'search', None) or getattr(self.memory_manager, 'query')
                results = method(query)
                if results:
                    if isinstance(results, list):
                        return "\n\n---\n\n".join(str(r) for r in results)
                    return str(results)
            return "No results found."
        except Exception as e:
            return f"Search error: {e}"
    
    def getattr(self, path: str) -> FileAttributes:
        """Get file/directory attributes."""
        path = self._normalize_path(path)
        
        if path == "/" or path == "":
            return FileAttributes(mode=0o755, file_type=FileType.DIRECTORY)
        
        parts = path.strip("/").split("/")
        
        if parts[0] in ["episodes", "working", "semantic"]:
            if len(parts) == 1:
                return FileAttributes(mode=0o755, file_type=FileType.DIRECTORY)
            
            if parts[0] == "episodes" and len(parts) == 2:
                # Date directory
                by_date = self._get_episodes_by_date()
                if parts[1] in by_date:
                    return FileAttributes(mode=0o755, file_type=FileType.DIRECTORY)
                raise FileNotFoundError(f"Date not found: {parts[1]}")
            
            # Files
            try:
                content = self.read(path)
                return FileAttributes(
                    mode=0o644,
                    file_type=FileType.FILE,
                    size=len(content.encode())
                )
            except (FileNotFoundError, IsADirectoryError):
                raise FileNotFoundError(f"Path not found: {path}")
        
        elif parts[0] == "stats.txt":
            content = self._get_stats()
            return FileAttributes(
                mode=0o644,
                file_type=FileType.FILE,
                size=len(content.encode())
            )
        
        raise FileNotFoundError(f"Path not found: {path}")
