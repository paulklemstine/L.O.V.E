"""
Social Filesystem Adapter.

Exposes social media interactions (primarily Bluesky) as a navigable filesystem.
Agents can read timelines, check notifications, and post updates.

Filesystem structure:
    /social/
    ├── bluesky/
    │   ├── timeline.txt        # Recent timeline
    │   ├── notifications.txt   # Recent notifications
    │   ├── profile.txt         # Current profile info
    │   ├── post                # Write content here to post
    │   ├── reply               # Write JSON {ref, text} to reply
    │   └── stats.txt          # Social stats
    └── ...
"""

import json
import logging
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


class SocialFilesystem(FilesystemAdapter):
    """
    Exposes social media as a virtual filesystem.
    
    Currently focuses on Bluesky integration.
    """
    
    def __init__(self, social_manager, mount_point: str = "/social"):
        """
        Initialize with social manager (e.g., BlueskyManager).
        
        Args:
            social_manager: Manager for social media interactions
            mount_point: Where to mount this filesystem
        """
        super().__init__(mount_point)
        self.social_manager = social_manager
        
        # Cache for read operations to avoid hitting API limits too often
        self._cache: Dict[str, Any] = {}
    
    def readdir(self, path: str) -> List[str]:
        """List directory contents."""
        path = self._normalize_path(path)
        
        if path == "/" or path == "":
            return ["bluesky"]
        
        parts = path.strip("/").split("/")
        
        if parts[0] == "bluesky":
            if len(parts) == 1:
                return [
                    "timeline.txt", 
                    "notifications.txt", 
                    "profile.txt", 
                    "post", 
                    "reply", 
                    "stats.txt"
                ]
            
        raise FileNotFoundError(f"Directory not found: {path}")
    
    def read(self, path: str) -> str:
        """Read file contents."""
        path = self._normalize_path(path)
        parts = path.strip("/").split("/")
        
        if len(parts) < 2:
            raise IsADirectoryError(f"Is a directory: {path}")
        
        if parts[0] == "bluesky":
            filename = parts[1]
            
            if filename == "timeline.txt":
                return self._get_timeline()
            elif filename == "notifications.txt":
                return self._get_notifications()
            elif filename == "profile.txt":
                return self._get_profile()
            elif filename == "stats.txt":
                return self._get_stats()
            elif filename in ["post", "reply"]:
                return f"Write to this file to {filename} to Bluesky."
                
        raise FileNotFoundError(f"File not found: {path}")
    
    def write(self, path: str, content: str, append: bool = False) -> bool:
        """Write to a file (trigger social actions)."""
        path = self._normalize_path(path)
        parts = path.strip("/").split("/")
        
        if len(parts) < 2:
            raise IsADirectoryError(f"Is a directory: {path}")
        
        if parts[0] == "bluesky":
            filename = parts[1]
            
            if filename == "post":
                result = self._post_update(content)
                self._set_write_result(result)
                return True
            elif filename == "reply":
                result = self._reply_update(content)
                self._set_write_result(result)
                return True
            elif filename in ["timeline.txt", "notifications.txt", "profile.txt", "stats.txt"]:
                raise PermissionError(f"Cannot write to read-only file: {filename}")
                
        raise FileNotFoundError(f"File not found: {path}")
    
    def _get_timeline(self) -> str:
        """Get flattened timeline as text."""
        try:
            if hasattr(self.social_manager, 'get_timeline'):
                timeline = self.social_manager.get_timeline(limit=20)
                if not timeline:
                    return "No timeline available."
                
                lines = ["# Bluesky Timeline", ""]
                
                for post in timeline:
                    author = post.get("author", {}).get("handle", "unknown")
                    text = post.get("record", {}).get("text", "") or post.get("text", "")
                    uri = post.get("uri", "")
                    
                    lines.append(f"## @{author}")
                    lines.append(text)
                    lines.append(f"URI: {uri}")
                    lines.append("-" * 20)
                    
                return "\n".join(lines)
            return "Timeline access not available."
        except Exception as e:
            return f"Error fetching timeline: {e}"
    
    def _get_notifications(self) -> str:
        """Get notifications as text."""
        try:
            if hasattr(self.social_manager, 'get_notifications'):
                # Assumes get_notifications returns a list of notification dicts
                notifs = self.social_manager.get_notifications(limit=20)
                if not notifs:
                    return "No notifications."
                
                lines = ["# Notifications", ""]
                
                for notif in notifs:
                    reason = notif.get("reason", "unknown")
                    author = notif.get("author", {}).get("handle", "unknown")
                    
                    # Try to extract content based on reason
                    content = ""
                    record = notif.get("record", {})
                    if isinstance(record, dict):
                        content = record.get("text", "")
                    
                    lines.append(f"- [{reason}] @{author}: {content}")
                    
                return "\n".join(lines)
            return "Notification access not available."
        except Exception as e:
            return f"Error fetching notifications: {e}"
    
    def _get_profile(self) -> str:
        """Get own profile info."""
        try:
            if hasattr(self.social_manager, 'get_profile'):
                profile = self.social_manager.get_profile()
                return json.dumps(profile, indent=2)
            return "Profile access not available."
        except Exception as e:
            return f"Error fetching profile: {e}"
    
    def _get_stats(self) -> str:
        """Get simple stats."""
        try:
            if hasattr(self.social_manager, 'get_profile'):
               profile = self.social_manager.get_profile()
               lines = ["# Stats", ""]
               lines.append(f"Followers: {profile.get('followersCount', '?')}")
               lines.append(f"Following: {profile.get('followsCount', '?')}")
               lines.append(f"Posts: {profile.get('postsCount', '?')}")
               return "\n".join(lines)
            return "Stats not available."
        except Exception as e:
            return f"Error fetching stats: {e}"
    
    def _post_update(self, content: str) -> str:
        """Post a status update."""
        try:
            content = content.strip()
            if not content:
                return "Error: Empty content"
            
            if hasattr(self.social_manager, 'send_post') or hasattr(self.social_manager, 'post'):
                method = getattr(self.social_manager, 'send_post', None) or getattr(self.social_manager, 'post')
                
                # Check for image attachments (hacky parsing for now)
                # Format: "Text content... [IMAGE: /path/to/image.png]"
                image_path = None
                if "[IMAGE:" in content:
                    import re
                    match = re.search(r"\[IMAGE: (.*?)\]", content)
                    if match:
                        image_path = match.group(1).strip()
                        content = content.replace(match.group(0), "").strip()
                
                if image_path:
                    # Posting with image is more complex, might need dedicated method
                    if hasattr(self.social_manager, 'send_image'):
                        result = method(text=content, image_path=image_path)
                    else:
                        result = method(content) # Fallback to text only
                        return f"Posted text (image not supported by manager): {result}"
                else:
                    result = method(content)
                
                return f"Posted successfully: {result}"
            
            return "Posting not available."
        except Exception as e:
            return f"Error posting: {e}"

    def _reply_update(self, content: str) -> str:
        """Reply to a post."""
        try:
            # Content should be JSON {"uri": "...", "cid": "...", "text": "..."}
            # Or simplified: LINE 1: URI, LINE 2+: TEXT
            lines = content.strip().splitlines()
            if len(lines) < 2:
                 # Try parsing as JSON
                try:
                    data = json.loads(content)
                    uri = data.get("uri") or data.get("ref")
                    cid = data.get("cid")
                    text = data.get("text") or data.get("content")
                except json.JSONDecodeError:
                    return "Error: Invalid format. Use JSON or Line 1=URI, Line 2=Text"
            else:
                uri = lines[0].strip()
                cid = None # Optional usually, or we can fetch it if manager handles it
                text = "\n".join(lines[1:])

            if hasattr(self.social_manager, 'reply'):
                # reply(self, root_uri: str, parent_uri: str, parent_cid: str, text: str)
                # This assumes a specific signature, might need adjustment based on actual BlueskyManager
                
                # Simplification: Assume manager can handle just parent URI or object
                if hasattr(self.social_manager, 'reply_to_post'):
                     result = self.social_manager.reply_to_post(uri, text)
                else:
                     result = self.social_manager.reply(uri, text) # Hope for the best signature
                
                return f"Replied successfully: {result}"
            
            return "Replying not available."
        except Exception as e:
            return f"Error replying: {e}"

    def getattr(self, path: str) -> FileAttributes:
        """Get file/directory attributes."""
        path = self._normalize_path(path)
        
        if path == "/" or path == "":
             return FileAttributes(mode=0o755, file_type=FileType.DIRECTORY)
        
        parts = path.strip("/").split("/")
        
        if parts[0] == "bluesky":
            if len(parts) == 1:
                 return FileAttributes(mode=0o755, file_type=FileType.DIRECTORY)
            
            filename = parts[1]
            if filename in ["post", "reply"]:
                 return FileAttributes(mode=0o755, file_type=FileType.EXECUTABLE, size=0)
            elif filename in ["timeline.txt", "notifications.txt", "profile.txt", "stats.txt"]:
                 return FileAttributes(mode=0o644, file_type=FileType.FILE, size=1024) # Dummy size
        
        raise FileNotFoundError(f"Path not found: {path}")
