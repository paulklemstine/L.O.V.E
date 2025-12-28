"""
Story 6.2: Collaborative Editor

Detects external file changes to prevent overwriting
human edits during agent operations.
"""
import os
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
from core.logging import log_event


@dataclass
class FileState:
    """Represents the state of a file at a point in time."""
    path: str
    hash: str
    mtime: float
    size: int
    captured_at: float
    
    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "hash": self.hash,
            "mtime": self.mtime,
            "size": self.size,
            "captured_at": self.captured_at,
        }


# Cache of tracked file states
_file_states: Dict[str, FileState] = {}


def compute_file_hash(path: str) -> str:
    """
    Computes MD5 hash of file contents.
    
    Args:
        path: Path to file
        
    Returns:
        Hex digest of file hash
    """
    try:
        with open(path, "rb") as f:
            content = f.read()
        return hashlib.md5(content).hexdigest()
    except Exception as e:
        log_event(f"Failed to hash file {path}: {e}", "ERROR")
        return ""


def get_file_state(path: str) -> Optional[FileState]:
    """
    Gets the current state of a file.
    
    Args:
        path: Path to file
        
    Returns:
        FileState object or None if file doesn't exist
    """
    if not os.path.exists(path):
        return None
    
    try:
        stat = os.stat(path)
        file_hash = compute_file_hash(path)
        
        state = FileState(
            path=path,
            hash=file_hash,
            mtime=stat.st_mtime,
            size=stat.st_size,
            captured_at=datetime.now().timestamp()
        )
        
        return state
        
    except Exception as e:
        log_event(f"Failed to get file state for {path}: {e}", "ERROR")
        return None


def track_file(path: str) -> Optional[FileState]:
    """
    Starts tracking a file for changes.
    
    Args:
        path: Path to file to track
        
    Returns:
        Current FileState
    """
    state = get_file_state(path)
    if state:
        _file_states[path] = state
        log_event(f"Tracking file: {path}", "DEBUG")
    return state


def has_external_changes(path: str, last_state: FileState = None) -> bool:
    """
    Checks if file has changed since last tracked state.
    
    Args:
        path: Path to file
        last_state: Optional explicit state to compare against
        
    Returns:
        True if file has been modified externally
    """
    # Get state to compare against
    if last_state is None:
        last_state = _file_states.get(path)
    
    if last_state is None:
        # Not tracked, assume no changes
        return False
    
    current_state = get_file_state(path)
    
    if current_state is None:
        # File was deleted
        log_event(f"File deleted externally: {path}", "WARNING")
        return True
    
    # Compare hash (most reliable)
    if current_state.hash != last_state.hash:
        log_event(f"File changed externally (hash mismatch): {path}", "WARNING")
        return True
    
    # Compare mtime as backup
    if current_state.mtime > last_state.mtime:
        log_event(f"File touched externally (mtime changed): {path}", "WARNING")
        return True
    
    return False


def check_before_write(path: str) -> dict:
    """
    Checks if it's safe to write to a file.
    
    Args:
        path: Path to file to write
        
    Returns:
        Dict with 'safe', 'reason', and 'action' keys
    """
    result = {
        "safe": True,
        "reason": None,
        "action": "proceed",
        "current_state": None,
        "tracked_state": None,
    }
    
    tracked_state = _file_states.get(path)
    current_state = get_file_state(path)
    
    result["current_state"] = current_state
    result["tracked_state"] = tracked_state
    
    if tracked_state is None:
        # New file or not tracked
        result["reason"] = "File not tracked"
        return result
    
    if has_external_changes(path, tracked_state):
        result["safe"] = False
        result["reason"] = "File has been modified externally since last read"
        result["action"] = "prompt_user"
        
        log_event(
            f"External changes detected in {path}. "
            f"Old hash: {tracked_state.hash[:8]}, "
            f"New hash: {current_state.hash[:8] if current_state else 'N/A'}",
            "WARNING"
        )
    
    return result


def confirm_write(path: str, choice: str = "overwrite") -> bool:
    """
    Handles user choice about file conflict.
    
    Args:
        path: Path to file
        choice: "overwrite" or "reload"
        
    Returns:
        True if write should proceed
    """
    if choice == "overwrite":
        # User chose to overwrite - update tracked state after write
        log_event(f"User chose to overwrite external changes in {path}", "INFO")
        return True
    
    elif choice == "reload":
        # User wants to reload - update tracked state to current
        new_state = get_file_state(path)
        if new_state:
            _file_states[path] = new_state
            log_event(f"Reloaded file state for {path}", "INFO")
        return False
    
    return False


def update_after_write(path: str) -> None:
    """
    Updates tracked state after writing to a file.
    
    Call this after successfully writing to update the tracked state.
    
    Args:
        path: Path to file that was written
    """
    track_file(path)
    log_event(f"Updated tracked state after write: {path}", "DEBUG")


def untrack_file(path: str) -> None:
    """
    Stops tracking a file.
    
    Args:
        path: Path to file
    """
    if path in _file_states:
        del _file_states[path]
        log_event(f"Stopped tracking: {path}", "DEBUG")


def get_tracked_files() -> list:
    """Returns list of currently tracked file paths."""
    return list(_file_states.keys())


class FileWatcher:
    """
    Context manager for safe file editing.
    
    Usage:
        with FileWatcher(path) as fw:
            content = fw.read()
            # modify content
            if fw.check_safe():
                fw.write(new_content)
    """
    
    def __init__(self, path: str):
        self.path = path
        self.initial_state = None
        self.content = None
    
    def __enter__(self):
        self.initial_state = track_file(self.path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
    
    def read(self) -> Optional[str]:
        """Reads file content."""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.content = f.read()
            return self.content
        except Exception as e:
            log_event(f"Failed to read {self.path}: {e}", "ERROR")
            return None
    
    def check_safe(self) -> bool:
        """Checks if safe to write."""
        result = check_before_write(self.path)
        return result["safe"]
    
    def write(self, content: str, force: bool = False) -> bool:
        """
        Writes content if safe.
        
        Args:
            content: New content to write
            force: If True, bypass safety check
            
        Returns:
            True if write succeeded
        """
        if not force and not self.check_safe():
            log_event(f"Refusing to write to {self.path} - external changes detected", "WARNING")
            return False
        
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(content)
            update_after_write(self.path)
            return True
        except Exception as e:
            log_event(f"Failed to write to {self.path}: {e}", "ERROR")
            return False
