"""
Background Docker image pre-warmer to avoid blocking on first use.

This module provides a SandboxPrewarmer class that builds the Docker sandbox
image in a background thread during L.O.V.E. startup, so it's ready when needed.
"""
import threading
import logging
from typing import Optional

from core.logging import log_event


class SandboxPrewarmer:
    """
    Pre-warms Docker sandbox in background thread.
    
    Usage:
        # On startup (non-blocking)
        prewarmer = SandboxPrewarmer.get_instance()
        prewarmer.start_prewarm()
        
        # Later, when sandbox is needed
        if prewarmer.is_ready():
            # Use sandbox
        else:
            # Fall back to subprocess
    """
    
    _instance: Optional["SandboxPrewarmer"] = None
    _lock = threading.Lock()
    
    def __init__(self):
        self.ready = threading.Event()
        self.error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._build_started = False
    
    @classmethod
    def get_instance(cls) -> "SandboxPrewarmer":
        """Returns singleton instance of the prewarmer."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance
    
    def start_prewarm(self) -> bool:
        """
        Starts background Docker build. Non-blocking.
        
        Returns:
            True if prewarm was started, False if already running or completed.
        """
        with self._lock:
            if self._build_started:
                return False  # Already started or completed
            self._build_started = True
        
        self._thread = threading.Thread(
            target=self._prewarm_worker,
            daemon=True,
            name="SandboxPrewarmer"
        )
        self._thread.start()
        log_event("Docker sandbox pre-warm started in background", "INFO")
        return True
    
    def _prewarm_worker(self):
        """Background worker that builds the Docker image."""
        try:
            from core.surgeon.sandbox import DockerSandbox, is_docker_available
            
            if not is_docker_available():
                log_event("Docker not available, skipping pre-warm", "INFO")
                self.ready.set()
                return
            
            log_event("Pre-warming Docker sandbox image...", "INFO")
            sandbox = DockerSandbox()
            sandbox.ensure_image_exists()  # This is the slow part
            log_event("✅ Docker sandbox pre-warmed successfully", "INFO")
            
        except Exception as e:
            self.error = str(e)
            log_event(f"Docker pre-warm failed: {e}", "WARNING")
        finally:
            self.ready.set()
    
    def wait_for_ready(self, timeout: float = None) -> bool:
        """
        Blocks until sandbox is ready.
        
        Args:
            timeout: Maximum seconds to wait (None = forever)
            
        Returns:
            True if ready, False if timed out.
        """
        return self.ready.wait(timeout=timeout)
    
    def is_ready(self) -> bool:
        """Non-blocking check if sandbox is ready."""
        return self.ready.is_set()
    
    def has_error(self) -> bool:
        """Returns True if pre-warm encountered an error."""
        return self.error is not None
    
    def get_error(self) -> Optional[str]:
        """Returns error message if pre-warm failed, else None."""
        return self.error


def get_prewarmer() -> SandboxPrewarmer:
    """Convenience function to get the singleton prewarmer."""
    return SandboxPrewarmer.get_instance()
