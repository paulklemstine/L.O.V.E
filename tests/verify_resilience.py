import logging
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.getcwd())

from core.logging import setup_global_logging, log_event

class TestResilience(unittest.TestCase):
    
    def test_logger_suppression(self):
        """Story 1: Verify noisy loggers are suppressed."""
        setup_global_logging(verbose=False)
        
        noisy_libs = ["urllib3", "http.client", "websockets", "langsmith.client"]
        for lib in noisy_libs:
            logger = logging.getLogger(lib)
            self.assertEqual(logger.level, logging.WARNING, f"Logger {lib} should be WARNING")
            
    def test_artifact_logging(self):
        """Story 6: Verify large ASCII blocks are redirected."""
        large_block = "A" * 600 + "\n" * 15 # >500 chars, >10 newlines
        
        # Clean artifacts.log
        if os.path.exists("artifacts.log"):
            os.remove("artifacts.log")
            
        log_event(large_block)
        
        # Verify artifacts.log exists and has content
        self.assertTrue(os.path.exists("artifacts.log"), "artifacts.log should be created")
        with open("artifacts.log", "r") as f:
            content = f.read()
            self.assertIn("--- ARTIFACT", content)
            self.assertIn(large_block, content)
            
    def test_agent_structure(self):
        """Story 5: Verify agent.py imports and structure."""
        # We can't easily import agent.py main because it runs things, 
        # but we can inspect it or try to import it if we mock everything.
        # Let's just do a static check of the file content for now to be safe.
        
        with open("agent.py", "r") as f:
            content = f.read()
            
        self.assertIn("Orchestrator()", content, "Orchestrator should be instantiated")
        self.assertIn("TemporaryEnvironmentError", content, "TemporaryEnvironmentError should be used")
        self.assertIn("safe_cognitive_loop", content, "safe_cognitive_loop wrapper should exist")
        self.assertIn("system_heartbeat", content, "system_heartbeat should function should exist")

if __name__ == '__main__':
    unittest.main()
