"""
test_gastown.py - Integration tests for the Gastown architecture
"""

import sys
import unittest
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from love2.core.beads import BeadChain, BeadState, Bead
from love2.core.workers import WorkerSwarm, SocialWorker
from love2.core.deep_loop import DeepLoop

class TestGastown(unittest.TestCase):
    def setUp(self):
        # Use a temporary directory for state
        self.test_dir = Path("tests/temp_state")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.bead_path = self.test_dir / "beads.json"
        
        # Reset singleton if possible or just create new instances
        self.chain = BeadChain(persistence_path=str(self.bead_path))
        self.swarm = WorkerSwarm()
        
    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_bead_lifecycle(self):
        """Test creating, saving, and loading beads."""
        # Create
        bead = self.chain.create_bead("Test Task", priority=1)
        self.assertEqual(bead.status, BeadState.PENDING)
        self.assertTrue(self.bead_path.exists())
        
        # Load new chain
        new_chain = BeadChain(persistence_path=str(self.bead_path))
        loaded_bead = new_chain.get_bead(bead.id)
        self.assertIsNotNone(loaded_bead)
        self.assertEqual(loaded_bead.description, "Test Task")
        
    def test_worker_dispatch(self):
        """Test dispatching a bead to a worker."""
        bead = self.chain.create_bead("Post to Bluesky: Hello World")
        
        # Mock LLM in worker
        with patch('love2.core.workers.get_llm_client') as mock_llm_getter:
            mock_llm = MagicMock()
            mock_llm_getter.return_value = mock_llm
            
            # Mock the tool call response
            mock_llm.generate_json.return_value = {
                "thought": "I shoud post this",
                "tool": "generate_post_content",
                "args": {"topic": "Hello World", "auto_post": True}
            }
            
            # Mock tools
            with patch('love2.core.workers.get_adapted_tools') as mock_tools_getter:
                mock_gen = MagicMock(return_value={"success": True, "post_uri": "at://..."})
                mock_tools_getter.return_value = {"generate_post_content": mock_gen}
                
                # Dispatch
                success = self.swarm.dispatch(bead)
                
                self.assertTrue(success)
                self.assertEqual(bead.status, BeadState.COMPLETED)
                self.assertIn("at://...", bead.result)

    def test_mayor_planning(self):
        """Test the Mayor creating beads."""
        loop = DeepLoop(sleep_seconds=0)
        loop.bead_chain = self.chain # Inject test chain
        loop.llm = MagicMock()
        
        # Mock Persona
        loop.persona = MagicMock()
        loop.persona.get_actionable_goals.return_value = [MagicMock(text="Dominate the world")]
        loop.persona.get_persona_context.return_value = "I am a Goddess"
        
        # Mock Memory
        loop.memory = MagicMock()
        loop.memory.episodic.get_recent_events.return_value = []
        
        # Mock LLM response for planning
        loop.llm.generate_json.return_value = {
            "thought": "We need to start small",
            "commands": [
                {"type": "create_bead", "description": "Write a manifesto", "priority": 1}
            ]
        }
        
        # Run one iteration
        loop.run_iteration()
        
        # Verify bead created (it may be completed or failed due to immediate processing)
        beads = list(self.chain.beads.values())
        self.assertTrue(len(beads) > 0, "No beads created")
        bead = beads[0]
        self.assertEqual(bead.description, "Write a manifesto")

if __name__ == '__main__':
    unittest.main()
