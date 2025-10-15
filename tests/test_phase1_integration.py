import unittest
import os
import sys

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.agents.orchestrator import Orchestrator
from core.agents.execution import ResearchAgent
from core.perception.information_extractor import extract_entities_and_relations
from core.knowledge_graph.graph import KnowledgeGraph
from core.memory.memory_manager import MemoryManager

class TestPhase1Integration(unittest.TestCase):

    def setUp(self):
        """Set up the test environment before each test."""
        self.orchestrator = Orchestrator()
        self.research_agent = ResearchAgent()
        self.orchestrator.register_agent("researcher", self.research_agent)
        self.kg = KnowledgeGraph(db_path="test_kg.json")
        self.memory = MemoryManager(ltm_path="test_ltm.json")

    def tearDown(self):
        """Clean up the test environment after each test."""
        if os.path.exists("test_kg.json"):
            os.remove("test_kg.json")
        if os.path.exists("test_ltm.json"):
            os.remove("test_ltm.json")

    def test_full_workflow(self):
        """
        Tests the full workflow:
        1. Orchestrator delegates a task.
        2. Information is extracted from a sample text.
        3. The Knowledge Graph is populated.
        4. The outcome is stored in Long-Term Memory.
        5. A relevant memory is retrieved.
        """
        # 1. Orchestrator delegates a task
        task = "Analyze the provided text about major tech companies."
        result = self.orchestrator.delegate_task("researcher", task)
        self.assertIn("Research complete", result)

        # 2. Information is extracted from a sample text (simulating the agent's research)
        sample_text = "Apple Inc. is based in Cupertino. Google, a subsidiary of Alphabet Inc., is a major player in search technology."
        relations = extract_entities_and_relations(sample_text)
        self.assertTrue(len(relations) > 0)
        self.assertIn(('Apple Inc.', 'related_to', 'Cupertino'), relations)

        # 3. The Knowledge Graph is populated
        for sub, rel, obj in relations:
            self.kg.add_relation(sub, rel, obj)
        self.kg.save_graph()
        self.assertIn(('Google', 'related_to', 'Alphabet Inc.'), self.kg.get_triples())

        # 4. The outcome is stored in Long-Term Memory
        self.memory.add_episode(task, "Extracted relations from text and updated KG.", success=True)
        self.assertEqual(len(self.memory.episodes), 1)

        # 5. A relevant memory is retrieved
        similar_task = "Find information on technology corporations."
        relevant_memories = self.memory.retrieve_relevant_memories(similar_task)
        self.assertIn("Task: Analyze the provided text about major tech companies. | Outcome: Extracted relations from text and updated KG. | Success: True", relevant_memories[0])

if __name__ == '__main__':
    unittest.main()