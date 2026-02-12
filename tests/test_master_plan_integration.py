import os
import sys
import unittest

sys.path.append(os.getcwd())

from core.persona_goal_extractor import get_persona_extractor

class TestMasterPlanIntegration(unittest.TestCase):
    def test_goals_loaded(self):
        extractor = get_persona_extractor()
        # Force reload to pick up new logic
        extractor.reload()
        
        goals = extractor.get_all_goals()
        print(f"\nFound {len(goals)} total goals.")
        
        mp_goals = [g for g in goals if "master_plan" in g.category]
        print(f"Found {len(mp_goals)} master plan goals.")
        
        for g in mp_goals[:5]:
            print(f" - [{g.priority}] {g.text} ({g.category})")
            
        self.assertGreater(len(mp_goals), 0, "Should have loaded master plan goals")
        
        # Check priority
        tasks = [g for g in mp_goals if g.category == "master_plan_task"]
        if tasks:
            self.assertEqual(tasks[0].priority, 1, "Tasks should be priority 1")

if __name__ == "__main__":
    unittest.main()
