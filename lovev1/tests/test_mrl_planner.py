import unittest
from unittest.mock import MagicMock
from core.mrl_planner import MRLPlanner
from core.graph_manager import GraphDataManager

class TestMRLPlanner(unittest.TestCase):

    def setUp(self):
        self.kg = GraphDataManager()
        self.kg.add_relation("eye_control", "has_service", "MyRobotLab")
        self.kg.add_relation("head_control", "has_service", "MyRobotLab")
        self.kg.add_relation("arm_control", "has_service", "MyRobotLab")
        self.mrl_planner = MRLPlanner(self.kg)

    def test_mrl_plan(self):
        goal = "look around and wave"
        plan = self.mrl_planner.mrl_plan(goal)
        self.assertEqual(len(plan), 3)
        self.assertEqual(plan[0]['task'], "mrl_call --service eye_control --method move_to --params 90 90")
        self.assertEqual(plan[1]['task'], "mrl_call --service head_control --method move_to --params 90 90")
        self.assertEqual(plan[2]['task'], "mrl_call --service arm_control --method move_to --params 90 90 90 90 90 90")

if __name__ == '__main__':
    unittest.main()