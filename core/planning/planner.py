from .decomposition import DecompositionModule
from .validator import PlanValidator

class Planner:
    def __init__(self):
        self.decomposition_module = DecompositionModule()
        self.validator = PlanValidator()
        print("Planner: Initialized.")

    def create_plan(self, goal):
        """
        Creates a new plan by decomposing a goal and validating the result.
        """
        print(f"Planner: Creating plan for goal '{goal}'...")
        plan = self.decomposition_module.decompose(goal)

        if self.validator.validate(plan):
            print("Planner: Plan created and validated successfully.")
            return plan
        else:
            print("Planner: Failed to create a valid plan.")
            return None