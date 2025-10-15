class PlanValidator:
    def validate(self, plan):
        """
        Validates a generated plan.
        This basic version checks if the plan is not empty and has sequential steps.
        """
        print("PlanValidator: Validating plan...")
        if not plan:
            print("PlanValidator: Validation FAILED - Plan is empty.")
            return False

        for i, step in enumerate(plan):
            if step.get("step") != i + 1:
                print(f"PlanValidator: Validation FAILED - Step {i+1} has incorrect step number.")
                return False

        print("PlanValidator: Validation PASSED.")
        return True