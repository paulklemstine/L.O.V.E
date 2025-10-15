class ExperimentPlanner:
    def create_experiment_plan(self, hypothesis):
        """
        Outlines a test plan for a given hypothesis.
        """
        print("ExperimentPlanner: Creating experiment plan...")
        if "CSS selectors" in hypothesis:
            plan = {
                "name": "Test Web Search Efficiency",
                "hypothesis": hypothesis,
                "control": "current web_search tool implementation",
                "variant": "web_search tool with CSS selector logic",
                "metric": "token_usage",
                "success_condition": "variant.token_usage < control.token_usage * 0.5"
            }
            print(f"ExperimentPlanner: Created plan: {plan}")
            return plan
        return None