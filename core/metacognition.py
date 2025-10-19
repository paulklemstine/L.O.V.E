class HypothesisFormatter:
    @staticmethod
    def format_hypothesis(insight: str) -> str:
        """
        Structures an insight into a formal, testable hypothesis.
        Example: IF we [make change], THEN [expected outcome] WILL OCCUR, as measured by [metric].
        """
        # This is a simplified implementation based on the insight from AnalystAgent
        if "inefficient" in insight and "perform_webrequest" in insight:
            return (
                "IF we modify the perform_webrequest tool to use targeted CSS selectors, "
                "THEN its token usage will decrease by over 50% "
                "WILL OCCUR, as measured by token_usage_metric."
            )
        return "No hypothesis generated."

class ExperimentPlanner:
    @staticmethod
    def design_experiment(hypothesis: str) -> dict:
        """
        Outlines a test plan based on the hypothesis.
        """
        if "CSS selectors" in hypothesis:
            return {
                "name": "Experiment: Efficient Web Search",
                "control": "current_web_search_tool",
                "variant": "new_web_search_tool_with_selectors",
                "metric": "token_usage_metric",
                "success_condition": "variant.token_usage < control.token_usage * 0.5"
            }
        return {}