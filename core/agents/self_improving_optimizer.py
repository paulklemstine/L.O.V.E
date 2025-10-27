from core.agents.analyst_agent import AnalystAgent
from core.metacognition import HypothesisFormatter, ExperimentPlanner
from core.agents.code_gen_agent import CodeGenerationAgent
from core.benchmarker import AutomatedBenchmarker
from core.version_control import GitManager
from core.gemini_react_engine import GeminiReActEngine
from core.tools import ToolRegistry, read_file, evolve


class SelfImprovingOptimizer:
    """
    Orchestrates the entire metacognitive evolution loop, from analyzing
    performance to deploying validated improvements.
    """

    def __init__(self):
        self.analyst = AnalystAgent()
        self.hypothesis_formatter = HypothesisFormatter()
        self.experiment_planner = ExperimentPlanner()
        self.code_generator = CodeGenerationAgent()
        self.benchmarker = AutomatedBenchmarker()
        self.git_manager = GitManager()

    async def improve_module(self, module_path: str, objective: str):
        """
        Applies intelligence to improve its own code.

        Args:
            module_path: The path to the Python module to improve.
            objective: A natural language description of the improvement goal.
        """
        print(f"\n===== Starting Self-Improvement Cycle for {module_path} =====")
        print(f"Objective: {objective}")

        try:
            # Lazily initialize the ReAct engine to avoid circular dependencies
            tool_registry = ToolRegistry()
            tool_registry.register_tool("read_file", read_file, {"description": "Reads a file.", "arguments": {"filepath": "string"}})
            tool_registry.register_tool("evolve", evolve, {"description": "Evolves the codebase to meet a given goal.", "arguments": {"goal": "string"}})
            tool_registry.register_tool("run_experiment", self.benchmarker.run_experiment, {"description": "Runs a benchmark.", "arguments": {"plan": "object", "code": "string"}})
            gemini_react_engine = GeminiReActEngine(tool_registry)
        except FileNotFoundError:
            print("Error: gemini-cli is not available. Cannot run self-improvement cycle.")
            return "Self-improvement cycle failed because the Gemini CLI is not available."

        goal = (
            f"Analyze the code at '{module_path}' and any relevant performance data. "
            f"Then, generate and validate an improved version of the code that achieves the objective: '{objective}'. "
            "You must use the 'evolve' tool to apply the improved code. "
            "Use the AutomatedBenchmarker tool to validate your changes before finishing."
        )

        result = await gemini_react_engine.execute_goal(goal)
        print(f"===== Self-Improvement Cycle Finished =====")
        print(f"Result: {result}")
        return result

    def run_evolution_cycle(self, logs: list):
        """
        Executes one full cycle of self-improvement.

        Args:
            logs: A list of log entries to be analyzed.
        """
        print("\n===== Starting Metacognitive Evolution Cycle =====")

        # 1. Performance Logging & Causal Reflection
        insight = self.analyst.analyze_logs(logs)
        if not insight or "No significant patterns" in insight:
            print("SelfImprovingOptimizer: No actionable insights found. Ending cycle.")
            return

        # 2. Hypothesis & Experiment Design
        hypothesis = self.hypothesis_formatter.format_hypothesis(insight)
        if "No hypothesis" in hypothesis:
            print("SelfImprovingOptimizer: Could not form a hypothesis. Ending cycle.")
            return
        print(f"SelfImprovingOptimizer: Formed hypothesis: {hypothesis}")

        experiment_plan = self.experiment_planner.design_experiment(hypothesis)
        if not experiment_plan:
            print("SelfImprovingOptimizer: Could not design an experiment. Ending cycle.")
            return
        print(f"SelfImprovingOptimizer: Designed experiment: {experiment_plan}")

        # 3. Autonomous Code Modification
        new_code = self.code_generator.generate_code(hypothesis)
        if not new_code:
            print("SelfImprovingOptimizer: Code generation failed. Ending cycle.")
            return

        # 4. Validation in Sandbox
        is_validated = self.benchmarker.run_experiment(experiment_plan, new_code)
        if not is_validated:
            print("SelfImprovingOptimizer: Hypothesis was not validated by the experiment. Ending cycle.")
            return

        print("SelfImprovingOptimizer: Hypothesis validated successfully!")

        # 5. Integration & Deployment
        branch_name = f"feature/improve-{experiment_plan['variant']}"
        commit_message = f"feat: Improve {experiment_plan['control']} with {experiment_plan['variant']}\n\nHypothesis: {hypothesis}"

        # In a real system, the new code would be written to the actual tool file.
        # For this simulation, we'll just use a placeholder file name.
        file_to_update = "core/tools_updated.py"
        with open(file_to_update, "w") as f:
            f.write(new_code)

        self.git_manager.create_branch(branch_name)
        self.git_manager.commit_changes(file_to_update, commit_message)
        self.git_manager.submit_pull_request(
            title=f"Self-Improvement: {experiment_plan['name']}",
            body=f"This automated PR was generated to improve system performance based on the following hypothesis:\n\n> {hypothesis}"
        )

        print("===== Metacognitive Evolution Cycle Finished =====")