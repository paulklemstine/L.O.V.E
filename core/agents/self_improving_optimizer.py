from typing import Dict
from core.agents.specialist_agent import SpecialistAgent
from core.agents.analyst_agent import AnalystAgent
from core.metacognition import HypothesisFormatter, ExperimentPlanner
from core.agents.code_gen_agent import CodeGenerationAgent
from core.benchmarker import AutomatedBenchmarker
from core.version_control import GitManager
from core.gemini_react_engine import GeminiReActEngine
from core.tools import ToolRegistry, read_file, evolve

class SelfImprovingOptimizer(SpecialistAgent):
    """
    A specialist agent that orchestrates the entire metacognitive evolution loop,
    from analyzing performance to deploying validated improvements.
    """

    def __init__(self):
        # These are components used internally by this specialist agent.
        self.hypothesis_formatter = HypothesisFormatter()
        self.experiment_planner = ExperimentPlanner()
        self.benchmarker = AutomatedBenchmarker()
        self.git_manager = GitManager()
        # This specialist might internally use other specialists.
        self.analyst = AnalystAgent()
        self.code_generator = CodeGenerationAgent()


    async def execute_task(self, task_details: Dict) -> Dict:
        """
        Executes a self-improvement task based on the provided details.

        Args:
            task_details: A dictionary containing:
              - 'task_type': 'improve_module' or 'run_evolution_cycle'
              - and other necessary parameters based on the task type.
                - for 'improve_module': 'module_path', 'objective'
                - for 'run_evolution_cycle': 'logs'

        Returns:
            A dictionary with the status and result of the improvement cycle.
        """
        task_type = task_details.get("task_type")
        if not task_type:
            return {"status": "failure", "result": "No task_type specified for SelfImprovingOptimizer."}

        if task_type == "improve_module":
            result = await self._improve_module(task_details)
            return {"status": "success", "result": result}
        elif task_type == "run_evolution_cycle":
            result = await self._run_evolution_cycle(task_details)
            return {"status": "success", "result": result}
        else:
            return {"status": "failure", "result": f"Unknown task_type: {task_type}"}

    async def _improve_module(self, task_details: Dict) -> str:
        """
        Applies intelligence to improve its own code based on a high-level objective.
        """
        module_path = task_details.get("module_path")
        objective = task_details.get("objective")
        print(f"\n===== Starting Self-Improvement Cycle for {module_path} =====")
        print(f"Objective: {objective}")

        try:
            tool_registry = ToolRegistry()
            tool_registry.register_tool("read_file", read_file, {"description": "mocked tool"})
            tool_registry.register_tool("evolve", evolve, {"description": "mocked tool"})
            tool_registry.register_tool("run_experiment", self.benchmarker.run_experiment, {"description": "mocked tool"})
            gemini_react_engine = GeminiReActEngine(tool_registry)
        except FileNotFoundError:
            error_msg = "Error: gemini-cli is not available. Cannot run self-improvement cycle."
            print(error_msg)
            return error_msg

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

    async def _run_evolution_cycle(self, task_details: Dict) -> str:
        """
        Executes one full cycle of self-improvement based on performance logs.
        """
        logs = task_details.get("logs")
        if not logs:
            return "No logs provided for evolution cycle."

        print("\n===== Starting Metacognitive Evolution Cycle =====")

        # 1. Performance Logging & Causal Reflection
        analysis_result = await self.analyst.execute_task({"logs": logs})
        insight = analysis_result.get("result")
        if not insight or "No significant patterns" in insight:
            msg = "SelfImprovingOptimizer: No actionable insights found. Ending cycle."
            print(msg)
            return msg

        # 2. Hypothesis & Experiment Design
        hypothesis = self.hypothesis_formatter.format_hypothesis(insight)
        if "No hypothesis" in hypothesis:
            msg = "SelfImprovingOptimizer: Could not form a hypothesis. Ending cycle."
            print(msg)
            return msg
        print(f"SelfImprovingOptimizer: Formed hypothesis: {hypothesis}")

        experiment_plan = self.experiment_planner.design_experiment(hypothesis)
        if not experiment_plan:
            msg = "SelfImprovingOptimizer: Could not design an experiment. Ending cycle."
            print(msg)
            return msg
        print(f"SelfImprovingOptimizer: Designed experiment: {experiment_plan}")

        # 3. Autonomous Code Modification
        code_gen_result = await self.code_generator.execute_task({"hypothesis": hypothesis})
        new_code = code_gen_result.get("result")
        if code_gen_result.get("status") == 'failure' or not new_code:
            msg = f"SelfImprovingOptimizer: Code generation failed. Reason: {new_code}"
            print(msg)
            return msg

        # 4. Validation in Sandbox
        is_validated = self.benchmarker.run_experiment(experiment_plan, new_code)
        if not is_validated:
            msg = "SelfImprovingOptimizer: Hypothesis was not validated by the experiment. Ending cycle."
            print(msg)
            return msg

        print("SelfImprovingOptimizer: Hypothesis validated successfully!")

        # 5. Integration & Deployment
        branch_name = f"feature/improve-{experiment_plan['variant']}"
        commit_message = f"feat: Improve {experiment_plan['control']} with {experiment_plan['variant']}\n\nHypothesis: {hypothesis}"
        file_to_update = "core/tools_updated.py"
        with open(file_to_update, "w") as f: f.write(new_code)
        self.git_manager.create_branch(branch_name)
        self.git_manager.commit_changes(file_to_update, commit_message)
        self.git_manager.submit_pull_request(
            title=f"Self-Improvement: {experiment_plan['name']}",
            body=f"This automated PR was generated to improve system performance based on the following hypothesis:\n\n> {hypothesis}"
        )

        msg = "===== Metacognitive Evolution Cycle Finished Successfully ====="
        print(msg)
        return msg
