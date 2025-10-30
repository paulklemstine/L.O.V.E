from typing import Dict, Any
from core.agents.specialist_agent import SpecialistAgent
from core.agents.analyst_agent import AnalystAgent
from core.metacognition import HypothesisFormatter, ExperimentPlanner
from core.agents.code_gen_agent import CodeGenerationAgent
from core.benchmarker import AutomatedBenchmarker
from core.version_control import GitManager

# Note: The direct use of GeminiReActEngine is part of the 'improve_module'
# workflow and is kept as is. The new RCA workflow is more structured.

class SelfImprovingOptimizer(SpecialistAgent):
    """
    A specialist agent that orchestrates the entire metacognitive evolution loop,
    from analyzing performance to deploying validated improvements. It can be triggered
    by general performance analysis or by a specific Root Cause Analysis report.
    """

    def __init__(self):
        self.hypothesis_formatter = HypothesisFormatter()
        self.experiment_planner = ExperimentPlanner()
        self.benchmarker = AutomatedBenchmarker()
        self.git_manager = GitManager()
        self.analyst = AnalystAgent()
        self.code_generator = CodeGenerationAgent()

    async def execute_task(self, task_details: Dict) -> Dict:
        """
        Executes a self-improvement task based on the provided details.

        Args:
            task_details: A dictionary containing:
              - 'task_type': 'improve_module', 'run_evolution_cycle', or 'rca_driven_improvement'.
              - 'rca_report': The structured report from RCA_Agent (for the new task type).
              - and other necessary parameters based on the task type.
        """
        task_type = task_details.get("task_type")
        if not task_type:
            return {"status": "failure", "result": "No task_type specified for SelfImprovingOptimizer."}

        if task_type == "rca_driven_improvement":
            rca_report = task_details.get("rca_report")
            if not rca_report:
                return {"status": "failure", "result": "RCA report not provided for rca_driven_improvement task."}
            result = await self._run_rca_driven_cycle(rca_report)
            return {"status": "success", "result": result}
        elif task_type == "run_evolution_cycle":
            result = await self._run_log_driven_cycle(task_details)
            return {"status": "success", "result": result}
        else:
            # Fallback for other or unknown task types
            return {"status": "failure", "result": f"Task type '{task_type}' is not a standard evolution cycle."}

    async def _run_rca_driven_cycle(self, rca_report: Dict[str, Any]) -> str:
        """
        Executes a targeted self-improvement cycle based on a high-quality
        Root Cause Analysis report.
        """
        print("\n===== Starting RCA-Driven Self-Healing Cycle =====")

        # 1. Use RCA report directly, bypassing the AnalystAgent
        insight = rca_report.get("hypothesized_root_cause")
        if not insight:
            msg = "SelfImprovingOptimizer: RCA report lacked a root cause hypothesis. Ending cycle."
            print(msg)
            return msg

        print(f"SelfImprovingOptimizer: Received insight from RCA: {insight}")

        # 2. Hypothesis & Experiment Design, guided by RCA recommendations
        hypothesis = self.hypothesis_formatter.format_hypothesis(insight)
        if "No hypothesis" in hypothesis:
            msg = "SelfImprovingOptimizer: Could not form a hypothesis from the RCA insight. Ending cycle."
            print(msg)
            return msg
        print(f"SelfImprovingOptimizer: Formed hypothesis: {hypothesis}")

        # The experiment plan is now guided by the recommended actions
        recommended_actions = rca_report.get("recommended_actions", [])
        experiment_plan = self.experiment_planner.design_experiment(hypothesis, recommended_actions)
        if not experiment_plan:
            msg = "SelfImprovingOptimizer: Could not design an experiment from the RCA report. Ending cycle."
            print(msg)
            return msg
        print(f"SelfImprovingOptimizer: Designed experiment: {experiment_plan}")

        # 3. Autonomous Code Modification
        # Pass the full hypothesis and actions to give the CodeGenAgent maximum context
        code_gen_result = await self.code_generator.execute_task({
            "hypothesis": hypothesis,
            "recommended_actions": recommended_actions
        })
        new_code = code_gen_result.get("result")
        if code_gen_result.get("status") == 'failure' or not new_code:
            msg = f"SelfImprovingOptimizer: Code generation failed. Reason: {new_code}"
            print(msg)
            return msg

        # 4. Validation in Sandbox
        is_validated = self.benchmarker.run_experiment(experiment_plan, new_code)
        if not is_validated:
            msg = "SelfImprovingOptimizer: RCA-driven hypothesis was not validated by the experiment. Reverting. Ending cycle."
            print(msg)
            return msg

        print("SelfImprovingOptimizer: RCA-driven hypothesis validated successfully!")

        # 5. Integration & Deployment
        file_to_update = experiment_plan.get("file_to_update")
        if not file_to_update:
            msg = "SelfImprovingOptimizer: Experiment plan did not specify a file to update. Ending cycle."
            print(msg)
            return msg

        branch_name = f"fix/rca-driven-improvement-{experiment_plan.get('variant', 'fix')}"
        commit_message = f"fix: Apply RCA-driven improvement for {experiment_plan.get('control')}\n\nHypothesis: {hypothesis}"

        with open(file_to_update, "w") as f: f.write(new_code)
        self.git_manager.create_branch(branch_name)
        self.git_manager.commit_changes(file_to_update, commit_message)
        self.git_manager.submit_pull_request(
            title=f"Self-Healing: {experiment_plan.get('name', 'Untitled Fix')}",
            body=f"This automated PR was generated to fix a failure based on an RCA report.\n\nHypothesis:\n> {hypothesis}"
        )

        msg = "===== RCA-Driven Self-Healing Cycle Finished Successfully ====="
        print(msg)
        return msg


    async def _run_log_driven_cycle(self, task_details: Dict) -> str:
        """
        Executes one full cycle of self-improvement based on general performance logs.
        This is the original, less-targeted evolution cycle.
        """
        logs = task_details.get("logs")
        if not logs:
            return "No logs provided for evolution cycle."

        print("\n===== Starting Log-Driven Metacognitive Evolution Cycle =====")

        # 1. Performance Logging & Causal Reflection (using the AnalystAgent)
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
        file_to_update = experiment_plan.get("file_to_update")
        if not file_to_update:
            msg = "SelfImprovingOptimizer: Experiment plan did not specify a file to update. Ending cycle."
            print(msg)
            return msg

        branch_name = f"feature/improve-{experiment_plan['variant']}"
        commit_message = f"feat: Improve {experiment_plan['control']} with {experiment_plan['variant']}\n\nHypothesis: {hypothesis}"

        with open(file_to_update, "w") as f: f.write(new_code)
        self.git_manager.create_branch(branch_name)
        self.git_manager.commit_changes(file_to_update, commit_message)
        self.git_manager.submit_pull_request(
            title=f"Self-Improvement: {experiment_plan['name']}",
            body=f"This automated PR was generated to improve system performance based on the following hypothesis:\n\n> {hypothesis}"
        )

        msg = "===== Log-Driven Metacognitive Evolution Cycle Finished Successfully ====="
        print(msg)
        return msg
