from typing import Dict
from core.agents.specialist_agent import SpecialistAgent
from core.agents.code_gen_agent import CodeGenerationAgent
from core.benchmarker import AutomatedBenchmarker
from core.version_control import GitManager
from core.gemini_react_engine import GeminiReActEngine
from core.tools import ToolRegistry, read_file, evolve
from core.llm_api import run_llm

class SelfImprovingOptimizer(SpecialistAgent):
    """
    A specialist agent that orchestrates the entire metacognitive evolution loop,
    from analyzing performance to deploying validated improvements.
    """

    def __init__(self, memory_manager=None):
        # These are components used internally by this specialist agent.
        self.benchmarker = AutomatedBenchmarker()
        self.git_manager = GitManager()
        self.memory_manager = memory_manager
        # This specialist might internally use other specialists.
        self.code_generator = CodeGenerationAgent()


    async def execute_task(self, task_details: Dict) -> Dict:
        """
        Executes a self-improvement task based on the provided details.

        Args:
            task_details: A dictionary containing:
              - 'task_type': 'improve_module' or 'run_evolution_cycle'
              - and other necessary parameters based on the task type.
                - for 'improve_module': 'module_path', 'objective'
                - for 'run_evolution_cycle': 'insight'

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
            result = await self._run_evolution_cycle(task_details.get('insight'))
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

    async def _run_evolution_cycle(self, insight: str) -> str:
        """
        Executes one full cycle of self-improvement based on a high-level insight.
        """
        if not insight:
            return "No insight provided for evolution cycle."

        print("\n===== Starting Metacognitive Evolution Cycle from Insight =====")
        print(f"Insight: {insight}")

        # 1. Hypothesis Generation
        prompt = f"""
        You are a hypothesis engine for a self-improving AI. Given the following high-level insight about a potential inefficiency, formulate a concrete and testable hypothesis for a code modification.
        The hypothesis should be in the format: "IF I [change], THEN [expected outcome], BECAUSE [reason]."

        Insight:
        ---
        {insight}
        ---

        Now, generate the hypothesis.
        """
        hypothesis = await run_llm(prompt)
        if "IF" not in hypothesis:
            msg = f"SelfImprovingOptimizer: Could not form a valid hypothesis from insight. LLM response: {hypothesis}"
            print(msg)
            return msg
        print(f"SelfImprovingOptimizer: Formed hypothesis: {hypothesis}")

        # 2. Autonomous Code Modification
        code_gen_result = await self.code_generator.execute_task({"hypothesis": hypothesis})
        new_code = code_gen_result.get("result")
        if code_gen_result.get("status") == 'failure' or not new_code:
            msg = f"SelfImprovingOptimizer: Code generation failed. Reason: {new_code}"
            print(msg)
            return msg

        # 3. Validation (Simplified for now - assumes code is correct)
        print("SelfImprovingOptimizer: Code generated. For now, we assume it is correct.")

        # 4. Integration & Deployment
        branch_name = f"feature/self-improve-{uuid.uuid4()[:8]}"
        commit_message = f"feat: Self-improve based on insight\n\nInsight: {insight}\nHypothesis: {hypothesis}"
        # This part is complex. For now, we'll just log the proposed change.
        # A real implementation would need to determine the file to change.
        print(f"PROPOSED CHANGE on branch {branch_name}:\n{new_code}")

        # Story 3.3: Remember the act of evolution
        if self.memory_manager:
            evolution_summary = f"Cognitive Event: Self-Improvement | Insight: '{insight}' | Hypothesis: '{hypothesis}'."
            await self.memory_manager.add_episode(evolution_summary)


        msg = "===== Metacognitive Evolution Cycle Finished Successfully (Simulation) ====="
        print(msg)
        return msg
