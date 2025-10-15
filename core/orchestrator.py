from .planning.planner import Planner
from .tools.registry import ToolRegistry
from .tools.executor import SecureExecutor
from .tools.web_search import WebSearchTool
from .reflection.logger import StructuredEventLogger
from .agents.analyst_agent import AnalystAgent
from .reflection.hypothesis import HypothesisFormatter
from .reflection.experiment import ExperimentPlanner
from .agents.code_gen_agent import CodeGenerationAgent
from .reflection.sandbox import SandboxEnvironment
from .reflection.benchmarker import AutomatedBenchmarker
from .deployment.vcs import VersionControlIntegration

class Orchestrator:
    def __init__(self):
        self.planner = Planner()
        self.tool_registry = ToolRegistry()
        self.executor = SecureExecutor(self.tool_registry)
        self.event_logger = StructuredEventLogger()
        self.analyst_agent = AnalystAgent()
        self.hypothesis_formatter = HypothesisFormatter()
        self.experiment_planner = ExperimentPlanner()
        self.code_gen_agent = CodeGenerationAgent()
        self.sandbox = SandboxEnvironment()
        self.benchmarker = AutomatedBenchmarker()
        self.vcs = VersionControlIntegration()
        self.plan_state = []
        self._register_tools()
        print("Orchestrator: Initialized with integrated modules.")

    def _register_tools(self):
        """Initializes and registers all available tools."""
        self.tool_registry.register_tool(WebSearchTool())
        # In the future, other tools would be registered here.

    def execute_goal(self, goal):
        """
        The main execution loop for a high-level goal.
        """
        print(f"\nOrchestrator: Received new goal: '{goal}'")

        # 1. Create a plan
        plan = self.planner.create_plan(goal)
        if not plan:
            print("Orchestrator: Halting execution due to planning failure.")
            return

        # 2. Execute the plan
        self.plan_state = [{"step": p, "status": "pending"} for p in plan]

        for i, step_info in enumerate(self.plan_state):
            step = step_info["step"]
            task = step["task"]
            tool_name = step["tool"]

            print(f"\n--- Executing Step {step['step']}: {task} (using tool: {tool_name}) ---")

            # For this simulation, we'll use the task description as the tool input.
            result, success = self.executor.execute_tool(tool_name, task)

            if success:
                self.plan_state[i]["status"] = "completed"
                self.plan_state[i]["result"] = result
                print(f"Step {step['step']} successful. Result: {result}")
            else:
                self.plan_state[i]["status"] = "failed"
                print(f"Step {step['step']} failed. Halting plan execution.")
                # Self-correction logic would be triggered here.
                # For now, we just stop.
                break

        print("\nOrchestrator: Plan execution finished.")
        return self.plan_state

    def run_evolution_cycle(self):
        """
        Executes a full metacognitive evolution loop.
        """
        print("\n" + "="*50)
        print("Orchestrator: Starting Metacognitive Evolution Cycle...")

        # Step 3.1: Log a simulated event and perform analysis
        self.event_logger.log_event("tool_failure", {"tool_name": "web_search", "reason": "Timeout"})
        insight = self.analyst_agent.analyze_logs()
        if not insight or "No significant patterns" in insight:
            print("Orchestrator: No actionable insights found. Ending cycle.")
            return

        # Step 3.2: Formulate hypothesis and plan experiment
        hypothesis = self.hypothesis_formatter.format_hypothesis(insight)
        if not hypothesis:
            print("Orchestrator: Could not formulate a hypothesis. Ending cycle.")
            return
        experiment_plan = self.experiment_planner.create_experiment_plan(hypothesis)
        if not experiment_plan:
            print("Orchestrator: Could not create an experiment plan. Ending cycle.")
            return

        # Step 3.3: Generate and validate new code
        new_code = self.code_gen_agent.generate_code(hypothesis)
        if not new_code or not self.sandbox.validate_code(new_code):
            print("Orchestrator: Code generation or validation failed. Ending cycle.")
            return

        # Step 3.4: Run benchmark and deploy
        success, benchmark_result = self.benchmarker.run_benchmark(new_code, experiment_plan)
        if not success:
            print("Orchestrator: Benchmark did not confirm hypothesis. Ending cycle.")
            return

        commit_message = f"feat(web_search): Evolve tool based on benchmark\n\n{benchmark_result['conclusion']}"
        pr_url = self.vcs.submit_pull_request(new_code, commit_message)

        print(f"Orchestrator: Evolution cycle complete. Pull request submitted at {pr_url}")
        print("="*50)
        return pr_url