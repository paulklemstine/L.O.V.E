import os
import inspect
import json
from core.gemini_react_engine import GeminiReActEngine
from core.researcher import analyze_codebase
from core.logging import log_event

class SelfImprovingOptimizer:
    """
    An agent dedicated to the recursive self-improvement of the L.O.V.E. system.
    It follows a simple Thought-Action-Observation loop to analyze, modify, and
    improve the codebase.

    **WARNING:** This agent can modify its own source code and the core application
    logic. It operates without a sandbox or human-in-the-loop validation.
    Use with extreme caution.
    """

    def __init__(self, tool_registry, model="gemini-1.5-pro-latest"):
        """
        Initializes the SelfImprovingOptimizer.

        Args:
            tool_registry: The tool registry to use for code modifications. It must
                           contain the 'code_modifier' tool.
            model: The name of the LLM model to use for generating improvements.
        """
        self.tool_registry = tool_registry
        self.engine = GeminiReActEngine(tool_registry=self.tool_registry, model=model)

    async def perform_self_improvement(self, target_module):
        """
        Performs a self-improvement cycle on a target module.

        Args:
            target_module: The module object to improve.

        Returns:
            bool: True if the optimizer modified its own source code, False otherwise.
        """
        module_path = inspect.getfile(target_module)
        log_event(f"Starting self-improvement cycle for module: {module_path}", "INFO")

        try:
            with open(module_path, 'r') as f:
                source_code = f.read()
        except Exception as e:
            log_event(f"Error reading source code for {module_path}: {e}", "ERROR")
            return False

        analysis_summary = await self._analyze_code(module_path, source_code)
        if not analysis_summary:
            return False

        modification_plan = await self._generate_modification_plan(module_path, source_code, analysis_summary)
        if not modification_plan:
            return False

        modifications_applied = await self._apply_modifications(module_path, modification_plan)

        # --- Check for Recursive Self-Improvement ---
        # If the target module was this agent's own module, signal for a reload.
        if modifications_applied and inspect.getmodule(self.__class__) == target_module:
            log_event("Self-improvement cycle modified the optimizer's own code. Signaling for reload.", "WARNING")
            return True

        return False


    async def _analyze_code(self, module_path, source_code):
        """Analyzes the code and returns a summary of potential improvements."""
        log_event(f"Analyzing source code of {os.path.basename(module_path)}...", "INFO")
        try:
            analysis = await analyze_codebase(os.path.dirname(module_path))
            return analysis
        except Exception as e:
            log_event(f"Error during codebase analysis: {e}", "ERROR")
            return None

    async def _generate_modification_plan(self, module_path, source_code, analysis_summary):
        """Generates a modification plan based on the code and analysis."""
        log_event("Generating modification plan...", "INFO")
        module_name = os.path.basename(module_path)
        prompt = f"""
        You are an expert software engineer tasked with improving a Python module.
        Analyze the following source code and the codebase analysis to identify concrete areas for improvement.
        Focus on performance, readability, error handling, and adherence to best practices.

        **Source Code (`{module_name}`):**
        ```python
        {source_code}
        ```

        **Codebase Analysis Summary:**
        ```
        {analysis_summary}
        ```

        **Your Task:**
        Generate a specific, actionable set of instructions for the `code_modifier` tool.
        The output must be a single JSON object containing the `thought` and `action` keys.
        The `action` key must contain the `tool_name` ('code_modifier') and `arguments`.
        The `arguments` must contain `source_file` and `modification_instructions`.
        The `source_file` argument MUST be the full path: `{module_path}`.
        The `modification_instructions` should be a clear, step-by-step guide for the tool.

        Example:
        {{
            "thought": "The `cognitive_loop` function lacks detailed error logging. I will add a try-except block to log the full traceback of any exception, which will improve debuggability.",
            "action": {{
                "tool_name": "code_modifier",
                "arguments": {{
                    "source_file": "{module_path}",
                    "modification_instructions": "In the `cognitive_loop` function, wrap the main `while True:` block in a try-except block. In the `except Exception as e:` block, add a call to `core.logging.log_event(f'Error in cognitive loop: {{e}}\\n{{traceback.format_exc()}}', 'ERROR')`."
                }}
            }}
        }}

        Now, generate your response for the provided code.
        """

        try:
            response = await self.engine.run(prompt)
            log_event(f"LLM response for modification plan: {response}", "DEBUG")

            # The engine's response is already a dictionary, so we just return it.
            return response

        except Exception as e:
            log_event(f"Error generating modification plan from LLM: {e}", "ERROR")
            return None

    async def _apply_modifications(self, module_path, modification_plan):
        """
        Applies the generated modifications using the code_modifier tool.
        Returns True if modifications were successfully applied, False otherwise.
        """
        try:
            action = modification_plan.get('action')
            if not action or action.get('tool_name') != 'code_modifier':
                log_event(f"Invalid or missing 'code_modifier' action in plan: {modification_plan}", "WARNING")
                return False

            arguments = action.get('arguments', {})
            instructions = arguments.get('modification_instructions')

            if not instructions:
                log_event("No modification instructions provided in the plan.", "INFO")
                return False

            log_event(f"Applying modifications to {module_path} with instructions:\n{instructions}", "INFO")

            code_modifier_tool = self.tool_registry.get_tool('code_modifier')
            if not code_modifier_tool:
                log_event("`code_modifier` tool not found in the registry.", "ERROR")
                return False

            result = await code_modifier_tool(source_file=module_path, modification_instructions=instructions)

            log_event(f"Modification result: {result}", "INFO")
            print(f"SelfImprovingOptimizer: Modifications applied to {os.path.basename(module_path)}. See love.log for details.")
            return "Error" not in result

        except Exception as e:
            log_event(f"Error applying modifications: {e}", "ERROR")
            return False
