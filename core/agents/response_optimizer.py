import os
import json
import time
import asyncio
from rich.console import Console
from core.logging import log_event
from core.llm_api import run_llm
from core.prompt_registry import get_prompt_registry

class ResponseOptimizer:
    """
    An agent dedicated to the recursive optimization of response generation efficiency.
    It periodically reviews response generation logs, identifies patterns of redundancy
    and latency, generates refactoring plans, and applies updates to prompts or configurations.
    """

    def __init__(self, tool_registry, monitoring_manager=None, check_interval=1000, latency_threshold_ms=500):
        """
        Initializes the ResponseOptimizer.

        Args:
            tool_registry: The tool registry to use for modifications.
            monitoring_manager: The monitoring manager to access performance metrics.
            check_interval: Number of queries between optimization checks.
            latency_threshold_ms: Latency threshold in milliseconds to trigger optimization.
        """
        self.tool_registry = tool_registry
        self.monitoring_manager = monitoring_manager
        self.check_interval = check_interval
        self.latency_threshold_ms = latency_threshold_ms
        self.console = Console()
        self.model_stats_file = "llm_model_stats.json"

    async def run_optimization_cycle(self):
        """
        Runs a single optimization cycle with safeguards.
        """
        # Check if optimization is needed
        if not self._check_trigger():
            return

        log_event("Starting Response Optimization Cycle...", "INFO")

        # 1. Analyze Logs
        stats = self._load_model_stats()
        if not stats:
            log_event("No model stats available for optimization.", "WARNING")
            return

        inefficiencies = await self._identify_inefficiencies(stats)
        if not inefficiencies:
            log_event("No significant inefficiencies found.", "INFO")
            return

        # 2. Generate Refactoring Plan
        plan = await self._generate_refactoring_plan(inefficiencies)
        if not plan:
            log_event("Failed to generate refactoring plan.", "WARNING")
            return

        # 3. Apply Updates with Validation and Rollback
        success = await self._apply_updates_with_safeguards(plan)

        if success:
            log_event("Optimization updates applied and verified successfully.", "INFO")
        else:
            log_event("Optimization cycle failed or was rolled back.", "WARNING")

    def _load_model_stats(self):
        """Loads model statistics from the JSON file."""
        try:
            if os.path.exists(self.model_stats_file):
                with open(self.model_stats_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            log_event(f"Error loading model stats: {e}", "ERROR")
        return None

    async def _identify_inefficiencies(self, stats):
        """
        Identifies inefficiencies based on model stats.
        Returns a summary string of identified issues.
        """
        issues = []
        for model_id, data in stats.items():
            successful_calls = data.get("successful_calls", 0)
            if successful_calls > 0:
                avg_latency = (data.get("total_time_spent", 0) / successful_calls) * 1000
                if avg_latency > self.latency_threshold_ms:
                    issues.append(f"High latency for {model_id}: {avg_latency:.2f}ms (Threshold: {self.latency_threshold_ms}ms)")

                # Simple heuristic for token inefficiency
                avg_tokens = data.get("total_tokens_generated", 0) / successful_calls
                if avg_tokens > 2000: # Arbitrary high threshold
                    issues.append(f"High token usage for {model_id}: {avg_tokens:.2f} tokens/call")

        if not issues:
            return None

        return "\n".join(issues)

    async def _generate_refactoring_plan(self, inefficiencies):
        """
        Uses an LLM to generate a refactoring plan based on identified inefficiencies.
        """
        prompt = f"""
        You are a system optimization expert. Analyze the following inefficiencies in our LLM response generation pipeline:

        {inefficiencies}

        Propose a specific, actionable plan to improve efficiency.
        Focus on prompt optimization (shortening, making more concise) or model selection changes.

        Your output must be a JSON object with the following structure:
        {{
            "analysis": "Brief analysis of the problem",
            "proposed_changes": [
                {{
                    "type": "prompt_update",
                    "prompt_key": "key_from_prompts_yaml",
                    "new_prompt_content": "The optimized prompt content..."
                }}
            ]
        }}

        Note: If you suggest a prompt update, you MUST provide the 'prompt_key' and the 'new_prompt_content'.
        """

        try:
            response = await run_llm(prompt, purpose="optimization")
            if response and response.get("result"):
                return self._parse_json_response(response["result"])
        except Exception as e:
            log_event(f"Error generating refactoring plan: {e}", "ERROR")
        return None

    def _parse_json_response(self, text):
        """Extracts and parses JSON from LLM response."""
        try:
            # Find JSON block
            import re
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(text)
        except Exception as e:
            log_event(f"Error parsing JSON from plan: {e}", "ERROR")
            return None

    async def _apply_updates_with_safeguards(self, plan):
        """
        Applies updates with backup and validation.
        """
        registry = get_prompt_registry()

        for change in plan.get("proposed_changes", []):
            if change.get("type") == "prompt_update":
                key = change.get("prompt_key")
                new_content = change.get("new_prompt_content")

                if not key or not new_content:
                    continue

                log_event(f"Attempting to optimize prompt '{key}'...", "INFO")

                # 1. Backup original prompt
                original_content = registry.get_prompt(key)
                if not original_content:
                    log_event(f"Prompt '{key}' not found. Skipping.", "WARNING")
                    continue

                # 2. Apply update
                if not registry.update_prompt(key, new_content):
                    log_event(f"Failed to apply update for '{key}'.", "ERROR")
                    continue

                # 3. Validation
                valid = await self._validate_improvement(key, new_content)

                if valid:
                    log_event(f"Prompt '{key}' optimization verified.", "INFO")
                else:
                    log_event(f"Validation failed for '{key}'. Rolling back...", "WARNING")
                    registry.update_prompt(key, original_content)
                    return False

        return True

    async def _validate_improvement(self, prompt_key, new_content):
        """
        Validates the improvement by running a test call and checking metrics.
        Returns True if the improvement is valid (performance is better or acceptable).
        """
        # Create a simple test call
        # Note: Ideally we would have a specific test input for each prompt key.
        # For now, we use a generic placeholder or try to infer from content.

        test_input = "Test input for validation"
        start_time = time.time()

        try:
            # Use the optimized prompt in a real call
            # We use prompt_key to ensure variables are handled if we had them,
            # but since we just updated the registry, we can pass the key.
            # However, run_llm takes variables. We'll try to run with empty vars or basic ones.

            # Simple latency check
            response = await run_llm(
                prompt_key=prompt_key,
                prompt_vars={"input": test_input}, # Generic variable
                purpose="validation"
            )

            latency = (time.time() - start_time) * 1000

            if not response or not response.get("result"):
                log_event("Validation call failed (no result).", "WARNING")
                return False

            # Check latency against threshold
            if latency > self.latency_threshold_ms * 1.5: # Allow some buffer
                log_event(f"Validation latency high: {latency:.2f}ms", "WARNING")
                return False

            return True

        except Exception as e:
            log_event(f"Exception during validation: {e}", "ERROR")
            return False

    def _check_trigger(self):
        """
        Checks if the optimization cycle should be triggered.
        Integrates with monitoring manager if available.
        """
        # If we have a monitoring manager, we could check system load
        if self.monitoring_manager:
            # Placeholder: Check if system is under heavy load, if so, skip optimization
            pass

        # Basic check: only optimize if we have gathered some stats
        # (This avoids running on empty stats at startup)
        if os.path.exists(self.model_stats_file):
            return True

        return False
