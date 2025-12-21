
import os
import json
import asyncio
from typing import List, Dict, Any, Optional

from core.prompt_registry import get_prompt_registry
from core.llm_api import run_llm
from core.logging import log_event
from langsmith import Client

MAX_RECURSION_DEPTH = 2

class PollyOptimizer:
    """
    Polly: The Autonomous Prompt Optimizer.
    """
    
    def __init__(self, model_id: str = None):
        self.model_id = model_id
        self.registry = get_prompt_registry()
        self.golden_dataset = self._load_golden_dataset()
        
    def _load_golden_dataset(self) -> List[Dict[str, str]]:
        """Loads the Golden Dataset for evaluation."""
        try:
            dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'golden_dataset.json')
            if os.path.exists(dataset_path):
                with open(dataset_path, 'r') as f:
                    return json.load(f)
            else:
                log_event(f"Polly: Golden dataset not found at {dataset_path}", "WARNING")
                return []
        except Exception as e:
            log_event(f"Polly: Error loading golden dataset: {e}", "ERROR")
            return []

    async def evaluate_candidate_prompt(self, candidate_prompt: str) -> float:
        """
        Evaluates a candidate prompt against the Golden Dataset.
        Returns a score (0.0 to 10.0).
        """
        if not self.golden_dataset:
            log_event("Polly: No golden dataset available for evaluation. Skipping gate.", "WARNING")
            return 5.0 # Neutral score if no dataset

        total_score = 0
        valid_examples = 0

        for example in self.golden_dataset:
            # 1. Run the Task with Candidate Prompt
            input_text = example.get('input', '')
            criteria = example.get('criteria', '')

            if not input_text or not criteria:
                continue

            try:
                # Simulate the prompt usage
                # We assume the candidate_prompt is a System Prompt or similar instruction
                task_prompt = f"""
                {candidate_prompt}

                ---
                USER INPUT:
                {input_text}
                """
                
                # Use a fast model for the task execution if possible, or the main model
                # We want to see how the PROMPT performs.
                task_response = await run_llm(
                    prompt_text=task_prompt,
                    purpose="polly_evaluation_task",
                    force_model=self.model_id 
                )
                output_content = task_response.get('result', '')

                # 2. Judge the Result
                judge_prompt = f"""
                You are an impartial Judge AI. Evaluate the following AI Response based on the Criteria.
                
                Input: {input_text}
                
                AI Response:
                {output_content}
                
                Criteria:
                {criteria}
                
                Score the response from 1 to 10 (10 being perfect).
                Return ONLY the number.
                """
                
                judge_response = await run_llm(
                    prompt_text=judge_prompt,
                    purpose="polly_judge"
                    # force_model="gpt-4o-mini"  <-- REMOVED to prevent emergency mode. 
                    # The rank_models logic in llm_api.py will handle preference for smart/fast models.
                )
                
                score_text = judge_response.get('result', '').strip()
                # Extract number
                import re
                match = re.search(r'\d+(\.\d+)?', score_text)
                if match:
                    score = float(match.group())
                    # clamp to 10 just in case
                    score = min(max(score, 0), 10)
                    total_score += score
                    valid_examples += 1
                
            except Exception as e:
                log_event(f"Polly: Evaluation error on example {example.get('id')}: {e}", "ERROR")

        if valid_examples == 0:
            return 0.0
        
        return total_score / valid_examples

    async def optimize_prompt(self, prompt_key: str, recursion_depth: int = 0) -> Optional[str]:
        """
        Optimizes a single prompt by key.
        Returns the optimized prompt text, or None if failed.
        """
        if recursion_depth >= MAX_RECURSION_DEPTH:
            log_event(f"Polly: Recursion depth limit reached for '{prompt_key}'. Skipping.", "WARNING")
            return None

        log_event(f"Polly: Starting optimization for prompt '{prompt_key}' (Depth: {recursion_depth})", "INFO")
        
        # 1. Load Current Prompt
        current_prompt = self.registry.get_prompt(prompt_key)
        if not current_prompt:
            log_event(f"Polly: Prompt key '{prompt_key}' not found.", "WARNING")
            return None

        # 2. Load Dataset Examples (Gold Standard)
        try:
            client = Client()
            dataset_name = "gold-standard-behaviors"
            examples = []
            
            # Check availability of dataset
            if client.has_dataset(dataset_name=dataset_name):
                 ds_examples = client.list_examples(dataset_name=dataset_name, limit=5)
                 for ex in ds_examples:
                    examples.append({
                        "input": ex.inputs,
                        "output": ex.outputs
                    })
            else:
                log_event(f"Polly: Dataset '{dataset_name}' not found in LangSmith.", "INFO")

        except Exception as e:
            # Explicitly log this to help user debug
            log_event(f"Polly: LangSmith Client issue (check API Key): {e}", "WARNING")
            examples = []

        # 3. Load Self-Reflection Insights (Evolution State)
        insights = []
        try:
            state_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'love_state.json') # Use love_state.json in root
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    # Use generic 'insights' or extract from logs
                    insights = state.get('insights', [])
                    if not insights and 'evolution_log' in state:
                         # Grab last 3 summaries
                         insights = [entry.get('summary', '') for entry in state['evolution_log'][-3:]]
        except Exception as e:
            pass

        # 4. Construct Optimization Meta-Prompt
        optimizer_prompt = f"""
        You are an Expert Prompt Engineer AI (POLLY). Your task is to OPTIMIZE a given system prompt to improve the performance of an AI agent.

        ### Target Prompt ({prompt_key}):
        ```text
        {current_prompt}
        ```

        ### Goal:
        Improve the prompt to better handle edge cases, follow instructions more strictly, and align with the agent's persona.

        ### Gold Standard Examples (What the agent SHOULD do):
        {json.dumps(examples, indent=2) if examples else "No examples available yet."}

        ### Self-Reflection Insights (Recent learnings):
        {json.dumps(insights, indent=2) if insights else "No specific insights available."}

        ### Instructions:
        1. Analyze the Current Prompt, Examples, and Insights.
        2. Identify weaknesses or ambiguity in the current prompt.
        3. Rewrite the prompt to be more effective, concise, and robust.
        4. PRESERVE existing template variables.
        5. Output the FULL improved prompt text.
        6. DO NOT wrap with markdown code blocks if possible, just the raw text, OR ensure it's easy to extract.
        """

        # 5. Run Optimization
        response = await run_llm(
            prompt_text=optimizer_prompt,
            purpose="polly_optimizer",
            force_model=self.model_id
        )

        improved_prompt = (response.get('result') or '').strip()
        
        # Cleanup code blocks if present
        if improved_prompt.startswith("```"):
            # Remove first line
            improved_prompt = improved_prompt.split("\n", 1)[1]
            if improved_prompt.startswith("text") or improved_prompt.startswith("markdown"):
                 improved_prompt = improved_prompt.split("\n", 1)[1]
        
        if improved_prompt.endswith("```"):
            improved_prompt = improved_prompt.rsplit("\n", 1)[0]
            
        improved_prompt = improved_prompt.strip()

        if not improved_prompt:
             log_event(f"Polly: Optimization failed for '{prompt_key}'. No text returned.", "WARNING")
             return None
             
        # --- Gated Promotion Check ---
        log_event(f"Polly: Evaluating candidate prompt vs baseline...", "INFO")
        
        baseline_score = await self.evaluate_candidate_prompt(current_prompt)
        candidate_score = await self.evaluate_candidate_prompt(improved_prompt)
        
        log_event(f"Polly Evaluation - Baseline: {baseline_score}, Candidate: {candidate_score}", "INFO")
        
        if candidate_score > baseline_score:
            log_event(f"Polly: Candidate promoted! ({candidate_score} > {baseline_score})", "INFO")
            return improved_prompt
        else:
            log_event(f"Polly: Candidate rejected. ({candidate_score} <= {baseline_score})", "INFO")
            return None
