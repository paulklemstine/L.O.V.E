
import os
import json
import asyncio
from typing import List, Dict, Any, Optional

from core.prompt_registry import get_prompt_registry
from core.llm_api import run_llm
from core.logging import log_event
from langsmith import Client

class PollyOptimizer:
    """
    Polly: The Autonomous Prompt Optimizer.
    """
    
    def __init__(self, model_id: str = "gemini-2.0-flash-exp"):
        self.model_id = model_id
        self.registry = get_prompt_registry()
        
    async def optimize_prompt(self, prompt_key: str) -> Optional[str]:
        """
        Optimizes a single prompt by key.
        Returns the optimized prompt text, or None if failed.
        """
        log_event(f"Polly: Starting optimization for prompt '{prompt_key}'", "INFO")
        
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
            purpose="review",
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
             log_event(f"Polly: Optimization failed for '{prompt_key}'. No text returned. Falling back to original.", "WARNING")
             return current_prompt
             
        return improved_prompt

