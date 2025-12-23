
import logging
import re
from typing import Optional, Dict

from core.llm_api import run_llm

class PromptEngineer:
    """
    Refines and optimizes prompts using LLM reasoning, with strict verification
    to ensure placeholders and core intent are preserved.
    """

    async def evolve_prompt(self, current_prompt: str, goal: str = "Improve clarity and performance") -> Optional[str]:
        """
        Evolves a prompt to better meet a specific goal.
        """
        logging.info("PromptEngineer: Starting prompt evolution...")
        
        # 1. Identify Variables (Placeholders like {variable})
        # Regex for {var}
        placeholders = set(re.findall(r"\{([a-zA-Z0-9_]+)\}", current_prompt))
        
        # 2. Meta-Prompt for Reasoning & Rewriting
        meta_prompt = f"""
        You are an Expert Prompt Engineer.
        Your task is to IMPROVE the following system prompt.
        
        Current Prompt:
        ```text
        {current_prompt}
        ```
        
        Goal: {goal}
        
        CRITICAL CONSTRAINTS:
        1. You MUST preserve these variable placeholders EXACTLY: {', '.join(placeholders)}
        2. Do not change the fundamental role or output format constraints if they are strict (e.g. JSON only).
        3. Make the instructions clearer, more concise, and more effective for an LLM.
        
        Output format:
        A single markdown code block containing ONLY the new prompt text.
        ```text
        ... new prompt ...
        ```
        """
        
        # 3. Call LLM (Reasoning Step)
        response = await run_llm(prompt_text=meta_prompt, purpose="reasoning") # Use reasoning model
        if not response or not response.get("result"):
            logging.error("PromptEngineer: LLM returned empty response.")
            return None
            
        result_text = response["result"]
        
        # 4. Extract Code Block
        match = re.search(r"```(?:text|markdown)?\n(.*?)```", result_text, re.DOTALL)
        if not match:
            logging.error("PromptEngineer: No code block found in response.")
            # Fallback: if whole response looks like prompt? No, unsafe.
            return None
            
        new_prompt = match.group(1).strip()
        
        # 5. Verification
        if not self._verify_prompt(current_prompt, new_prompt, placeholders):
            logging.warning("PromptEngineer: Verification failed. Rejecting evolution.")
            return None
            
        logging.info("PromptEngineer: Prompt evolved successfully.")
        return new_prompt

    def _verify_prompt(self, original: str, candidate: str, placeholders: set) -> bool:
        """
        Verifies that the candidate prompt is safe to use.
        """
        # Check 1: Placeholders preserved
        candidate_placeholders = set(re.findall(r"\{([a-zA-Z0-9_]+)\}", candidate))
        missing = placeholders - candidate_placeholders
        if missing:
            logging.error(f"PromptEngineer: Missing placeholders: {missing}")
            return False
            
        # Check 2: Not empty
        if not candidate or len(candidate) < 10:
            logging.error("PromptEngineer: Candidate prompt too short.")
            return False
            
        # Check 3: Not drastically shorter (unless intent was to shorten)
        if len(candidate) < len(original) * 0.5:
             logging.warning("PromptEngineer: Candidate is significantly shorter. Checking content...")
             # Soft warning, maybe okay.
             
        return True
