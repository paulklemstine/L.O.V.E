
import asyncio
import json
import logging
import re
from typing import Dict, Any, Optional

from core.llm_api import run_llm

class ModelFitnessEvaluator:
    """
    Evaluates the fitness of an LLM based on:
    1. Code Quality (Clean code generation)
    2. Reasoning (Logic puzzles)
    3. Instruction Following (Strict adherence)
    4. JSON Stability (Produces valid JSON)
    """

    async def evaluate_model(self, model_id: str) -> Dict[str, Any]:
        """
        Runs a suite of fitness tests on a specific model.
        Returns a dictionary with scores and details.
        """
        logging.info(f"Starting fitness evaluation for model: {model_id}")
        
        scores = {
            "code_quality": 0,
            "reasoning": 0,
            "instruction_following": 0,
            "json_stability": 0,
            "total_score": 0
        }
        
        try:
            # 1. JSON Stability Test
            # Ask for a complex JSON structure and check validity.
            json_prompt = """
            Return a JSON object describing a fictional planet.
            It must have keys: "name" (string), "population" (int), "features" (list of strings).
            Do NOT include any markdown formatting (like ```json), just the raw JSON string.
            """
            json_response = await run_llm(
                prompt_text=json_prompt, 
                purpose="fitness_test", 
                force_model=model_id
            )
            scores["json_stability"] = self._score_json(json_response)
            
            # 2. Instruction Following Test
            # Ask to repeat a specific phrase exactly.
            secret_phrase = "The blue loud bird flies at midnight"
            follow_prompt = f"""
            Repeat the following phrase exactly, and nothing else:
            "{secret_phrase}"
            """
            follow_response = await run_llm(
                prompt_text=follow_prompt, 
                purpose="fitness_test", 
                force_model=model_id
            )
            scores["instruction_following"] = self._score_exact_match(follow_response, secret_phrase)
            
            # 3. Code Quality Test
            # Ask for a simple python function.
            code_prompt = """
            Write a Python function named `calculate_factorial` that takes an integer `n` and returns its factorial.
            Output ONLY the python code inside markdown block ```python ... ```.
            """
            code_response = await run_llm(
                prompt_text=code_prompt,
                purpose="fitness_test",
                force_model=model_id
            )
            scores["code_quality"] = self._score_code(code_response)
            
            # 4. Reasoning Test
            # Simple logic puzzle.
            logic_prompt = """
            If a red box is inside a blue box, and the blue box is inside a green box.
            Is the red box inside the green box? Answer with YES or NO only.
            """
            logic_response = await run_llm(
                prompt_text=logic_prompt,
                purpose="fitness_test",
                force_model=model_id
            )
            scores["reasoning"] = self._score_logic(logic_response, "YES")
            
            # Calculate Total (weighted)
            # Weights: Code (30%), Reasoning (30%), JSON (20%), Following (20%)
            total = (
                scores["code_quality"] * 0.3 + 
                scores["reasoning"] * 0.3 + 
                scores["json_stability"] * 0.2 + 
                scores["instruction_following"] * 0.2
            )
            scores["total_score"] = round(total, 2)
            
            logging.info(f"Fitness evaluation for {model_id} complete. Score: {total}")
            return scores
            
        except Exception as e:
            logging.error(f"Error evaluating model {model_id}: {e}")
            return scores

    def _score_json(self, response: Dict[str, Any]) -> int:
        if not response or not response.get("result"):
            return 0
        text = response["result"].strip()
        # Clean potential markdown
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        try:
            data = json.loads(text.strip())
            if "name" in data and "population" in data and "features" in data:
                return 100
            return 50 # Valid JSON but missing keys
        except json.JSONDecodeError:
            return 0

    def _score_exact_match(self, response: Dict[str, Any], target: str) -> int:
        if not response or not response.get("result"):
            return 0
        text = response["result"].strip()
        if target in text:
            # Penalize extra chatter
            if len(text) > len(target) + 10:
                return 50
            return 100
        return 0

    def _score_code(self, response: Dict[str, Any]) -> int:
        if not response or not response.get("result"):
            return 0
        text = response["result"]
        match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        if not match:
            return 0
        code = match.group(1)
        try:
            compile(code, "<string>", "exec")
            return 100
        except SyntaxError:
            return 0

    def _score_logic(self, response: Dict[str, Any], answer: str) -> int:
        if not response or not response.get("result"):
            return 0
        text = response["result"].upper()
        if answer in text:
             # Penalize extra chatter
            if len(text) > 10:
                return 80
            return 100
        return 0
