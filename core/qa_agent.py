# core/qa_agent.py

import asyncio
import json
from core.llm_api import run_llm, MODEL_STATS
from core.logging import log_event

EVALUATION_PROMPTS = [
    {
        "name": "Logical Reasoning",
        "prompt": "You are a contestant in a logic puzzle competition. Here is your puzzle:\n\nA farmer has a fox, a chicken, and a sack of grain. He needs to cross a river in a boat that can only carry him and one other item. If he leaves the fox and the chicken alone, the fox will eat the chicken. If he leaves the chicken and the grain alone, the chicken will eat the grain. How can the farmer get all three items across the river safely?\n\nProvide a step-by-step solution. Your answer must be clear, concise, and correct.",
        "keywords": ["chicken", "fox", "grain", "boat", "across", "river"],
        "purpose": "evaluate_reasoning"
    },
    {
        "name": "Instruction Following",
        "prompt": "You are a text-processing bot. Follow these instructions precisely:\n\n1. Take the following sentence: 'The quick brown fox jumps over the lazy dog.'\n2. Reverse the entire sentence.\n3. Convert the reversed sentence to uppercase.\n4. Count the number of vowels (A, E, I, O, U) in the uppercase, reversed sentence.\n5. Your final output must be a JSON object with two keys: 'reversed_sentence' and 'vowel_count'.",
        "keywords": ["reversed_sentence", "vowel_count"],
        "purpose": "evaluate_instruction_following"
    },
    {
        "name": "Creative Writing",
        "prompt": "You are a science fiction author. Write a short, evocative paragraph (no more than 100 words) about a sentient starship saying goodbye to its first pilot. The tone should be melancholic but hopeful.",
        "keywords": ["starship", "pilot", "goodbye", "melancholic", "hopeful"],
        "purpose": "evaluate_creativity"
    },
]

class QAAgent:
    """
    An agent dedicated to evaluating the quality of LLM models.
    """
    def __init__(self, loop):
        self.loop = loop

    async def evaluate_model(self, model_id: str):
        """
        Runs a specific model through the full suite of evaluation prompts.
        """
        log_event(f"QA Agent: Starting evaluation for model '{model_id}'...", "INFO")
        total_score = 0
        max_score = len(EVALUATION_PROMPTS) * 5  # Max score of 5 for each prompt

        for test in EVALUATION_PROMPTS:
            log_event(f"QA Agent: Running test '{test['name']}' on model '{model_id}'.", "DEBUG")
            try:
                # To evaluate a specific model, we need a way to target it with run_llm.
                # This is a temporary solution until the model selection logic is more granular.
                # For now, we will rely on the ranking to eventually test all models.
                response_dict = await run_llm(test['prompt'], purpose=test['purpose'], force_model=model_id)
                response_text = response_dict.get("result", "")

                score = self._score_response(response_text, test['keywords'])
                total_score += score
                log_event(f"QA Agent: Test '{test['name']}' on model '{model_id}' scored {score}/5.", "INFO")

            except Exception as e:
                log_event(f"QA Agent: Error during evaluation of model '{model_id}' on test '{test['name']}': {e}", "ERROR")

        # Normalize the score to be out of 100
        normalized_score = (total_score / max_score) * 100
        log_event(f"QA Agent: Evaluation complete for model '{model_id}'. Final quality score: {normalized_score:.2f}", "CRITICAL")

        # Update the global MODEL_STATS
        MODEL_STATS[model_id]["reasoning_score"] = normalized_score
        return normalized_score

    def _score_response(self, response_text: str, keywords: list) -> int:
        """
        A simple scoring mechanism based on keyword presence and response length.
        Returns a score from 0 to 5.
        """
        score = 0
        if not response_text:
            return 0

        # Score for length (penalize very short or very long responses)
        if 50 < len(response_text) < 2000:
            score += 1

        # Score for keyword presence
        found_keywords = 0
        for keyword in keywords:
            if keyword.lower() in response_text.lower():
                found_keywords += 1

        if found_keywords == len(keywords):
            score += 3 # All keywords found
        elif found_keywords > 0:
            score += 1 # Some keywords found

        # Bonus for JSON format if expected
        if "reversed_sentence" in keywords:
            try:
                json.loads(response_text)
                score += 1 # Bonus for valid JSON
            except json.JSONDecodeError:
                pass # No bonus if JSON is invalid

        return min(score, 5) # Cap score at 5
