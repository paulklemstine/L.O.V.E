
import json
import asyncio
import traceback
from typing import Dict, Any, Optional


class ModelQAManager:
    """
    Manages Quality Assurance checks for LLM models.
    Tests models for their ability to follow strict JSON instructions.
    """
    def __init__(self):
        self.qa_scores: Dict[str, float] = {}
        self.tested_models = set()

    async def test_model_capability(self, model_id: str) -> float:
        """
        Runs a standard QA test on the model to determine its reasoning and formatting capability.
        Returns a score from 0.0 to 1.0.
        """
        if model_id in self.qa_scores:
            return self.qa_scores[model_id]

        print(f"--- Running QA Test for Model: {model_id} ---")
        
        test_prompt = """
        SYSTEM_TEST_PROTOCOL_INITIATED
        
        TASK: Output a valid JSON object with specific data.
        
        REQUIRED JSON STRUCTURE:
        {
            "status": "operational",
            "test_code": 9988,
            "message": "QA Verified"
        }
        
        Output ONLY the JSON object. No markdown, no preambles.
        """
        
        try:
            # Run with a short timeout and low temp
            from core.llm_api import run_llm
            response = await run_llm(
                test_prompt, 
                purpose="scoring", 
                force_model=model_id, 
                temperature=0.1,
                allow_fallback=False # Must test THIS model
            )
            
            raw_text = response.get("result", "")
            
            # Scoring Logic
            score = 0.0
            
            # 1. JSON Validity
            try:
                # Clean up markdown
                clean_text = raw_text.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_text)
                
                # Check formatting
                if isinstance(data, dict):
                    score += 0.4
                    
                    # 2. Content Accuracy
                    if data.get("status") == "operational":
                        score += 0.2
                    if data.get("test_code") == 9988:
                        score += 0.2
                    if data.get("message") == "QA Verified":
                        score += 0.2
                        
                else:
                    print(f"Model {model_id} returned valid JSON but not a dict: {type(data)}")
                    score = 0.2 # Partial credit for valid JSON
                    
            except json.JSONDecodeError:
                print(f"Model {model_id} failed JSON parsing. Output: {raw_text[:50]}...")
                score = 0.0
                
            print(f"Model {model_id} QA Score: {score:.2f}")
            self.qa_scores[model_id] = score
            self.tested_models.add(model_id)
            return score
            
        except Exception as e:
            print(f"Model QA System Error for {model_id}: {e}")
            self.qa_scores[model_id] = 0.0
            return 0.0

# Singleton instance
qa_manager = ModelQAManager()
