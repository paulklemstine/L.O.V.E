
import asyncio
from core.model_fitness import ModelFitnessEvaluator
from core.prompt_engineer import PromptEngineer

async def verify():
    fit = ModelFitnessEvaluator()
    eng = PromptEngineer()
    print("Classes instantiated successfully.")
    
    # Optional: Mock run for prompt engineer if we want to confirm regex logic (mocking run_llm)
    # But import success is good enough for structure check.

if __name__ == "__main__":
    asyncio.run(verify())
