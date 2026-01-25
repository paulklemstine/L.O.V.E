
import argparse
import sys
import os
from pydantic import BaseModel, Field
from typing import List, Optional

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.llm_api import run_llm

async def generate_schema_code(description: str) -> str:
    """
    Generates Python code for a Pydantic model based on a literal description.
    """
    prompt = f"""
    You are an expert Python developer specialized in Pydantic.
    Your task is to generate a Pydantic `BaseModel` class definition based on the following description.
    
    Description: "{description}"
    
    Requirements:
    1. Import `BaseModel`, `Field` from `pydantic` and `typing` imports.
    2. Name the class `ExtractedData` (unless the description implies a specific entity name, then use that).
    3. Include docstrings for the class and fields.
    4. Return ONLY the valid Python code. No markdown code blocks.
    
    Example input: "A list of users with name and age"
    Example output:
    from pydantic import BaseModel, Field
    from typing import List
    
    class User(BaseModel):
        name: str = Field(..., description="The name of the user")
        age: int = Field(..., description="The age of the user")
        
    class ExtractedData(BaseModel):
        users: List[User] = Field(..., description="List of extracted users")
    """
    
    response = await run_llm(prompt_text=prompt, is_source_code=True, purpose="general") # Using general purpose for code gen
    code = response.get('result', '').strip()
    
    # Clean markdown if present
    if code.startswith("```python"):
        code = code.replace("```python", "").replace("```", "").strip()
    elif code.startswith("```"):
        code = code.replace("```", "").strip()
        
    return code

if __name__ == "__main__":
    import asyncio
    parser = argparse.ArgumentParser(description="Generate Pydantic schema from description.")
    parser.add_argument("description", help="Description of the data structure to generate.")
    args = parser.parse_args()
    
    code = asyncio.run(generate_schema_code(args.description))
    print(code)
