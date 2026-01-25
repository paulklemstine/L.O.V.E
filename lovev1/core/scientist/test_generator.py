
import os
import logging
import re
from typing import Optional
from core.llm_api import run_llm
from core.surgeon.ast_parser import extract_function_metadata

class TestGenerator:
    def __init__(self):
        # We rely on the global run_llm configuration, so no specific client init needed here for now.
        pass

    async def generate_test(self, file_path: str, function_name: str, output_path: str) -> bool:
        """
        Generates a standalone pytest file for a given function using an LLM.
        
        Args:
            file_path: Path to the source file containing the function.
            function_name: Name of the function to test.
            output_path: Path where the generated test file should be saved.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # 1. Extract metadata
            meta = extract_function_metadata(file_path, function_name)
            source_code = meta.get("source", "")
            if not source_code:
                logging.error(f"Could not extract source code for {function_name} in {file_path}")
                return False

            # 2. Construct Prompt
            # We need to inform the LLM about the context (imports?) or ask it to mock everything.
            # ideally we provide the imports from the file too, but for "Zero-Shot" on a single function,
            # we ask it to infer or mock.
            
            prompt = f"""
You are an expert Python QA engineer. Write a complete, standalone `pytest` test file for the following function.
The test file should:
1. Import the function correctly. Assume the source file is importable.
   (However, since this is a generated test, you might need to mock imports if dependencies are complex).
   Actually, assume the source file is at `{file_path}`.
   You should likely add the directory to sys.path if needed or treat it as a module.
   Better yet, assume standard import `from {os.path.basename(file_path).replace('.py', '')} import {function_name}` if in same dir,
   but since `output_path` might be elsewhere, handle imports robustly (e.g. sys.path.append).
2. Cover happy paths and edge cases.
3. Be syntactically correct Python.
4. Output ONLY the python code for the test file, inside a markdown code block ```python ... ```.

Target Function in `{file_path}`:
```python
{source_code}
```
"""

            # 3. Call LLM
            # purpose="coding" or "reasoning"? "coding" (if available) or default "general".
            response = await run_llm(prompt_text=prompt, purpose="general")
            
            if not response or not isinstance(response, dict) or not response.get("result"):
                logging.error(f"LLM returned invalid response: {response}")
                return False
            
            text_response = response["result"]

            # 4. Extract Code
            code = self._extract_code_block(text_response)
            if not code:
                logging.error("No code block found in LLM response")
                return False
                
            # 5. Write to file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(code)
                
            logging.info(f"Generated test saved to {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to generate test: {e}")
            return False

    def _extract_code_block(self, text: str) -> Optional[str]:
        """
        Extracts content from ```python ... ``` or just ``` ... ```
        """
        # Regex for python code block
        match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        # Fallback only if no python block
        match = re.search(r"```\n(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        return None
