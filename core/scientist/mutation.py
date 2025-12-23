
import logging
import asyncio
import re
import libcst as cst
from typing import Optional
from core.llm_api import run_llm
from core.surgeon.ast_parser import extract_function_metadata

class MutationEngine:
    async def evolve_function(self, file_path: str, function_name: str, goal: str) -> Optional[str]:
        """
        Proposes a new implementation for a function based on a goal.
        Ensures the signature remains exactly the same.
        """
        try:
            # 1. Extract metadata
            meta = extract_function_metadata(file_path, function_name)
            original_code = meta.get("source", "")
            if not original_code:
                logging.error(f"Could not extract source for {function_name}")
                return None
                
            original_signature = self._extract_signature(original_code)
            
            # 2. Prompt
            prompt = f"""
You are an expert Python developer. Your task is to rewrite the following function to achieve this goal:
"{goal}"

CRITICAL RULES:
1. You MUST preserve the exact function signature (name, arguments, return type annotation).
2. You MUST preserve the functionality unless the goal is to change it (but signature must stay).
3. Output ONLY the new function code inside a markdown code block ```python ... ```.
4. Do not include any other text.

Current Code:
```python
{original_code}
```
"""
            # 3. Call LLM
            response = await run_llm(prompt_text=prompt, purpose="coding")
            if not response or not isinstance(response, dict) or not response.get("result"):
                logging.error(f"Empty or invalid response from LLM: {response}")
                return None
                
            # 4. Extract Code
            new_code = self._extract_code_block(response["result"])
            if not new_code:
                logging.error("No code block found in response")
                return None
            
            # 5. Verify Signature
            new_signature = self._extract_signature(new_code)
            
            # We compare normalized signatures or specific parts
            # Let's compare the def line name and args. return annotation might be tricky if formatting changes.
            # Using LibCST to parse and inspect is safer.
            
            if not self._signatures_match(original_code, new_code):
                logging.warning(f"Signature mismatch! Original: {original_signature} vs New: {new_signature}")
                # We could try to retry or just fail. For now, fail safe.
                return None
                
            return new_code
            
        except Exception as e:
            logging.error(f"Mutation failed: {e}")
            return None

    def _extract_code_block(self, text: str) -> Optional[str]:
        match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
        if match: return match.group(1).strip()
        match = re.search(r"```\n(.*?)```", text, re.DOTALL)
        if match: return match.group(1).strip()
        return None

    def _extract_signature(self, code: str) -> str:
        """
        Returns a normalized signature string for comparison.
        """
        try:
            module = cst.parse_module(code)
            # Find the function def
            func_def = None
            for node in module.body:
                if isinstance(node, cst.FunctionDef):
                    func_def = node
                    break
            if not func_def:
                return "nofunc"
            
            # Reconstruct signature part: defname(params) -> ret:
            # We can use code_for_node on specific parts or just name + params
            # Standardizing whitespace is key.
            # Using cst.Module([]).code_for_node(func_def.params)
            
            # Or just verify name and param names/order?
            # User requirement: "Signature preservation is verified."
            
            name = func_def.name.value
            params = []
            for param in func_def.params.params:
                p_name = param.name.value
                p_anno = ""
                if param.annotation:
                    # Normalized annotation code
                    p_anno = cst.Module([]).code_for_node(param.annotation.annotation).strip()
                params.append(f"{p_name}:{p_anno}")
                
            ret_anno = ""
            if func_def.returns:
                ret_anno = cst.Module([]).code_for_node(func_def.returns.annotation).strip()
                
            return f"{name}({','.join(params)})->{ret_anno}"
            
        except Exception:
            return "error"

    def _signatures_match(self, code1: str, code2: str) -> bool:
        sig1 = self._extract_signature(code1)
        sig2 = self._extract_signature(code2)
        # We might want to be loose on whitespace in annotation?
        # My _extract_signature strips some things, but let's compare normalized strings.
        # Remove whitespace checks
        return sig1.replace(" ", "") == sig2.replace(" ", "")
