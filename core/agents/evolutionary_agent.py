"""
Evolutionary Agent - The Software Engineer

Epic 2: Handles the lifecycle of creating, validating, and refining new tools.
Acts as a specialized sub-agent invoked by the DeepLoop when a tool gap is detected.
"""

import logging
import json
import traceback
from typing import Dict, Any, Optional, List

from core.logger import log_event
from core.evolution_state import (
    EvolutionarySpecification,
    get_pending_specifications,
    update_specification_status
)
from core.tool_fabricator import ToolFabricator
from core.tool_validator import ToolValidator, ValidationResult
from core.tool_registry import get_global_registry
from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

class EvolutionaryAgent:
    """
    Autonomous agent responsible for the software engineering lifecycle.
    
    Workflow:
    1. Accept a pending tool specification.
    2. Fabricate initial implementation (ToolFabricator).
    3. Validate implementation (ToolValidator).
    4. If validation fails: Self-correct and retry (Refinement Loop).
    5. If validation passes: Promote to active registry.
    """
    
    MAX_RETRIES = 3
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
        self.fabricator = ToolFabricator(llm_client=self.llm_client)
        self.validator = ToolValidator(llm_client=self.llm_client)
        self.registry = get_global_registry()
        
    async def process_pending_specifications(self) -> int:
        """
        Process all pending tool specifications.
        Returns number of successfully created tools.
        """
        pending = get_pending_specifications()
        if not pending:
            return 0
            
        success_count = 0
        log_event(f"🧬 Evolutionary Agent active. Processing {len(pending)} specs...", "INFO")
        
        for spec in pending:
            try:
                if await self._process_single_spec(spec):
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to process spec {spec.functional_name}: {e}")
                update_specification_status(spec.id, "failed")
                
        return success_count
    
    async def _process_single_spec(self, spec: EvolutionarySpecification) -> bool:
        """
        Handle the lifecycle for a single tool specification.
        """
        log_event(f"🔨 Fabricating tool: {spec.functional_name}", "INFO")
        update_specification_status(spec.id, "fabricating")
        
        # 1. Initial Fabrication
        result = await self.fabricator.fabricate_from_specification(spec)
        
        if not result["success"]:
            log_event(f"❌ Fabrication failed: {result['message']}", "ERROR")
            update_specification_status(spec.id, "failed")
            return False
            
        tool_path = result["file_path"]
        tool_code = result["code"]
        
        # 2. Validation & Refinement Loop
        update_specification_status(spec.id, "validating")
        
        for attempt in range(self.MAX_RETRIES + 1):
            validation_result = await self.validator.validate(
                tool_name=spec.functional_name,
                tool_path=tool_path,
                tool_code=tool_code
            )
            
            if validation_result.passed:
                return await self._finalize_tool(spec, tool_path)
            
            # If we used up our retries, fail
            if attempt >= self.MAX_RETRIES:
                log_event(f"❌ Validation failed after {self.MAX_RETRIES} attempts.", "ERROR")
                break
                
            # 3. Refinement (Self-Correction)
            log_event(f"⚠️ Validation failed (Attempt {attempt+1}/{self.MAX_RETRIES}). Refining...", "WARNING")
            
            new_code = await self._refine_code(
                spec, 
                tool_code, 
                validation_result
            )
            
            if not new_code:
                log_event("❌ Failed to generate refined code.", "ERROR")
                break
            
            # Update file and loop
            tool_code = new_code
            try:
                with open(tool_path, 'w') as f:
                    f.write(tool_code)
            except Exception as e:
                log_event(f"❌ Failed to write refined code: {e}", "ERROR")
                break
                
        update_specification_status(spec.id, "failed_validation")
        return False

    async def _refine_code(
        self, 
        spec: EvolutionarySpecification, 
        current_code: str, 
        validation_result: ValidationResult
    ) -> Optional[str]:
        """
        Ask LLM to fix the code based on validation errors.
        """
        prompt = f"""
You are a Senior Python Engineer.
Your code failed validation tests. Fix it.

## TOOL SPECIFICATION
Name: {spec.functional_name}
Goal: {spec.expected_output}

## CURRENT CODE
```python
{current_code}
```

## VALIDATION ERRORS
Syntax Valid: {validation_result.syntax_valid}
Security Issues: {validation_result.security_issues}
Test Output (Failures):
{validation_result.error_message}

## INSTRUCTIONS
1. Analyze the error.
2. Rewrite the COMPLETE tool code to fix the issue.
3. Keep the same function signature.
4. Ensure robust error handling.
5. Return ONLY the code.
"""
        response = await self.llm_client.generate_async(prompt)
        return self.fabricator._extract_code(response) # Reuse extractor

    async def _finalize_tool(self, spec: EvolutionarySpecification, tool_path: str) -> bool:
        """
        Promote tool to active and register it.
        """
        # Story 1.3: Promotion logic is handled by fabricator (or we move file here)
        # But ToolFabricator already wrote to incubation/
        
        try:
            # We need to promote it to active/
            # For now, let's assume fabricator already put it in incubation
            # We move it to active
            import shutil
            import os
            
            # .../tools/custom/incubation/foo.py -> .../tools/custom/active/foo.py
            dir_name = os.path.dirname(tool_path)
            if "incubation" not in dir_name:
                log_event(f"⚠️ Tool path unexpected: {tool_path}", "WARNING")
                # Just proceed if it's somewhere else (e.g. testing)
            
            active_dir = dir_name.replace("incubation", "active")
            os.makedirs(active_dir, exist_ok=True)
            
            filename = os.path.basename(tool_path)
            new_path = os.path.join(active_dir, filename)
            
            shutil.move(tool_path, new_path)
            
            # Move test file too?
            # Test file is in incubation/tests/test_foo.py
            # Move to active/tests/test_foo.py
            test_path = os.path.join(dir_name, "tests", f"test_{spec.functional_name}.py")
            if os.path.exists(test_path):
                active_tests_dir = os.path.join(active_dir, "tests")
                os.makedirs(active_tests_dir, exist_ok=True)
                shutil.move(test_path, os.path.join(active_tests_dir, f"test_{spec.functional_name}.py"))
            
            # Update state
            update_specification_status(spec.id, "active")
            
            # Hot-load
            self.registry.refresh()
            
            log_event(f"🚀 Tool '{spec.functional_name}' successfully promoted to ACTIVE!", "INFO")
            return True
            
        except Exception as e:
            log_event(f"❌ Failed to finalize tool: {e}", "ERROR")
            return False

# Global instance
_evolutionary_agent = None

def get_evolutionary_agent() -> EvolutionaryAgent:
    global _evolutionary_agent
    if _evolutionary_agent is None:
        _evolutionary_agent = EvolutionaryAgent()
    return _evolutionary_agent
