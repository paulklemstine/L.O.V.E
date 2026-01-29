"""
Tool Fabricator - Just-in-Time Tool Generation

Epic 1, Story 1.2: Enables L.O.V.E. to create new tools dynamically when 
encountering problems that existing tools cannot address.

The Morphic Principle: The agent should not just use tools; it should build
tools when it encounters a novel problem.

Workflow:
    1. Agent identifies missing capability (via ToolGapDetector or direct request)
    2. LLM generates tool code following @tool_schema pattern
    3. Code is validated (syntax check)
    4. Tool is saved to tools/custom/incubation/{tool_name}.py
    5. After validation passes, tool is promoted to tools/custom/active/
    6. ToolRegistry hot-loads the new tool
"""

import os
import re
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field, asdict

import logging
from core.logger import log_event
from core.evolution_state import EvolutionarySpecification

logger = logging.getLogger(__name__)


class ToolFabricationError(Exception):
    """Raised when tool fabrication fails."""
    pass


# EvolutionarySpecification imported from core.evolution_state


class ToolFabricator:
    """
    Just-in-Time Tool Fabrication.
    
    Generates new Python tools based on capability descriptions or 
    EvolutionarySpecifications. Tools are written to an incubation
    directory first, then promoted to active after validation.
    """
    
    INCUBATION_DIR = "tools/custom/incubation"
    ACTIVE_DIR = "tools/custom/active"
    
    def __init__(self, llm_client=None):
        """
        Initialize the ToolFabricator.
        
        Args:
            llm_client: LLM client with generate() method. If None, will
                       attempt to use core.llm_client.LLMClient
        """
        self.llm_client = llm_client
        
        # Get project root (parent of core/)
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        self.incubation_dir = os.path.join(self.project_root, self.INCUBATION_DIR)
        self.active_dir = os.path.join(self.project_root, self.ACTIVE_DIR)
        
        # Ensure directories exist
        os.makedirs(self.incubation_dir, exist_ok=True)
        os.makedirs(self.active_dir, exist_ok=True)
    
    def _get_llm_client(self):
        """Lazy-load the LLM client."""
        if self.llm_client is None:
            try:
                from core.llm_client import LLMClient
                self.llm_client = LLMClient()
            except ImportError:
                raise ToolFabricationError(
                    "No LLM client available. Provide one to __init__ or install core.llm_client"
                )
        return self.llm_client
    
    def _get_fabrication_prompt(
        self, 
        capability_description: str,
        spec: Optional[EvolutionarySpecification] = None
    ) -> str:
        """Generates the prompt for tool fabrication."""
        
        # Build argument specification if we have a spec
        arg_spec = ""
        if spec and spec.required_arguments:
            args_list = [f"  - {name}: {typ}" for name, typ in spec.required_arguments.items()]
            arg_spec = f"""
## Required Arguments
{chr(10).join(args_list)}

## Expected Output
{spec.expected_output}

## Safety Constraints
{chr(10).join(f'- {c}' for c in spec.safety_constraints) if spec.safety_constraints else '- None specified'}
"""
        
        return f"""You are a Tool Fabricator for L.O.V.E., an autonomous AI agent.
Your task is to create a new Python tool that can be dynamically loaded.

## CAPABILITY TO IMPLEMENT
{capability_description}
{arg_spec}

## REQUIREMENTS

The tool MUST:
1. Be a single, self-contained Python function
2. Have a descriptive docstring with Args and Returns sections
3. Have complete type hints for all parameters
4. Be decorated with @tool_schema from core.tool_registry
5. Handle errors gracefully with try/except
6. NOT import external dependencies not already in the project
7. Return a meaningful result (not just None)

## TEMPLATE

```python
from core.tool_registry import tool_schema


@tool_schema
def tool_name(param1: str, param2: int = 0) -> str:
    '''
    Brief description of what this tool does.
    
    This tool is designed to help L.O.V.E. accomplish specific tasks
    that require this capability.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 0)
        
    Returns:
        Description of return value
        
    Example:
        result = tool_name("example", 42)
    '''
    try:
        # Implementation here
        result = f"Processed {{param1}} with value {{param2}}"
        return result
    except Exception as e:
        return f"Error: {{str(e)}}"
```

## ALLOWED IMPORTS (use these if needed)

- os, sys, json, re, datetime, time
- urllib, http, hashlib
- typing (Dict, List, Optional, Any)
- core.tool_registry (tool_schema)
- core.logger (get_logger)

## OUTPUT

Return ONLY the Python code, no explanation, no markdown fences.
The code must be immediately executable when saved to a .py file.
"""
    
    def _extract_tool_name(self, code: str) -> Optional[str]:
        """Extracts the tool name from the generated code."""
        # Look for @tool_schema decorated function
        match = re.search(r'@tool_schema\s*\n\s*def\s+(\w+)\s*\(', code)
        if match:
            return match.group(1)
        
        # Fallback: look for any function definition
        match = re.search(r'def\s+(\w+)\s*\(', code)
        if match:
            return match.group(1)
        
        return None
    
    def _sanitize_tool_name(self, name: str) -> str:
        """Converts a description to a valid Python function name."""
        # Convert to lowercase and replace spaces/hyphens with underscores
        sanitized = name.lower().replace(" ", "_").replace("-", "_")
        # Remove non-alphanumeric characters except underscores
        sanitized = re.sub(r'[^a-z0-9_]', '', sanitized)
        # Ensure doesn't start with number
        if sanitized and sanitized[0].isdigit():
            sanitized = "tool_" + sanitized
        # Truncate if too long
        if len(sanitized) > 50:
            sanitized = sanitized[:50]
        return sanitized or "custom_tool"
    
    def _validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validates the generated code using compile and basic checks.
        
        Returns:
            {"valid": bool, "error": str or None}
        """
        # Basic checks
        if not code.strip():
            return {"valid": False, "error": "Empty code"}
        
        if "@tool_schema" not in code:
            return {"valid": False, "error": "Missing @tool_schema decorator"}
        
        if "def " not in code:
            return {"valid": False, "error": "No function definition found"}
        
        # Check for docstring
        if '"""' not in code and "'''" not in code:
            return {"valid": False, "error": "Missing docstring"}
        
        # Try to compile
        try:
            compile(code, "<fabricated_tool>", "exec")
            return {"valid": True, "error": None}
        except SyntaxError as e:
            return {
                "valid": False,
                "error": f"Syntax error at line {e.lineno}: {e.msg}"
            }
    
    def _clean_code(self, code: str) -> str:
        """Clean up LLM-generated code (remove markdown fences, etc)."""
        code = code.strip()
        
        # Remove markdown fences
        if "```python" in code:
            match = re.search(r'```python\n(.*?)\n```', code, re.DOTALL)
            if match:
                code = match.group(1)
        elif "```" in code:
            match = re.search(r'```\n?(.*?)\n?```', code, re.DOTALL)
            if match:
                code = match.group(1)
        
        return code.strip()
    
    async def fabricate_from_specification(
        self, 
        spec: EvolutionarySpecification,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Generate tool code from an evolutionary specification.
        
        Story 1.2: This is the specification-driven fabrication path,
        used when a ToolGapDetector has identified a missing capability.
        
        Args:
            spec: EvolutionarySpecification defining what to build
            max_retries: Number of times to retry on failure
            
        Returns:
            Dict with success, tool_name, file_path, message, code keys
        """
        # Build capability description from spec
        capability = f"{spec.functional_name}: {spec.trigger_context}"
        
        return await self.fabricate_tool(
            capability_description=capability,
            tool_name=spec.functional_name,
            max_retries=max_retries,
            spec=spec
        )
    
    async def fabricate_tool(
        self,
        capability_description: str,
        tool_name: str = None,
        max_retries: int = 2,
        spec: Optional[EvolutionarySpecification] = None
    ) -> Dict[str, Any]:
        """
        Generates a new tool based on the capability description.
        
        Story 1.2: This is the core of Just-in-Time Tool Fabrication.
        The agent can call this when it needs a capability it doesn't have.
        
        Args:
            capability_description: What the tool should do
            tool_name: Optional name for the tool (generated if not provided)
            max_retries: Number of times to retry on failure
            spec: Optional EvolutionarySpecification with detailed requirements
            
        Returns:
            {
                "success": bool,
                "tool_name": str,
                "file_path": str,
                "message": str,
                "code": str (if success),
                "location": "incubation" | "active"
            }
        """
        result = {
            "success": False,
            "tool_name": "",
            "file_path": "",
            "message": "",
            "code": "",
            "location": "incubation"
        }
        
        llm = self._get_llm_client()
        prompt = self._get_fabrication_prompt(capability_description, spec)
        
        for attempt in range(max_retries + 1):
            try:
                logger.info(
                    f"🔧 Fabricating tool for: {capability_description[:50]}... (attempt {attempt + 1})"
                )
                
                # Generate code via LLM
                response = await llm.generate(prompt)
                code = self._clean_code(response)
                
                # Validate the code
                validation = self._validate_code(code)
                if not validation["valid"]:
                    if attempt < max_retries:
                        # Add error context to prompt for retry
                        prompt += f"\n\n## PREVIOUS ERROR\n{validation['error']}\nPlease fix this issue."
                        continue
                    else:
                        result["message"] = f"Validation failed after {max_retries + 1} attempts: {validation['error']}"
                        return result
                
                # Extract or generate tool name
                extracted_name = self._extract_tool_name(code)
                final_name = tool_name or extracted_name or self._sanitize_tool_name(
                    capability_description[:30]
                )
                
                # Save to INCUBATION directory (not active yet)
                file_name = f"{final_name}.py"
                file_path = os.path.join(self.incubation_dir, file_name)
                
                # Add header comment
                header = f'''"""
Auto-fabricated tool: {final_name}
Created: {datetime.now().isoformat()}
Capability: {capability_description}
Status: INCUBATION - Pending validation and L.O.V.E. approval
"""

'''
                full_code = header + code
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(full_code)
                
                logger.info(f"📁 Saved fabricated tool to incubation: {file_path}")
                
                result["success"] = True
                result["tool_name"] = final_name
                result["file_path"] = file_path
                result["code"] = full_code
                result["location"] = "incubation"
                result["message"] = f"Successfully fabricated tool '{final_name}' (pending validation)"
                
                logger.info(
                    f"✅ Tool fabrication complete: {final_name} awaiting validation"
                )
                
                return result
                
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"⚠️ Fabrication attempt {attempt + 1} failed: {e}")
                    continue
                else:
                    result["message"] = f"Fabrication failed after {max_retries + 1} attempts: {str(e)}"
                    logger.error(f"❌ Tool fabrication failed: {e}")
                    return result
        
        return result
    
    def promote_tool(self, tool_name: str) -> bool:
        """
        Promote a tool from incubation to active directory.
        
        This should be called after L.O.V.E. has validated and approved
        the tool for use.
        
        Args:
            tool_name: Name of the tool to promote
            
        Returns:
            True if promoted successfully, False otherwise
        """
        import shutil
        
        incubation_path = os.path.join(self.incubation_dir, f"{tool_name}.py")
        active_path = os.path.join(self.active_dir, f"{tool_name}.py")
        
        if not os.path.exists(incubation_path):
            logger.error(f"Tool not found in incubation: {tool_name}")
            return False
        
        try:
            # Read, update header, and write to active
            with open(incubation_path, 'r') as f:
                code = f.read()
            
            # Update status in header
            code = code.replace(
                "Status: INCUBATION - Pending validation and L.O.V.E. approval",
                f"Status: ACTIVE - Promoted at {datetime.now().isoformat()}"
            )
            
            with open(active_path, 'w') as f:
                f.write(code)
            
            # Remove from incubation
            os.remove(incubation_path)
            
            logger.info(f"🎓 Tool promoted to active: {tool_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote tool {tool_name}: {e}")
            return False
    
    def list_incubation_tools(self) -> List[str]:
        """Lists all tools in the incubation directory."""
        tools = []
        if os.path.exists(self.incubation_dir):
            for filename in os.listdir(self.incubation_dir):
                if filename.endswith(".py") and not filename.startswith("_"):
                    tools.append(filename[:-3])
        return tools
    
    def list_active_tools(self) -> List[str]:
        """Lists all tools in the active directory."""
        tools = []
        if os.path.exists(self.active_dir):
            for filename in os.listdir(self.active_dir):
                if filename.endswith(".py") and not filename.startswith("_"):
                    tools.append(filename[:-3])
        return tools
    
    def delete_tool(self, tool_name: str, location: str = "incubation") -> bool:
        """
        Deletes a fabricated tool.
        
        Args:
            tool_name: Name of the tool to delete
            location: "incubation" or "active"
            
        Returns:
            True if deleted, False if not found
        """
        if location == "incubation":
            file_path = os.path.join(self.incubation_dir, f"{tool_name}.py")
        else:
            file_path = os.path.join(self.active_dir, f"{tool_name}.py")
        
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"🗑️ Deleted tool: {tool_name} from {location}")
            return True
        
        return False


# Convenience function for easy access
async def fabricate_tool(
    capability: str,
    name: str = None,
    llm_client = None
) -> Dict[str, Any]:
    """
    Convenience function to fabricate a new tool.
    
    Args:
        capability: Description of what the tool should do
        name: Optional name for the tool
        llm_client: Optional LLM client
        
    Returns:
        Result dict with success status and tool info
    """
    fabricator = ToolFabricator(llm_client=llm_client)
    return await fabricator.fabricate_tool(capability, name)
