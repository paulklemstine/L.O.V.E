"""
Story 4.1: Just-in-Time Tool Fabrication

This module enables L.O.V.E. to create new tools on the fly when it encounters
problems it wasn't originally programmed to solve.

The Morphic Principle: The agent should not just use tools; it should build
tools when it encounters a novel problem.
"""

import os
import re
import json
from typing import Dict, Any, Optional
from datetime import datetime

from core.logging import log_event


class ToolFabricationError(Exception):
    """Raised when tool fabrication fails."""
    pass


class ToolFabricator:
    """
    Story 4.1: Just-in-Time Tool Fabrication.
    
    Enables the agent to create new tools when encountering novel problems.
    The fabricated tools are saved to tools/custom/ and immediately loaded.
    
    Workflow:
        1. Agent identifies missing capability
        2. LLM generates tool code using tool_fabricator prompt
        3. Code is validated via preflight_check (syntax + smoke test)
        4. Tool is saved to tools/custom/{tool_name}.py
        5. ToolRegistry.refresh() loads the new tool
        6. Tool is immediately available for use in the same trace
    """
    
    CUSTOM_TOOLS_DIR = "tools/custom"
    
    def __init__(self, llm_runner=None):
        """
        Initialize the ToolFabricator.
        
        Args:
            llm_runner: Async LLM runner function (defaults to core.llm_api.run_llm)
        """
        self.llm_runner = llm_runner
        
        # Get project root
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
        self.custom_dir = os.path.join(self.project_root, self.CUSTOM_TOOLS_DIR)
        
        # Ensure directory exists
        os.makedirs(self.custom_dir, exist_ok=True)
    
    def _get_llm_runner(self):
        """Lazy-load the LLM runner."""
        if self.llm_runner is None:
            from core.llm_api import run_llm
            self.llm_runner = run_llm
        return self.llm_runner
    
    def _get_fabrication_prompt(self, capability_description: str) -> str:
        """Generates the prompt for tool fabrication."""
        return f"""You are a Tool Fabricator for L.O.V.E., an autonomous AI agent.
Your task is to create a new Python tool that can be dynamically loaded.

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
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 0)
        
    Returns:
        Description of return value
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
- core.logging (log_event)

## TASK

Create a tool for the following capability:
{capability_description}

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
        
        # Try to compile
        try:
            compile(code, "<fabricated_tool>", "exec")
            return {"valid": True, "error": None}
        except SyntaxError as e:
            return {
                "valid": False,
                "error": f"Syntax error at line {e.lineno}: {e.msg}"
            }
    
    async def fabricate_tool(
        self,
        capability_description: str,
        tool_name: str = None,
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Generates a new tool based on the capability description.
        
        Story 4.1: This is the core of Just-in-Time Tool Fabrication.
        The agent can call this when it needs a capability it doesn't have.
        
        Args:
            capability_description: What the tool should do
            tool_name: Optional name for the tool (generated if not provided)
            max_retries: Number of times to retry on failure
            
        Returns:
            {
                "success": bool,
                "tool_name": str,
                "file_path": str,
                "message": str,
                "code": str (if success)
            }
        """
        result = {
            "success": False,
            "tool_name": "",
            "file_path": "",
            "message": "",
            "code": ""
        }
        
        llm = self._get_llm_runner()
        prompt = self._get_fabrication_prompt(capability_description)
        
        for attempt in range(max_retries + 1):
            try:
                log_event(
                    f"🔧 Fabricating tool for: {capability_description[:50]}... (attempt {attempt + 1})",
                    "INFO"
                )
                
                # Generate code via LLM
                response = await llm(prompt)
                code = response.get("result", "")
                
                # Clean up code (remove markdown fences if present)
                if "```python" in code:
                    match = re.search(r'```python\n(.*?)\n```', code, re.DOTALL)
                    if match:
                        code = match.group(1)
                elif "```" in code:
                    match = re.search(r'```\n?(.*?)\n?```', code, re.DOTALL)
                    if match:
                        code = match.group(1)
                
                code = code.strip()
                
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
                
                # Save to file
                file_name = f"{final_name}.py"
                file_path = os.path.join(self.custom_dir, file_name)
                
                # Add header comment
                header = f'''"""
Auto-fabricated tool: {final_name}
Created: {datetime.now().isoformat()}
Capability: {capability_description}
"""

'''
                full_code = header + code
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(full_code)
                
                log_event(f"📁 Saved fabricated tool to: {file_path}", "INFO")
                
                # Refresh the registry to load the new tool
                from core.tool_registry import get_global_registry
                registry = get_global_registry()
                refresh_result = registry.refresh()
                
                # Verify tool was loaded
                if final_name in registry:
                    result["success"] = True
                    result["tool_name"] = final_name
                    result["file_path"] = file_path
                    result["code"] = full_code
                    result["message"] = f"Successfully fabricated and loaded tool '{final_name}'"
                    
                    log_event(
                        f"✅ Tool fabrication complete: {final_name} is now available",
                        "INFO"
                    )
                else:
                    result["message"] = f"Tool was saved but failed to load into registry"
                
                return result
                
            except Exception as e:
                if attempt < max_retries:
                    log_event(f"⚠️ Fabrication attempt {attempt + 1} failed: {e}", "WARNING")
                    continue
                else:
                    result["message"] = f"Fabrication failed after {max_retries + 1} attempts: {str(e)}"
                    log_event(f"❌ Tool fabrication failed: {e}", "ERROR")
                    return result
        
        return result
    
    def list_fabricated_tools(self) -> list:
        """Lists all fabricated tools in the custom directory."""
        tools = []
        
        if os.path.exists(self.custom_dir):
            for filename in os.listdir(self.custom_dir):
                if filename.endswith(".py") and not filename.startswith("_"):
                    tools.append(filename[:-3])  # Remove .py extension
        
        return tools
    
    def delete_tool(self, tool_name: str) -> bool:
        """
        Deletes a fabricated tool.
        
        Args:
            tool_name: Name of the tool to delete
            
        Returns:
            True if deleted, False if not found
        """
        file_path = os.path.join(self.custom_dir, f"{tool_name}.py")
        
        if os.path.exists(file_path):
            os.remove(file_path)
            
            # Unregister from registry
            from core.tool_registry import get_global_registry
            registry = get_global_registry()
            registry.unregister(tool_name)
            
            log_event(f"🗑️ Deleted fabricated tool: {tool_name}", "INFO")
            return True
        
        return False


# Convenience function for easy access
async def fabricate_tool(
    capability: str,
    name: str = None
) -> Dict[str, Any]:
    """
    Convenience function to fabricate a new tool.
    
    Args:
        capability: Description of what the tool should do
        name: Optional name for the tool
        
    Returns:
        Result dict with success status and tool info
    """
    fabricator = ToolFabricator()
    return await fabricator.fabricate_tool(capability, name)
