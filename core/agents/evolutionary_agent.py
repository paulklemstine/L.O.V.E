"""
Evolutionary Agent - The Software Engineer

Epic 2: Handles the lifecycle of creating, validating, and refining new tools.
Acts as a specialized sub-agent invoked by the PiLoop when a tool gap is detected.
"""

import os
import shutil
import logging
import json
import traceback
from typing import Dict, Any, Optional, List
from datetime import datetime

from core.logger import log_event
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
from core.state_manager import get_state_manager

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
            get_state_manager().update_agent_status("EvolutionaryAgent", "Idle", action="No specs")
            return 0
            
        get_state_manager().update_agent_status(
            "EvolutionaryAgent", 
            "Active",
            action=f"Found {len(pending)} specs",
            subtasks=[s.functional_name for s in pending]
        )
            
        success_count = 0
        log_event(f"🧬 Evolutionary Agent active. Processing {len(pending)} specs...", "INFO")
        
        for spec in pending:
            try:
                if await self._process_single_spec(spec):
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to process spec {spec.functional_name}: {e}")
                update_specification_status(spec.id, "failed")
                
        get_state_manager().update_agent_status("EvolutionaryAgent", "Idle", action="Cycle complete")
        return success_count
    
    async def _process_single_spec(self, spec: EvolutionarySpecification) -> bool:
        """
        Handle the lifecycle for a single tool specification.
        """
        log_event(f"🔨 Fabricating tool: {spec.functional_name}", "INFO")
        update_specification_status(spec.id, "fabricating")
        
        get_state_manager().update_agent_status(
            "EvolutionaryAgent", 
            "Fabricating", 
            action=f"Building {spec.functional_name}",
            info={"spec_id": spec.id, "goal": spec.expected_output, "status": "fabricating"}
        )
        
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
            
            get_state_manager().update_agent_status(
                "EvolutionaryAgent", 
                "Refining", 
                action=f"Refining {spec.functional_name}",
                info={"attempt": attempt + 1, "errors": validation_result.error_message[:200]}
            )
            
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
        return self.fabricator._clean_code(response) # Reuse extractor

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
    
    # =========================================================================
    # MCP Server Generation (Open Agentic Web Pattern)
    # =========================================================================
    
    async def synthesize_mcp_server(
        self,
        capability_description: str,
        server_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a complete MCP server for a novel capability.
        
        This implements the "agent as engineer" pattern from the Open Agentic Web
        vision - agents can create their own MCP servers to extend capabilities.
        
        Args:
            capability_description: What the server should do
            server_name: Optional name for the server
            
        Returns:
            Dict with success, file_path, and server config
        """
        
        log_event(f"🔧 Synthesizing MCP server for: {capability_description}", "INFO")
        
        # Generate server name if not provided
        if not server_name:
            server_name = await self._generate_server_name(capability_description)
        
        # Generate server code
        server_code = await self._generate_mcp_server_code(capability_description, server_name)
        
        if not server_code:
            return {"success": False, "message": "Failed to generate server code"}
        
        # Write to mcp_servers directory
        mcp_dir = os.path.join(os.path.dirname(__file__), "..", "..", "mcp_servers")
        os.makedirs(mcp_dir, exist_ok=True)
        
        server_dir = os.path.join(mcp_dir, server_name)
        os.makedirs(server_dir, exist_ok=True)
        
        server_path = os.path.join(server_dir, "server.py")
        with open(server_path, 'w') as f:
            f.write(server_code)
        
        # Generate requirements.txt
        requirements = await self._extract_requirements(server_code)
        if requirements:
            with open(os.path.join(server_dir, "requirements.txt"), 'w') as f:
                f.write('\n'.join(requirements))
        
        # Optionally generate Dockerfile
        dockerfile = self._generate_dockerfile(server_name)
        with open(os.path.join(server_dir, "Dockerfile"), 'w') as f:
            f.write(dockerfile)
        
        log_event(f"✅ MCP server '{server_name}' generated at {server_path}", "INFO")
        
        return {
            "success": True,
            "file_path": server_path,
            "server_name": server_name,
            "server_dir": server_dir,
            "config": {
                "name": server_name,
                "command": "python",
                "args": [server_path],
                "type": "stdio"
            }
        }
    
    async def _generate_server_name(self, description: str) -> str:
        """Generate a server name from description."""
        prompt = f"""Generate a short, lowercase, snake_case name for an MCP server:
        
Description: {description}

Return ONLY the name (e.g., "weather_api" or "file_operations"), no explanation."""
        
        response = await self.llm_client.generate_async(prompt)
        name = response.strip().lower().replace(' ', '_').replace('-', '_')
        # Sanitize
        name = ''.join(c for c in name if c.isalnum() or c == '_')
        return name[:30] or "custom_server"
    
    async def _generate_mcp_server_code(self, description: str, name: str) -> Optional[str]:
        """Generate MCP server Python code."""
        prompt = f"""Write a Python MCP (Model Context Protocol) server for:

{description}

Server name: {name}

Requirements:
1. Use stdio transport (read from stdin, write to stdout)
2. Implement JSON-RPC 2.0 message handling
3. Support tools/list and tools/call methods
4. Include proper error handling
5. Use only standard library + common packages

Output ONLY the complete Python code, no explanations.

Example structure:
```python
import sys
import json

def handle_request(request):
    method = request.get("method")
    if method == "tools/list":
        return {{"tools": [...]}}
    elif method == "tools/call":
        # Handle tool execution
        pass
    return {{"error": {{"code": -32601, "message": "Method not found"}}}}

def main():
    for line in sys.stdin:
        request = json.loads(line)
        result = handle_request(request)
        result["jsonrpc"] = "2.0"
        result["id"] = request.get("id")
        print(json.dumps(result), flush=True)

if __name__ == "__main__":
    main()
```"""
        
        response = await self.llm_client.generate_async(prompt)
        return self.fabricator._clean_code(response)
    
    async def _extract_requirements(self, code: str) -> List[str]:
        """Extract pip requirements from generated code."""
        standard_libs = {'sys', 'json', 'os', 'typing', 'datetime', 're', 'collections', 'io'}
        imports = set()
        
        for line in code.split('\n'):
            line = line.strip()
            if line.startswith('import '):
                module = line.split()[1].split('.')[0]
                if module not in standard_libs:
                    imports.add(module)
            elif line.startswith('from ') and ' import ' in line:
                module = line.split()[1].split('.')[0]
                if module not in standard_libs:
                    imports.add(module)
        
        return list(imports)
    
    def _generate_dockerfile(self, server_name: str) -> str:
        """Generate a Dockerfile for the MCP server."""
        return f'''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || true

COPY server.py /app/server.py

CMD ["python", "server.py"]
'''
    
    # =========================================================================
    # Voyager Pattern: Persistent Skill Library
    # =========================================================================
    
    async def add_to_skill_library(
        self,
        skill_name: str,
        skill_code: str,
        description: str,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Add a successful tool/skill to the persistent library.
        
        Implements the Voyager pattern where successful skills are
        accumulated into a reusable library for future tasks.
        
        Args:
            skill_name: Name of the skill
            skill_code: The code that implements the skill
            description: What the skill does
            tags: Optional tags for categorization
        """
        
        skill_library_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "skill_library.json"
        )
        
        # Load existing library
        try:
            if os.path.exists(skill_library_path):
                with open(skill_library_path, 'r') as f:
                    library = json.load(f)
            else:
                library = {"skills": []}
        except:
            library = {"skills": []}
        
        # Add new skill
        skill_entry = {
            "name": skill_name,
            "description": description,
            "code": skill_code,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "usage_count": 0
        }
        
        # Check for duplicates
        existing_names = {s["name"] for s in library["skills"]}
        if skill_name in existing_names:
            # Update existing
            for i, s in enumerate(library["skills"]):
                if s["name"] == skill_name:
                    library["skills"][i] = skill_entry
                    break
        else:
            library["skills"].append(skill_entry)
        
        # Save
        try:
            with open(skill_library_path, 'w') as f:
                json.dump(library, f, indent=2)
            log_event(f"📚 Added '{skill_name}' to skill library", "INFO")
            return True
        except Exception as e:
            logger.error(f"Failed to save skill library: {e}")
            return False
    
    def get_relevant_skills(self, task_description: str, max_skills: int = 5) -> List[Dict]:
        """
        Retrieve relevant skills from the library for a task.
        
        Args:
            task_description: What the current task needs
            max_skills: Maximum skills to return
        """
        
        skill_library_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "skill_library.json"
        )
        
        if not os.path.exists(skill_library_path):
            return []
        
        try:
            with open(skill_library_path, 'r') as f:
                library = json.load(f)
        except:
            return []
        
        # Simple keyword matching (could be enhanced with embeddings)
        task_words = set(task_description.lower().split())
        scored_skills = []
        
        for skill in library.get("skills", []):
            skill_words = set(skill.get("description", "").lower().split())
            skill_words.update(skill.get("tags", []))
            skill_words.add(skill.get("name", "").lower())
            
            overlap = len(task_words & skill_words)
            if overlap > 0:
                scored_skills.append((overlap, skill))
        
        scored_skills.sort(reverse=True)
        return [s[1] for s in scored_skills[:max_skills]]

# Global instance
_evolutionary_agent = None

def get_evolutionary_agent() -> EvolutionaryAgent:
    global _evolutionary_agent
    if _evolutionary_agent is None:
        _evolutionary_agent = EvolutionaryAgent()
    return _evolutionary_agent
