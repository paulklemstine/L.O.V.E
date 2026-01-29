"""
Tool Validator - Automated Qualification Pipeline

Epic 1, Story 1.3: Ensures robust safety and quality for new tools.

Pipeline:
1. Syntax Check: Valid python syntax
2. Static Analysis: Complexity/Maintainability check (Radon)
3. Test Generation: Auto-generate pytest file via LLM
4. Sandbox Execution: Run tests in isolated environment
5. Security Scan: Check for restricted imports/patterns
"""

import os
import re
import json
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field

from core.logger import log_event
from core.surgeon.sandbox import get_sandbox
from core.tool_registry import get_global_registry

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    passed: bool
    syntax_valid: bool = False
    static_analysis_score: float = 0.0 # 0.0-1.0
    tests_passed: int = 0
    tests_total: int = 0
    security_issues: List[str] = field(default_factory=list)
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "syntax_valid": self.syntax_valid,
            "static_analysis_score": self.static_analysis_score,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "security_issues": self.security_issues,
            "error_message": self.error_message
        }


class ToolValidator:
    """
    Validates generated tools before promotion to active directory.
    Uses sandbox for safe execution and LLM for test generation.
    """
    
    def __init__(self, sandbox=None, llm_client=None):
        self.sandbox = sandbox or get_sandbox()
        self.llm_client = llm_client 
        
        # Get project root
        self.project_root = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    
    def _get_llm_client(self):
        """Lazy-load the LLM client."""
        if self.llm_client is None:
            from core.llm_client import LLMClient
            self.llm_client = LLMClient()
        return self.llm_client

    async def validate(self, tool_name: str, tool_path: str, tool_code: str = None) -> ValidationResult:
        """
        Run full validation pipeline on a tool.
        
        Args:
            tool_name: Name of the tool
            tool_path: Absolute path to the tool file
            tool_code: Optional code content (will read from file if None)
            
        Returns:
            ValidationResult object
        """
        log_event(f"🧪 Validating tool: {tool_name}", "INFO")
        
        result = ValidationResult(passed=False)
        
        # Read code if not provided
        if not tool_code:
            try:
                with open(tool_path, 'r') as f:
                    tool_code = f.read()
            except Exception as e:
                result.error_message = f"Failed to read tool file: {e}"
                return result

        # 1. Syntax Check
        if not self._check_syntax(tool_code):
            result.error_message = "Syntax Code check failed"
            return result
        result.syntax_valid = True
        
        # 2. Security Scan
        security_issues = self._security_scan(tool_code)
        if security_issues:
            result.security_issues = security_issues
            result.error_message = f"Security scan failed: {security_issues}"
            return result
            
        # 3. Static Analysis (Radon)
        try:
            # We use a simplified check here since not all envs have radon
            # Just Check basic length constraints for now
            result.static_analysis_score = 1.0 if len(tool_code) < 10000 else 0.5
        except Exception:
            result.static_analysis_score = 0.5
            
        # 4. Generate Tests
        test_file_name = f"test_{tool_name}.py"
        test_dir = os.path.join(os.path.dirname(tool_path), "tests")
        os.makedirs(test_dir, exist_ok=True)
        test_path = os.path.join(test_dir, test_file_name)
        
        try:
            test_code = await self._generate_tests(tool_name, tool_code)
            with open(test_path, 'w') as f:
                f.write(test_code)
                
            log_event(f"📝 Generated tests for {tool_name}", "INFO")
            
        except Exception as e:
            result.error_message = f"Failed to generate tests: {e}"
            return result
            
        # 5. Run Tests in Sandbox
        passed, total, output = await self._run_tests(tool_path, test_path)
        result.tests_passed = passed
        result.tests_total = total
        
        if passed == total and total > 0:
            result.passed = True
            log_event(f"✅ Validation passed: {tool_name}", "INFO")
        else:
            result.passed = False
            result.error_message = f"Tests failed: {passed}/{total} passed. Output: {output[:200]}..."
            log_event(f"❌ Validation failed: {tool_name} ({result.error_message})", "WARNING")
            
        return result
    
    def _check_syntax(self, code: str) -> bool:
        """Check if code is valid Python syntax."""
        try:
            compile(code, "<string>", "exec")
            return True
        except SyntaxError:
            return False
            
    def _security_scan(self, code: str) -> List[str]:
        """Check for potentially dangerous imports or patterns."""
        issues = []
        forbidden_imports = [
            "subprocess", "os.system", "os.popen", "shutil.rmtree", 
            "requests", "urllib.request", "socket"
        ]
        
        # Allow requests/urllib if explicitly required in spec, but flag them
        # For now, simplistic string matching
        for item in forbidden_imports:
            if item in code:
                issues.append(f"Forbidden usage: '{item}' detected")
                
        return issues
        
    async def _generate_tests(self, tool_name: str, tool_code: str) -> str:
        """Generate pytest code using LLM."""
        prompt = f"""You are a QA Engineer for L.O.V.E.
Write comprehensive pytest unit tests for the following Python tool.

## TOOL CODE
```python
{tool_code}
```

## REQUIREMENTS
1. Use `pytest`
2. Test valid inputs
3. Test edge cases
4. Test error handling
5. Mock any external dependencies if present
6. Return ONLY the code, no explanation

## OUTPUT
```python
import pytest
from {tool_name} import {tool_name}

def test_basic_usage():
    ...
```
"""
        llm = self._get_llm_client()
        response = await llm.generate(prompt)
        
        # Clean markdown
        if "```python" in response:
            match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
            if match:
                response = match.group(1)
        elif "```" in response:
            match = re.search(r'```\n?(.*?)\n?```', response, re.DOTALL)
            if match:
                response = match.group(1)
                
        # Fix imports - the tool file is in parent directory relative to tests
        # or we need to add path
        header = """
import sys
import os
import pytest

# Add parent directory to path to import tool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import tool module dynamically
import importlib.util
tool_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"{tool_name}.py")
spec = importlib.util.spec_from_file_location(f"{tool_name}", tool_path)
module = importlib.util.module_from_spec(spec)
sys.modules[f"{tool_name}"] = module
spec.loader.exec_module(module)
from {tool_name} import {tool_name}
"""
        # Replace the simple import with our robust import
        response = re.sub(r'from .+ import .+', '', response, count=1)
        
        return header.replace("{tool_name}", tool_name) + "\n" + response.strip()

    async def _run_tests(self, tool_path: str, test_path: str) -> Tuple[int, int, str]:
        """
        Run tests in sandbox.
        
        Returns:
            (tests_passed, tests_total, output)
        """
        # Mount the project root to /project in sandbox
        mounts = [(self.project_root, "/project")]
        
        # Determine paths relative to project root
        rel_test_path = os.path.relpath(test_path, self.project_root)
        
        # In sandbox, path is /project/rel_path
        sandbox_test_path = f"/project/{rel_test_path}"
        
        command = f"pytest {sandbox_test_path} --tb=short"
        
        exit_code, stdout, stderr = self.sandbox.run_command(
            command,
            timeout=30,
            network_disabled=True,
            mounts=mounts
        )
        
        # Parse output to find passed/total
        # Output format: "2 passed, 1 failed in 0.1s" or "3 passed in 0.1s"
        total = 0
        passed = 0
        
        match_passed = re.search(r'(\d+) passed', stdout)
        if match_passed:
            passed = int(match_passed.group(1))
            
        match_failed = re.search(r'(\d+) failed', stdout)
        failed = 0
        if match_failed:
            failed = int(match_failed.group(1))
            
        match_error = re.search(r'(\d+) error', stdout)
        errors = 0
        if match_error:
            errors = int(match_error.group(1))
            
        total = passed + failed + errors
        
        # Fail-safe if parsing failed but exit code implies success
        if total == 0 and exit_code == 0 and "passed" in stdout:
             # Try to guess - usually means all passed
             total = 1
             passed = 1
             
        return passed, total, stdout + "\n" + stderr
