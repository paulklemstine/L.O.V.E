
import logging
import ast
import subprocess
import os
from typing import Tuple

class VerificationPipeline:
    def __init__(self, sandbox):
        """
        Args:
            sandbox: Instance of DockerSandbox (or compatible interface)
        """
        self.sandbox = sandbox

    def verify_syntax(self, code: str) -> bool:
        """
        Gate 1: Syntax Check.
        Returns True if the code is syntactically valid python.
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logging.error(f"Syntax Verification Failed: {e}")
            return False

    def verify_semantics(self, test_file_path: str) -> bool:
        """
        Gate 2: Semantic Check.
        Runs the provided test file in the sandbox.
        Assumes the test file is already reachable by the sandbox (e.g. in the mounted volume).
        If the sandbox mounts the root, we generally pass relative path or path inside container.
        
        Args:
            test_file_path: Path to the test file *relative to the project root* (or absolute path that works in container).
            Ideally relative.
        """
        # We assume instructions are to run pytest on this file
        command = f"python3 -m pytest {test_file_path}"
        
        logging.info(f"Running Semantic Verification: {command}")
        exit_code, stdout, stderr = self.sandbox.run_command(command)
        
        if exit_code != 0:
            logging.error(f"Semantic Verification Failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
            return False
            
        logging.info("Semantic Verification Passed.")
        return True

    def verify_style(self, file_path: str) -> bool:
        """
        Gate 3: Style Check.
        Runs 'ruff check' on the file.
        Returns True if passed.
        """
        # We run this locally or in sandbox? 
        # Ideally in sandbox to match environment, but valid syntax/style is usually environment independent-ish.
        # But 'ruff' needs to be installed. It's in requirements.txt. 
        # If we run locally, we rely on local env. If sandbox, we rely on docker image.
        # Dockerfile installs requirements.txt. So running in sandbox is safer/more consistent.
        # Let's run in Sandbox for consistency with "The Guardian".
        
        command = f"ruff check {file_path}"
        exit_code, stdout, stderr = self.sandbox.run_command(command)
        
        if exit_code != 0:
            logging.warning(f"Style Verification (Ruff) Failed/Warned:\n{stdout}")
            # Depending on severity, we might still return True if we only care about errors not warnings?
            # But "Rejects changes" implies strictness.
            # Ruff exit code 0 means no lint violations (or fixes applied if --fix used, but verify shouldn't fix).
            # Ruff exit code 1 means violations found.
            return False
            
        return True

    def verify_all(self, target_file_relative: str, test_file_relative: str) -> bool:
        """
        Runs the full pipeline.
        
        Args:
            target_file_relative: e.g. "core/agent.py" (used for Style Check)
            test_file_relative: e.g. "tests/temp_evolve.py" (used for Semantic Check)
        
        Note: Syntax check requires the *content*, but here we assume the file is already written 
        and we are verifying the *repo state*? 
        Or do we pass code content for syntax check before writing?
        Usually:
        1. Agent generates code -> Syntax Check Code String (Gate 1).
        2. Agent writes code to file (or temp file).
        3. Agent runs Tests (Gate 2).
        4. Agent runs Linter (Gate 3).
        
        If this method assumes files are written:
        It reads target file for syntax? Or skips it because it's already written?
        Let's perform syntax check on the file on disk to be sure.
        """
        # 1. Syntax (Read from disk since we are verifying the file state)
        if not os.path.exists(target_file_relative):
             logging.error(f"Target file {target_file_relative} does not exist for verification.")
             return False
             
        with open(target_file_relative, "r", encoding="utf-8") as f:
            code = f.read()
            
        if not self.verify_syntax(code):
            return False
            
        # 2. Semantics
        if not self.verify_semantics(test_file_relative):
            return False
            
        # 3. Style
        if not self.verify_style(target_file_relative):
            return False
            
        return True
