
import os
import json
import logging
from typing import List, Dict, Any, Tuple
# radon
from radon.complexity import cc_visit
import radon.visitors
# coverage
try:
    import coverage
except ImportError:
    coverage = None

from core.surgeon.ast_parser import extract_function_metadata

class Scout:
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.cov = None
        if coverage:
            try:
                self.cov = coverage.Coverage(data_file=os.path.join(project_root, ".coverage"))
                self.cov.load()
            except Exception:
                self.cov = None

    def analyze_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Analyzes a single file for functions, their complexity, and coverage.
        """
        if not file_path.endswith(".py"):
            return []
            
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
            
        try:
            blocks = cc_visit(code)
        except Exception as e:
            logging.warning(f"Failed to analyze complexity for {file_path}: {e}")
            return []
            
        results = []
        for block in blocks:
            # Check type using isinstance or class name
            # radon types: Function, Method, Class
            b_type = type(block).__name__.lower() # 'function', 'method', 'class'
            
            if b_type == 'class':
                # We typically ignore class blocks themselves, as we care about methods inside?
                # Actually, radon flattens methods usually?
                pass
            
            if b_type in ('function', 'method'):
                name = block.name
                if b_type == 'method':
                     if getattr(block, 'classname', None):
                         name = f"{block.classname}.{block.name}"
                
                complexity = block.complexity
                
                start = block.lineno
                end = getattr(block, 'endline', start) 
                
                cov_percent = 0.0
                if self.cov:
                    try:
                        pass # Coverage logic pending
                    except Exception:
                        pass
                
                score_val = complexity * (1.0 - cov_percent)
                
                results.append({
                    "file": file_path,
                    "name": name,
                    "type": b_type,
                    "complexity": complexity,
                    "coverage": cov_percent,
                    "score": score_val,
                    "start_line": start,
                    "end_line": end
                })
                
        return results

    def scan(self) -> List[Dict[str, Any]]:
        targets = []
        for root, dirs, files in os.walk(self.project_root):
            if "venv" in root or ".git" in root or "__pycache__" in root:
                continue
            for f in files:
                if f.endswith(".py"):
                    full_path = os.path.join(root, f)
                    targets.extend(self.analyze_file(full_path))
                    
        targets.sort(key=lambda x: x["score"], reverse=True)
        return targets

if __name__ == "__main__":
    scout = Scout(os.getcwd())
    top_targets = scout.scan()[:10]
    print(json.dumps(top_targets, indent=2))
