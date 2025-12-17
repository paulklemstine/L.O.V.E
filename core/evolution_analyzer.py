import os
import json
from pathlib import Path
import core.logging
import ast

# Attempt to import radon for complexity analysis, fallback to simple length check
try:
    from radon.complexity import cc_visit
except ImportError:
    cc_visit = None

async def determine_evolution_goal(knowledge_base=None, love_state=None, deep_agent_instance=None) -> str:
    """
    Determines a specific, incremental evolution goal based on codebase hotspots.
    """
    core.logging.log_event("[EvolutionAnalyzer] identifying high-priority micro-evolution target...", "INFO")
    
    # 1. Find the "Hotspot" - the file most in need of love
    hotspot_file, issue_type, score = _find_codebase_hotspot()
    
    if hotspot_file:
        goal = f"Micro-Evolution: Refactor '{hotspot_file}' to reduce {issue_type} (current score: {score}). Focus on splitting large functions and adding type hints. DO NOT break existing functionality."
        core.logging.log_event(f"[EvolutionAnalyzer] Selected Target: {goal}", "INFO")
        return goal

    # Fallback to standard analysis if code is clean
    return "Perform a routine health check and add one unit test to 'tests/'."

def _find_codebase_hotspot():
    """Scans for the most complex or largest Python file."""
    max_score = 0
    worst_file = None
    issue_type = "complexity"

    for py_file in Path(".").rglob("*.py"):
        if "venv" in str(py_file) or "test" in str(py_file):
            continue
            
        try:
            content = py_file.read_text(encoding='utf-8', errors='ignore')
            
            # Metric 1: Cyclomatic Complexity (if radon available)
            if cc_visit:
                blocks = cc_visit(content)
                score = sum(b.complexity for b in blocks)
                # Normalize by file length to find dense complexity
                if score > max_score and score > 20: # Threshold
                    max_score = score
                    worst_file = py_file.name
                    issue_type = "Cyclomatic Complexity"
            
            # Metric 2: Raw Line Count (Simple heuristic)
            elif len(content.splitlines()) > max_score:
                max_score = len(content.splitlines())
                worst_file = py_file.name
                issue_type = "Line Count"
                
        except Exception:
            continue

    return worst_file, issue_type, max_score
