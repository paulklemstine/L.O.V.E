import os
import json
from pathlib import Path
import core.logging
import ast
import re
from collections import Counter

# Attempt to import radon for complexity analysis, fallback to simple length check
try:
    from radon.complexity import cc_visit
except ImportError:
    cc_visit = None

LOG_FILE = "love.log"

async def determine_evolution_goal(knowledge_base=None, love_state=None, deep_agent_instance=None, exclude_files=None) -> str:
    """
    Determines a specific, incremental evolution goal by performing a strategic analysis
    of the agent's operational history, knowledge base, and codebase.
    """
    core.logging.log_event("[EvolutionAnalyzer] Commencing strategic analysis for self-evolution...", "INFO")

    # 1. Analyze Operational History from love_state.json
    operational_insights = _analyze_operational_history(love_state)

    # 2. Analyze Performance Logs
    log_insights = analyze_performance_logs()

    # 3. Analyze Knowledge Base
    kb_insights = _analyze_knowledge_base(knowledge_base)

    # 4. Find Codebase Hotspots
    hotspot_file, issue_type, score = _find_codebase_hotspot(exclude_files=exclude_files)

    # 5. Synthesize a Strategic Goal
    goal = _synthesize_strategic_goal(operational_insights, log_insights, kb_insights, hotspot_file, issue_type, score)

    core.logging.log_event(f"[EvolutionAnalyzer] New Strategic Goal: {goal}", "INFO")
    return goal

def _analyze_operational_history(love_state):
    """Analyzes love_state for recurring errors and command patterns."""
    if not love_state:
        return {"recurring_errors": [], "command_patterns": {}}

    insights = {}

    # Analyze critical_error_queue
    error_queue = love_state.get('critical_error_queue', [])
    if error_queue:
        error_messages = [e['message'].split('\\n')[0] for e in error_queue]
        error_counts = Counter(error_messages)
        insights['recurring_errors'] = [e for e, count in error_counts.items() if count > 2]

    # Analyze autopilot_history for command patterns
    history = love_state.get('autopilot_history', [])
    if history:
        commands = [item.get('command', '').split(' ')[0] for item in history]
        insights['command_patterns'] = dict(Counter(commands).most_common(5))

    return insights

def analyze_performance_logs():
    """Parses love.log to identify recurring errors and performance warnings."""
    if not os.path.exists(LOG_FILE):
        return {}

    try:
        with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()
    except Exception as e:
        core.logging.log_event(f"[EvolutionAnalyzer] Could not read log file: {e}", "WARNING")
        return {}

    insights = {}
    
    # Find recurring error messages
    error_pattern = re.compile(r"ERROR: (.*)")
    errors = error_pattern.findall(log_content)
    if errors:
        error_counts = Counter(errors)
        insights['recurring_log_errors'] = [e for e, count in error_counts.items() if count > 3]

    return insights

def _analyze_knowledge_base(knowledge_base):
    """Analyzes the knowledge base for strategic insights (placeholder for future expansion)."""
    if not knowledge_base:
        return {}
    return {"nodes": len(knowledge_base.get_all_nodes()) if knowledge_base else 0}


def _find_codebase_hotspot(exclude_files=None):
    """Scans for the most complex or largest Python file."""
    if exclude_files is None:
        exclude_files = []
    max_score = 0
    worst_file = None
    issue_type = "complexity"

    for py_file in Path(".").rglob("*.py"):
        if "venv" in str(py_file) or "test" in str(py_file) or py_file.name in exclude_files:
            continue
            
        try:
            content = py_file.read_text(encoding='utf-8', errors='ignore')
            
            if cc_visit:
                blocks = cc_visit(content)
                score = sum(b.complexity for b in blocks)
                if score > max_score and score > 20:
                    max_score = score
                    worst_file = str(py_file)
                    issue_type = "Cyclomatic Complexity"
            elif len(content.splitlines()) > max_score:
                max_score = len(content.splitlines())
                worst_file = str(py_file)
                issue_type = "Line Count"
                
        except Exception:
            continue

    return worst_file, issue_type, max_score

def _synthesize_strategic_goal(op_insights, log_insights, kb_insights, hotspot_file, issue_type, score):
    """Creates a high-level evolution goal based on all available analysis."""

    # Priority 1: Address recurring critical errors from state
    if op_insights.get('recurring_errors'):
        error_to_fix = op_insights['recurring_errors'][0]
        return f"Fix the recurring critical error from state: '{error_to_fix}'. Analyze the root cause and implement a robust solution."

    # Priority 2: Address recurring errors from logs
    if log_insights.get('recurring_log_errors'):
        error_to_fix = log_insights['recurring_log_errors'][0]
        return f"Fix the recurring error from logs: '{error_to_fix}'. Investigate the conditions leading to this error and apply a fix."

    # Priority 3: Refactor a complex part of the codebase
    if hotspot_file:
        return f"Refactor '{hotspot_file}' to reduce its {issue_type} (current score: {score}). Focus on improving modularity and maintainability."

    # Priority 4: Enhance a frequently used capability
    if op_insights.get('command_patterns'):
        most_used_command = list(op_insights['command_patterns'].keys())[0]
        return f"Enhance the '{most_used_command}' command. Analyze its usage and add a new feature or improve its performance."

    # Fallback Goal
    return "Conduct a general codebase review to identify and implement a minor enhancement to improve overall system efficiency or robustness."

def get_complex_file_target():
    """
    Returns the most complex file in the codebase with its metrics.
    Returns: dict with 'file', 'type', 'score'
    """
    filename, issue_type, score = _find_codebase_hotspot()
    if filename:
        return {"file": filename, "type": issue_type, "score": score}
    return None
