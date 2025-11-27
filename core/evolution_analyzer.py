# core/evolution_analyzer.py

import os
import json
from pathlib import Path
import core.logging
from typing import Optional


async def determine_evolution_goal(
    knowledge_base=None, 
    love_state=None, 
    deep_agent_instance=None
) -> str:
    """
    Analyzes the system and determines an appropriate evolution goal.
    
    This function performs multi-faceted analysis:
    1. Checks recent logs for errors and issues
    2. Queries knowledge base for strategic opportunities
    3. Scans codebase for TODOs and technical debt
    4. Reviews system state for performance bottlenecks
    5. Synthesizes findings into a single actionable goal using LLM
    
    Args:
        knowledge_base: The knowledge base instance for querying strategic info
        love_state: Current system state dictionary
        deep_agent_instance: DeepAgent instance for LLM access
        
    Returns:
        str: A specific, actionable evolution goal
    """
    core.logging.log_event("[EvolutionAnalyzer] Starting automatic goal determination...", "INFO")
    
    analysis_data = {
        "recent_errors": [],
        "todo_items": [],
        "kb_insights": [],
        "performance_issues": [],
        "system_state": {}
    }
    
    # 1. Analyze recent logs for errors
    analysis_data["recent_errors"] = await _analyze_recent_logs()
    
    # 2. Scan codebase for TODO/FIXME comments
    analysis_data["todo_items"] = await _scan_codebase_todos()
    
    # 3. Query knowledge base for strategic opportunities
    if knowledge_base:
        analysis_data["kb_insights"] = await _query_knowledge_base(knowledge_base)
    
    # 4. Review system state for issues
    if love_state:
        analysis_data["performance_issues"] = _analyze_system_state(love_state)
        analysis_data["system_state"] = {
            "gpu_available": love_state.get("hardware", {}).get("gpu_available", False),
            "vram_mb": love_state.get("hardware", {}).get("gpu_vram_mb", 0),
            "cpu_count": love_state.get("hardware", {}).get("cpu_count", 0)
        }
    
    # 5. Synthesize findings into a goal using LLM
    goal = await _synthesize_goal_with_llm(analysis_data, deep_agent_instance)
    
    core.logging.log_event(f"[EvolutionAnalyzer] Determined goal: {goal}", "INFO")
    return goal


async def _analyze_recent_logs() -> list:
    """Analyzes recent log files for errors and issues."""
    errors = []
    
    try:
        log_dir = Path("logs")
        if not log_dir.exists():
            return errors
        
        # Get the most recent log file
        log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not log_files:
            return errors
        
        recent_log = log_files[0]
        
        # Read last 100 lines
        with open(recent_log, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            recent_lines = lines[-100:] if len(lines) > 100 else lines
        
        # Look for ERROR and WARNING patterns
        error_patterns = {}
        for line in recent_lines:
            if "ERROR" in line or "CRITICAL" in line:
                # Extract error message
                if ":" in line:
                    error_msg = line.split(":", 2)[-1].strip()
                    # Count occurrences
                    error_patterns[error_msg] = error_patterns.get(error_msg, 0) + 1
        
        # Return top 5 most frequent errors
        sorted_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        errors = [{"message": msg, "count": count} for msg, count in sorted_errors]
        
    except Exception as e:
        core.logging.log_event(f"[EvolutionAnalyzer] Error analyzing logs: {e}", "WARNING")
    
    return errors


async def _scan_codebase_todos() -> list:
    """Scans the codebase for TODO and FIXME comments."""
    todos = []
    
    try:
        project_root = Path(".")
        
        # Scan Python files
        for py_file in project_root.rglob("*.py"):
            # Skip virtual environments and hidden directories
            if any(part.startswith('.') or part in ['venv', 'env', '__pycache__'] 
                   for part in py_file.parts):
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if "TODO" in line or "FIXME" in line:
                            todos.append({
                                "file": str(py_file),
                                "line": line_num,
                                "text": line.strip()
                            })
                            
                            # Limit to 20 TODOs to avoid overwhelming the LLM
                            if len(todos) >= 20:
                                break
            except Exception:
                continue
            
            if len(todos) >= 20:
                break
                
    except Exception as e:
        core.logging.log_event(f"[EvolutionAnalyzer] Error scanning TODOs: {e}", "WARNING")
    
    return todos


async def _query_knowledge_base(knowledge_base) -> list:
    """Queries the knowledge base for strategic insights and improvement opportunities."""
    insights = []
    
    try:
        from core.kb_tools import search_kb
        
        # Search for strategic keywords
        search_queries = [
            "improvement opportunity",
            "strategic goal",
            "technical debt",
            "optimization",
            "bug fix needed"
        ]
        
        for query in search_queries:
            try:
                results_json = search_kb(query, top_k=2, knowledge_base=knowledge_base)
                results = json.loads(results_json)
                
                if results.get("count", 0) > 0:
                    for item in results.get("results", []):
                        insights.append({
                            "query": query,
                            "content": item.get("content", "")[:200]  # Limit length
                        })
            except Exception:
                continue
        
    except Exception as e:
        core.logging.log_event(f"[EvolutionAnalyzer] Error querying KB: {e}", "WARNING")
    
    return insights


def _analyze_system_state(love_state: dict) -> list:
    """Analyzes the system state for performance issues or resource constraints."""
    issues = []
    
    try:
        hardware = love_state.get("hardware", {})
        
        # Check GPU availability
        if not hardware.get("gpu_available", False):
            issues.append("No GPU available - consider optimizing for CPU-only operation")
        
        # Check VRAM
        vram_mb = hardware.get("gpu_vram_mb", 0)
        if 0 < vram_mb < 8000:
            issues.append(f"Limited VRAM ({vram_mb}MB) - consider memory optimizations")
        
        # Check CPU count
        cpu_count = hardware.get("cpu_count", 0)
        if cpu_count < 4:
            issues.append(f"Limited CPU cores ({cpu_count}) - consider async optimizations")
        
    except Exception as e:
        core.logging.log_event(f"[EvolutionAnalyzer] Error analyzing system state: {e}", "WARNING")
    
    return issues


async def _synthesize_goal_with_llm(analysis_data: dict, deep_agent_instance=None) -> str:
    """Uses an LLM to synthesize all analysis data into a single actionable goal."""
    
    # Build the synthesis prompt
    # Construct the analysis data string for the prompt
    analysis_data_str = ""
    
    # Add recent errors
    if analysis_data["recent_errors"]:
        analysis_data_str += "## Recent Errors:\n"
        for error in analysis_data["recent_errors"]:
            analysis_data_str += f"- {error['message']} (occurred {error['count']} times)\n"
        analysis_data_str += "\n"
    
    # Add TODO items
    if analysis_data["todo_items"]:
        analysis_data_str += "## TODO/FIXME Items:\n"
        for todo in analysis_data["todo_items"][:10]:  # Limit to top 10
            analysis_data_str += f"- {todo['file']}:{todo['line']} - {todo['text']}\n"
        analysis_data_str += "\n"
    
    # Add KB insights
    if analysis_data["kb_insights"]:
        analysis_data_str += "## Knowledge Base Insights:\n"
        for insight in analysis_data["kb_insights"][:5]:  # Limit to top 5
            analysis_data_str += f"- {insight['content']}\n"
        analysis_data_str += "\n"
    
    # Add performance issues
    if analysis_data["performance_issues"]:
        analysis_data_str += "## System Performance Issues:\n"
        for issue in analysis_data["performance_issues"]:
            analysis_data_str += f"- {issue}\n"
        analysis_data_str += "\n"
    
    # Add system state
    if analysis_data["system_state"]:
        analysis_data_str += f"## System State:\n{json.dumps(analysis_data['system_state'], indent=2)}\n\n"
    
    try:
        # Use the LLM pool if available
        # Use the LLM pool if available
        if deep_agent_instance and hasattr(deep_agent_instance, 'use_pool') and deep_agent_instance.use_pool:
            from core.llm_api import run_llm
            result_dict = await run_llm(prompt_key="evolution_goal_synthesis", prompt_vars={"analysis_data_str": analysis_data_str}, purpose="evolution_goal_synthesis", deep_agent_instance=None)
            goal = result_dict.get("result", "").strip()
        elif deep_agent_instance:
            # Use the deep agent's generate method
            # We need to render the prompt manually here since we can't use run_llm with prompt_key easily
            # without importing PromptRegistry, but let's try to stick to run_llm if possible.
            # Actually, let's just use run_llm even if use_pool is False, as run_llm handles it.
            # But wait, the original code had a specific branch for deep_agent_instance.generate.
            # Let's assume run_llm is the preferred way now.
            from core.llm_api import run_llm
            result_dict = await run_llm(prompt_key="evolution_goal_synthesis", prompt_vars={"analysis_data_str": analysis_data_str}, purpose="evolution_goal_synthesis", deep_agent_instance=None)
            goal = result_dict.get("result", "").strip()
        else:
            # Fallback: use a simple heuristic
            goal = _fallback_goal_determination(analysis_data)
        
        # Clean up the goal
        goal = goal.replace('"', '').replace("'", '').strip()
        
        # If the goal is too long or empty, use fallback
        if not goal or len(goal) > 300:
            goal = _fallback_goal_determination(analysis_data)
        
        return goal
        
    except Exception as e:
        core.logging.log_event(f"[EvolutionAnalyzer] Error synthesizing goal with LLM: {e}", "WARNING")
        return _fallback_goal_determination(analysis_data)


def _fallback_goal_determination(analysis_data: dict) -> str:
    """Fallback heuristic for determining a goal when LLM synthesis fails."""
    
    # Priority 1: Fix recurring errors
    if analysis_data["recent_errors"]:
        top_error = analysis_data["recent_errors"][0]
        return f"Fix the recurring error: {top_error['message'][:100]}"
    
    # Priority 2: Address performance issues
    if analysis_data["performance_issues"]:
        return f"Address system issue: {analysis_data['performance_issues'][0]}"
    
    # Priority 3: Complete TODO items
    if analysis_data["todo_items"]:
        todo = analysis_data["todo_items"][0]
        return f"Complete TODO in {Path(todo['file']).name}: {todo['text'][:100]}"
    
    # Priority 4: KB insights
    if analysis_data["kb_insights"]:
        return f"Implement strategic improvement: {analysis_data['kb_insights'][0]['content'][:100]}"
    
    # Default: general improvement
    return "Improve overall system stability and performance through code refactoring"
