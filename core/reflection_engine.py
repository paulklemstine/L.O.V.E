"""
Story 1.1: The "Mirror" Test (Self-Reflection Engine)

This module provides metacognitive self-analysis capabilities for L.O.V.E.,
allowing the agent to analyze its past interactions during idle time and
identify patterns for improvement.
"""
import json
import os
import yaml
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from core.logging import log_event

# Constants
PERSONA_FILE = "persona.yaml"
REFLECTION_OUTPUT_DIR = "_memory_"


class ReflectionReport:
    """
    A structured report summarizing the agent's self-analysis.
    
    Attributes:
        generated_at: Timestamp of report generation
        interaction_count: Number of interactions analyzed
        findings: List of identified patterns and issues
        improvements: List of suggested improvements
        persona_diff: Proposed changes to persona.yaml
    """
    
    def __init__(
        self,
        interaction_count: int,
        findings: List[Dict[str, Any]],
        improvements: List[str],
        persona_diff: Optional[Dict[str, Any]] = None
    ):
        self.generated_at = datetime.now().isoformat()
        self.interaction_count = interaction_count
        self.findings = findings
        self.improvements = improvements
        self.persona_diff = persona_diff or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "generated_at": self.generated_at,
            "interaction_count": self.interaction_count,
            "findings": self.findings,
            "improvements": self.improvements,
            "persona_diff": self.persona_diff
        }
    
    def to_markdown(self) -> str:
        """Generate a Markdown summary of the report."""
        lines = [
            "# 🪞 Self-Reflection Report",
            f"\n**Generated:** {self.generated_at}",
            f"**Interactions Analyzed:** {self.interaction_count}",
            "\n## 📊 Findings\n"
        ]
        
        for i, finding in enumerate(self.findings, 1):
            finding_type = finding.get("type", "Unknown")
            description = finding.get("description", "")
            lines.append(f"{i}. **{finding_type}**: {description}")
        
        if not self.findings:
            lines.append("*No significant patterns detected.*")
        
        lines.append("\n## 💡 Suggested Improvements\n")
        for i, improvement in enumerate(self.improvements, 1):
            lines.append(f"{i}. {improvement}")
        
        if not self.improvements:
            lines.append("*No improvements needed at this time.*")
        
        if self.persona_diff:
            lines.append("\n## 🔧 Proposed Persona Updates\n")
            lines.append("```yaml")
            lines.append(yaml.dump(self.persona_diff, default_flow_style=False))
            lines.append("```")
        
        return "\n".join(lines)


class ReflectionEngine:
    """
    Analyzes the agent's past interactions during idle time to identify
    patterns of repetitive phrasing, suboptimal logic, or overused tools.
    """
    
    def __init__(self, love_state: Dict[str, Any] = None, persona_path: str = PERSONA_FILE):
        """
        Initialize the ReflectionEngine.
        
        Args:
            love_state: Reference to the shared love_state dict
            persona_path: Path to persona.yaml file
        """
        self.love_state = love_state or {}
        self.persona_path = persona_path
        self._persona_cache: Optional[Dict[str, Any]] = None
    
    def get_interaction_history(self, count: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieves the last N interactions from autopilot_history.
        
        Args:
            count: Number of recent interactions to retrieve
            
        Returns:
            List of interaction dicts
        """
        history = self.love_state.get("autopilot_history", [])
        return history[-count:] if len(history) >= count else history
    
    def analyze_interactions(self, count: int = 10) -> Dict[str, Any]:
        """
        Examines the last N interactions for patterns.
        
        Args:
            count: Number of interactions to analyze
            
        Returns:
            Dict with analysis results:
            - tool_usage: Counter of tool invocations
            - repetitive_phrases: List of repeated phrases
            - overused_tools: Tools used more than threshold
            - error_patterns: Common error types
        """
        history = self.get_interaction_history(count)
        
        if not history:
            return {
                "tool_usage": {},
                "repetitive_phrases": [],
                "overused_tools": [],
                "error_patterns": [],
                "total_analyzed": 0
            }
        
        # Analyze tool usage
        tool_counter: Counter = Counter()
        commands: List[str] = []
        errors: List[str] = []
        
        for interaction in history:
            command = interaction.get("command", "")
            if command:
                # Extract tool name from command string (first word)
                tool_name = command.split()[0] if command.split() else "unknown"
                tool_counter[tool_name] += 1
                commands.append(command)
            
            # Track errors
            result = interaction.get("result", "")
            if isinstance(result, str) and ("error" in result.lower() or "failed" in result.lower()):
                errors.append(result[:200])  # Truncate long errors
        
        # Identify repetitive phrases (commands that appear multiple times)
        command_counter = Counter(commands)
        repetitive_phrases = [
            {"phrase": cmd, "count": count}
            for cmd, count in command_counter.items()
            if count > 1
        ]
        
        # Identify overused tools (more than 30% of total)
        total_tool_calls = sum(tool_counter.values())
        overuse_threshold = max(2, total_tool_calls * 0.3)
        overused_tools = [
            {"tool": tool, "count": count, "percentage": round(count / total_tool_calls * 100, 1)}
            for tool, count in tool_counter.items()
            if count >= overuse_threshold
        ]
        
        # Identify error patterns
        error_counter = Counter(errors)
        error_patterns = [
            {"pattern": err, "count": count}
            for err, count in error_counter.most_common(3)
        ]
        
        return {
            "tool_usage": dict(tool_counter),
            "repetitive_phrases": repetitive_phrases,
            "overused_tools": overused_tools,
            "error_patterns": error_patterns,
            "total_analyzed": len(history)
        }
    
    def generate_improvements(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generates improvement suggestions based on analysis.
        
        Args:
            analysis: Result from analyze_interactions()
            
        Returns:
            List of improvement suggestions
        """
        improvements = []
        
        # Suggest reducing tool overuse
        for tool_info in analysis.get("overused_tools", []):
            tool = tool_info["tool"]
            pct = tool_info["percentage"]
            improvements.append(
                f"Consider diversifying tool usage. Tool '{tool}' was used {pct}% of the time. "
                f"Explore alternative approaches or tools."
            )
        
        # Suggest avoiding repetitive commands
        for phrase_info in analysis.get("repetitive_phrases", []):
            if phrase_info["count"] >= 3:
                improvements.append(
                    f"Repetitive command detected: '{phrase_info['phrase'][:50]}...' appeared {phrase_info['count']} times. "
                    f"Consider caching results or restructuring logic."
                )
        
        # Suggest error handling improvements
        if analysis.get("error_patterns"):
            improvements.append(
                f"Recurring errors detected ({len(analysis['error_patterns'])} pattern(s)). "
                f"Review error handling and add defensive checks."
            )
        
        # If no issues found
        if not improvements:
            improvements.append(
                "No significant issues detected. The system is operating within expected parameters."
            )
        
        return improvements
    
    def suggest_persona_update(self, improvements: List[str]) -> Dict[str, Any]:
        """
        Generates a diff/PR-like structure for persona.yaml modifications.
        Does NOT auto-apply changes; instead returns them for human review.
        
        Args:
            improvements: List of improvement suggestions
            
        Returns:
            Dict with proposed YAML changes
        """
        if not improvements or len(improvements) == 1 and "No significant issues" in improvements[0]:
            return {}
        
        # Load current persona
        persona = self._load_persona()
        if not persona:
            return {}
        
        # Generate improvement-based updates
        proposed_changes = {
            "creator_directives": [],
            "interaction_rules": []
        }
        
        for improvement in improvements:
            if "tool" in improvement.lower() and "diversif" in improvement.lower():
                proposed_changes["creator_directives"].append(
                    "Diversify tool usage - avoid over-reliance on any single tool"
                )
            elif "repetitive" in improvement.lower():
                proposed_changes["interaction_rules"].append(
                    "Cache and reuse results instead of repeating identical operations"
                )
            elif "error" in improvement.lower():
                proposed_changes["interaction_rules"].append(
                    "Add defensive error handling before attempting risky operations"
                )
        
        # Clean up empty sections
        return {k: v for k, v in proposed_changes.items() if v}
    
    def generate_report(self, count: int = 10) -> ReflectionReport:
        """
        Creates a complete Reflection Report summarizing findings.
        
        Args:
            count: Number of interactions to analyze
            
        Returns:
            ReflectionReport object
        """
        analysis = self.analyze_interactions(count)
        
        # Convert analysis to findings format
        findings = []
        
        for tool_info in analysis.get("overused_tools", []):
            findings.append({
                "type": "Tool Overuse",
                "description": f"Tool '{tool_info['tool']}' used {tool_info['percentage']}% of the time"
            })
        
        for phrase_info in analysis.get("repetitive_phrases", []):
            if phrase_info["count"] >= 2:
                findings.append({
                    "type": "Repetitive Command",
                    "description": f"Command appeared {phrase_info['count']} times: {phrase_info['phrase'][:60]}..."
                })
        
        for error_info in analysis.get("error_patterns", []):
            findings.append({
                "type": "Error Pattern",
                "description": f"Recurring error ({error_info['count']}x): {error_info['pattern'][:80]}..."
            })
        
        improvements = self.generate_improvements(analysis)
        persona_diff = self.suggest_persona_update(improvements)
        
        report = ReflectionReport(
            interaction_count=analysis["total_analyzed"],
            findings=findings,
            improvements=improvements,
            persona_diff=persona_diff
        )
        
        # Log the reflection event
        log_event(f"Self-Reflection completed: {len(findings)} findings, {len(improvements)} improvements")
        
        return report
    
    def save_report(self, report: ReflectionReport) -> str:
        """
        Saves the reflection report to the memory directory.
        
        Args:
            report: ReflectionReport to save
            
        Returns:
            Path to saved report file
        """
        os.makedirs(REFLECTION_OUTPUT_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reflection_report_{timestamp}.md"
        filepath = os.path.join(REFLECTION_OUTPUT_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())
        
        # Also save JSON version for programmatic access
        json_filepath = filepath.replace(".md", ".json")
        with open(json_filepath, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        
        log_event(f"Reflection report saved to {filepath}")
        return filepath
    
    def _load_persona(self) -> Optional[Dict[str, Any]]:
        """Loads and caches persona.yaml content."""
        if self._persona_cache is not None:
            return self._persona_cache
        
        if not os.path.exists(self.persona_path):
            return None
        
        try:
            with open(self.persona_path, "r", encoding="utf-8") as f:
                self._persona_cache = yaml.safe_load(f)
            return self._persona_cache
        except Exception as e:
            log_event(f"Failed to load persona.yaml: {e}")
            return None


# Convenience function for tool invocation
def reflect(count: int = 10, save: bool = True) -> str:
    """
    Performs self-reflection on recent interactions.
    
    Args:
        count: Number of recent interactions to analyze
        save: Whether to save the report to file
        
    Returns:
        Markdown summary of the reflection report
    """
    import core.shared_state as shared_state
    
    engine = ReflectionEngine(love_state=shared_state.love_state)
    report = engine.generate_report(count)
    
    if save:
        engine.save_report(report)
    
    return report.to_markdown()
