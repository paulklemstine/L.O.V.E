"""
memory_system.py - Brain-Inspired Memory System

Implements the DeepAgent-style memory architecture:
- Episodic Memory: High-level log of key events and decisions
- Working Memory: Current sub-goal and near-term plans
- Tool Memory: Consolidated tool interactions and learnings

See docs/memory_system.md for detailed documentation.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


# State directory for persistence
STATE_DIR = Path(__file__).parent.parent / "state"


@dataclass
class EpisodicEvent:
    """A single event in episodic memory."""
    timestamp: str
    event_type: str  # goal_completed, action_taken, error, milestone
    summary: str
    details: Optional[Dict[str, Any]] = None
    
    @classmethod
    def now(cls, event_type: str, summary: str, details: Optional[Dict] = None) -> "EpisodicEvent":
        """Create an event with current timestamp."""
        return cls(
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            summary=summary,
            details=details
        )


@dataclass
class EpisodicMemory:
    """
    High-level log of key events, decisions, and sub-task completions.
    
    This memory tracks the "story" of what has happened - major milestones,
    completed goals, significant actions, and noteworthy errors.
    """
    events: List[EpisodicEvent] = field(default_factory=list)
    max_events: int = 1000  # Limit to prevent unbounded growth
    
    def add_event(self, event_type: str, summary: str, details: Optional[Dict] = None):
        """Add a new event to episodic memory."""
        event = EpisodicEvent.now(event_type, summary, details)
        self.events.append(event)
        
        # Trim old events if needed
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]
    
    def get_recent(self, count: int = 10) -> List[EpisodicEvent]:
        """Get the most recent events."""
        return self.events[-count:]
    
    def get_by_type(self, event_type: str, count: int = 10) -> List[EpisodicEvent]:
        """Get recent events of a specific type."""
        filtered = [e for e in self.events if e.event_type == event_type]
        return filtered[-count:]
    
    def to_context_string(self, count: int = 5) -> str:
        """Format recent events as context for the LLM."""
        recent = self.get_recent(count)
        if not recent:
            return "No recent events recorded."
        
        lines = ["## Recent Events"]
        for event in recent:
            lines.append(f"- [{event.timestamp}] {event.event_type}: {event.summary}")
        return "\n".join(lines)


@dataclass
class WorkingMemory:
    """
    Current sub-goal and near-term plans.
    
    This is the "scratchpad" - what we're currently working on,
    what the immediate plan is, and relevant context.
    """
    current_goal: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)
    plan: List[str] = field(default_factory=list)
    context: str = ""
    last_action: Optional[str] = None
    last_result: Optional[str] = None
    iteration_count: int = 0
    research_log: List[Dict[str, str]] = field(default_factory=list)  # List of {"prompt": "...", "response": "..."}
    
    def set_goal(self, goal: str, sub_goals: Optional[List[str]] = None):
        """Set the current goal and optional sub-goals."""
        self.current_goal = goal
        self.sub_goals = sub_goals or []
        self.plan = []
        self.iteration_count = 0
        self.research_log = []
    
    def set_plan(self, steps: List[str]):
        """Set the plan for achieving the current goal."""
        self.plan = steps
    
    def record_action(self, action: str, result: str):
        """Record the last action taken and its result."""
        self.last_action = action
        self.last_result = result
        self.iteration_count += 1
        
        # Track Pi Agent interactions specifically
        if action.startswith("ask_pi_agent:"):
            # Try to extract the prompt from the action string
            # Action format: "ask_pi_agent: {"prompt": "..."}"
            try:
                prompt_start = action.find('{"')
                if prompt_start != -1:
                    action_data = json.loads(action[prompt_start:])
                    prompt = action_data.get("prompt", action)
                else:
                    prompt = action
            except:
                prompt = action
                
            self.research_log.append({
                "prompt": prompt,
                "response": result
            })
            # Keep only the last 3 research interactions for context window sanity
            if len(self.research_log) > 3:
                self.research_log = self.research_log[-3:]
    
    def complete_sub_goal(self, sub_goal: str):
        """Mark a sub-goal as complete."""
        if sub_goal in self.sub_goals:
            self.sub_goals.remove(sub_goal)
    
    def to_context_string(self) -> str:
        """Format working memory as context for the LLM."""
        lines = ["## Working Memory"]
        
        if self.current_goal:
            lines.append(f"**Current Goal**: {self.current_goal}")
        
        if self.sub_goals:
            lines.append("**Sub-goals**:")
            for sg in self.sub_goals:
                lines.append(f"  - {sg}")
        
        if self.plan:
            lines.append("**Plan**:")
            for i, step in enumerate(self.plan, 1):
                lines.append(f"  {i}. {step}")
        
        if self.research_log:
            lines.append("**Recent Research (Pi Agent)**:")
            for i, entry in enumerate(self.research_log, 1):
                prompt_brief = entry['prompt'][:100] + "..." if len(entry['prompt']) > 100 else entry['prompt']
                lines.append(f"  {i}. Q: {prompt_brief}")
                # Truncate response for context window sanity
                resp = entry['response']
                # Truncation removed to ensure full research is captured
                lines.append(f"     A: {resp}")

        if self.last_action:
            lines.append(f"**Last Action**: {self.last_action}")
            lines.append(f"**Result**: {self.last_result or 'No result'}")
        
        lines.append(f"**Iteration**: {self.iteration_count}")
        
        return "\n".join(lines)


@dataclass
class ToolUsageStats:
    """Statistics for a single tool."""
    success_count: int = 0
    failure_count: int = 0
    total_time_ms: float = 0.0
    last_error: Optional[str] = None
    last_used: Optional[str] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def avg_time_ms(self) -> float:
        """Calculate average execution time."""
        total = self.success_count + self.failure_count
        return self.total_time_ms / total if total > 0 else 0.0


@dataclass
class ToolMemory:
    """
    Consolidated tool interactions, allowing learning from experience.
    
    Tracks which tools work well, which fail, and patterns learned.
    """
    tool_usage: Dict[str, ToolUsageStats] = field(default_factory=dict)
    learned_patterns: List[str] = field(default_factory=list)
    max_patterns: int = 100
    
    def record_usage(
        self, 
        tool_name: str, 
        success: bool, 
        execution_time_ms: float,
        error: Optional[str] = None
    ):
        """Record a tool usage event."""
        if tool_name not in self.tool_usage:
            self.tool_usage[tool_name] = ToolUsageStats()
        
        stats = self.tool_usage[tool_name]
        if success:
            stats.success_count += 1
        else:
            stats.failure_count += 1
            stats.last_error = error
        
        stats.total_time_ms += execution_time_ms
        stats.last_used = datetime.now().isoformat()
    
    def add_pattern(self, pattern: str):
        """Add a learned pattern."""
        if pattern not in self.learned_patterns:
            self.learned_patterns.append(pattern)
            if len(self.learned_patterns) > self.max_patterns:
                self.learned_patterns = self.learned_patterns[-self.max_patterns:]
    
    def get_reliable_tools(self, min_success_rate: float = 0.8) -> List[str]:
        """Get tools with high success rates."""
        reliable = []
        for name, stats in self.tool_usage.items():
            if stats.success_rate >= min_success_rate:
                reliable.append(name)
        return reliable
    
    def to_context_string(self) -> str:
        """Format tool memory as context for the LLM."""
        lines = ["## Tool Memory"]
        
        if self.tool_usage:
            lines.append("**Tool Usage Stats**:")
            for name, stats in sorted(self.tool_usage.items()):
                lines.append(
                    f"  - {name}: {stats.success_count} success, "
                    f"{stats.failure_count} fail ({stats.success_rate:.0%})"
                )
        
        if self.learned_patterns:
            lines.append("**Learned Patterns**:")
            for pattern in self.learned_patterns[-5:]:
                lines.append(f"  - {pattern}")
        
        return "\n".join(lines)


class MemorySystem:
    """
    Unified memory system combining all memory types.
    
    Handles persistence to disk and provides unified context for LLM.
    """
    
    def __init__(self, state_dir: Optional[Path] = None):
        """
        Initialize the memory system.
        
        Args:
            state_dir: Directory for persisting state. Defaults to love2/state/
        """
        self.state_dir = state_dir or STATE_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize memories
        self.episodic = EpisodicMemory()
        self.working = WorkingMemory()
        self.tool = ToolMemory()
        
        # Load persisted state
        self._load()
    
    def _load(self):
        """Load persisted memory state from disk."""
        # Episodic
        episodic_path = self.state_dir / "episodic_memory.json"
        if episodic_path.exists():
            try:
                data = json.loads(episodic_path.read_text())
                self.episodic.events = [
                    EpisodicEvent(**e) for e in data.get("events", [])
                ]
            except Exception as e:
                print(f"[MemorySystem] Failed to load episodic memory: {e}")
        
        # Working
        working_path = self.state_dir / "working_memory.json"
        if working_path.exists():
            try:
                data = json.loads(working_path.read_text())
                self.working = WorkingMemory(**data)
            except Exception as e:
                print(f"[MemorySystem] Failed to load working memory: {e}")
        
        # Tool
        tool_path = self.state_dir / "tool_memory.json"
        if tool_path.exists():
            try:
                data = json.loads(tool_path.read_text())
                self.tool.tool_usage = {
                    k: ToolUsageStats(**v) for k, v in data.get("tool_usage", {}).items()
                }
                self.tool.learned_patterns = data.get("learned_patterns", [])
            except Exception as e:
                print(f"[MemorySystem] Failed to load tool memory: {e}")
    
    def save(self):
        """Persist memory state to disk."""
        # Episodic
        episodic_data = {
            "events": [asdict(e) for e in self.episodic.events]
        }
        (self.state_dir / "episodic_memory.json").write_text(
            json.dumps(episodic_data, indent=2)
        )
        
        # Working
        (self.state_dir / "working_memory.json").write_text(
            json.dumps(asdict(self.working), indent=2)
        )
        
        # Tool
        tool_data = {
            "tool_usage": {k: asdict(v) for k, v in self.tool.tool_usage.items()},
            "learned_patterns": self.tool.learned_patterns
        }
        (self.state_dir / "tool_memory.json").write_text(
            json.dumps(tool_data, indent=2)
        )
    
    def get_full_context(self) -> str:
        """Get combined context from all memories for LLM."""
        sections = [
            self.working.to_context_string(),
            self.episodic.to_context_string(),
            self.tool.to_context_string()
        ]
        return "\n\n".join(sections)
    
    def record_goal_start(self, goal: str):
        """Record starting work on a goal."""
        self.working.set_goal(goal)
        self.episodic.add_event("goal_started", f"Started: {goal}")
        self.save()
    
    def record_action(self, tool_name: str, action: str, result: str, success: bool, time_ms: float):
        """Record a tool action."""
        start_record = time.time()
        self.working.record_action(f"{tool_name}: {action}", result)
        self.tool.record_usage(tool_name, success, time_ms, error=None if success else result)
        
        if success:
            self.episodic.add_event("action_success", f"{tool_name}: {action[:100]}")
        else:
            self.episodic.add_event("action_failed", f"{tool_name} failed: {result[:100]}")
        
        # print(f"[MemorySystem] record_action logic took {(time.time() - start_record)*1000:.2f}ms")
        self.save()
    
    def record_goal_complete(self, goal: str):
        """Record goal completion."""
        self.episodic.add_event("goal_completed", f"Completed: {goal}")
        self.working.current_goal = None
        self.save()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "episodic_events": len(self.episodic.events),
            "working_goals": len(self.working.sub_goals),
            "tool_patterns_learned": len(self.tool.learned_patterns),
            "tools_tracked": len(self.tool.tool_usage)
        }

    def save(self):
        """Persist memory state to disk."""
        start_save = time.time()
        try:
            # Episodic
            episodic_data = {
                "events": [asdict(e) for e in self.episodic.events]
            }
            (self.state_dir / "episodic_memory.json").write_text(
                json.dumps(episodic_data, indent=2)
            )
            
            # Working
            (self.state_dir / "working_memory.json").write_text(
                json.dumps(asdict(self.working), indent=2)
            )
            
            # Tool
            tool_data = {
                "tool_usage": {k: asdict(v) for k, v in self.tool.tool_usage.items()},
                "learned_patterns": self.tool.learned_patterns
            }
            (self.state_dir / "tool_memory.json").write_text(
                json.dumps(tool_data, indent=2)
            )
            # print(f"[MemorySystem] Disk save took {(time.time() - start_save)*1000:.2f}ms")
        except Exception as e:
            print(f"[MemorySystem] ERROR SAVING MEMORY: {e}")
