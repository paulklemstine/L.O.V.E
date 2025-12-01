from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class KeyEvent(BaseModel):
    step: int
    action: str
    outcome: str

class EpisodicMemory(BaseModel):
    task_description: str = Field(default="")
    key_events: List[KeyEvent] = Field(default_factory=list)

class WorkingMemory(BaseModel):
    current_subgoal: str = Field(default="")
    pending_tasks: List[str] = Field(default_factory=list)
    active_variables: Dict[str, Any] = Field(default_factory=dict)

class ToolUsage(BaseModel):
    tool_name: str
    success_rate: float
    effective_params: Dict[str, Any]

class ToolMemory(BaseModel):
    tools_used: List[ToolUsage] = Field(default_factory=list)
