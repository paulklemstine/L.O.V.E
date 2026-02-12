"""
Master Plan Manager

Parses the Master Plan markdown file and manages the structured goal state.
"""

import os
import json
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger("MasterPlanManager")

MASTER_PLAN_MD = "goals/master_plan_autonomous.md"
MASTER_PLAN_JSON = "goals/master_plan.json"

@dataclass
class PlanTask:
    id: str
    title: str
    description: str = ""
    status: str = "pending" # pending, active, completed
    dependencies: List[str] = None

@dataclass
class PlanFeature:
    id: str
    title: str
    description: str = ""
    tasks: List[PlanTask] = None
    status: str = "pending"

@dataclass
class PlanEpic:
    id: str
    title: str
    description: str = ""
    features: List[PlanFeature] = None
    status: str = "pending"

class MasterPlanManager:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.md_path = os.path.join(root_dir, MASTER_PLAN_MD)
        self.json_path = os.path.join(root_dir, MASTER_PLAN_JSON)
        self.plan: List[PlanEpic] = []

    def parse_markdown(self) -> bool:
        """Parses the markdown file and populates the plan structure."""
        if not os.path.exists(self.md_path):
            logger.error(f"Master plan markdown not found at {self.md_path}")
            return False

        with open(self.md_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        current_epic = None
        current_feature = None
        
        # Simple state machine parser
        # Assuming structure:
        # ## Epics
        # 1. **Epic Title**
        #    - **Goal:** ...
        # ## Features
        # 1. **Feature Title**
        #    - ...
        # ## Tasks
        # 1. **Task Title**
        
        # ACTUALLY, the pi-agent output format varies. 
        # The last output had separate sections for Epics, Features, and Tasks lists.
        # We need to try to link them or just treat them as a flat list if linking is hard.
        
        # Let's try to capture the high-level items as "Goals" for the DeepLoop.
        
        epics = []
        features = []
        tasks = []
        
        mode = None # "epics", "features", "tasks"
        
        for line in lines:
            line = line.strip()
            if "## Epics" in line:
                mode = "epics"
                continue
            elif "## Features" in line:
                mode = "features"
                continue
            elif "## Tasks" in line:
                mode = "tasks"
                continue
            elif line.startswith("## "):
                mode = None
                continue
                
            if not line:
                continue
                
            # Parse numbered items: "1. **Title**"
            match = re.match(r"^\d+\.\s+\*\*(.+?)\*\*", line)
            if match:
                title = match.group(1)
                if mode == "epics":
                    epics.append({"title": title, "type": "epic"})
                elif mode == "features":
                    features.append({"title": title, "type": "feature"})
                elif mode == "tasks":
                    tasks.append({"title": title, "type": "task"})
        
        # Construct a simple unified plan for now
        # We will treat Epics as high priority, Features as medium, Tasks as immediate
        
        structured_plan = {
            "epics": epics,
            "features": features,
            "tasks": tasks,
            "updated_at": str(os.path.getmtime(self.md_path))
        }
        
        self._save_json(structured_plan)
        return True

    def _save_json(self, data: Dict[str, Any]):
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        with open(self.json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved structured plan to {self.json_path}")

    def load_plan(self) -> Dict[str, Any]:
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def get_goals_for_extractor(self) -> List[Dict[str, Any]]:
        """Returns a list of goals formatted for PersonaGoalExtractor."""
        plan = self.load_plan()
        goals = []
        
        # Add Tasks (Highest Priority for execution)
        for i, task in enumerate(plan.get("tasks", [])):
            goals.append({
                "text": f"Task: {task['title']}",
                "priority": 1, # High priority to start executing
                "category": "master_plan_task",
                "status": task.get("status", "pending")
            })
            
        # Add Features (Medium)
        for feature in plan.get("features", []):
            goals.append({
                "text": f"Feature: {feature['title']}",
                "priority": 2,
                "category": "master_plan_feature",
                "status": feature.get("status", "pending")
            })
            
        # Epics (Strategic)
        for epic in plan.get("epics", []):
            goals.append({
                "text": f"Epic: {epic['title']}",
                "priority": 3,
                "category": "master_plan_epic",
                "status": epic.get("status", "pending")
            })
            
        return goals

    def update_task_status(self, title: str, status: str) -> bool:
        """Updates the status of a task by title."""
        plan = self.load_plan()
        updated = False
        
        for task in plan.get("tasks", []):
            if task["title"] == title or f"Task: {task['title']}" == title:
                task["status"] = status
                updated = True
                break
                
        if updated:
            self._save_json(plan)
            return True
        return False

    def add_checklist(self, parent_goal: str, subtasks: List[str]) -> bool:
        """Adds a list of subtasks to a goal."""
        plan = self.load_plan()
        
        # Find if it's an Epic, Feature or Task
        found = False
        
        # We'll just add them to the flat 'tasks' list for now but prefixed with the parent
        for sub in subtasks:
            # Check if it already exists
            exists = False
            for existing in plan.get("tasks", []):
                if existing["title"] == sub:
                    exists = True
                    break
            
            if not exists:
                plan.setdefault("tasks", []).append({
                    "title": sub,
                    "type": "task",
                    "status": "pending",
                    "parent": parent_goal
                })
                found = True
        
        if found:
            self._save_json(plan)
            return True
        return False

# Singleton
_manager = None

def get_master_plan_manager(root_dir: str = None) -> MasterPlanManager:
    global _manager
    if not _manager:
        if not root_dir:
            root_dir = os.getcwd()
        _manager = MasterPlanManager(root_dir)
    return _manager
