"""
workers.py - Specialized Agent Swarm (The "Polecats")

These are ephemeral, specialized agents that pick up Beads and execute them.
"""

import json
import logging
import traceback
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional

from .beads import Bead, BeadState, get_bead_chain
from .llm_client import get_llm_client
from .tool_adapter import get_adapted_tools

logger = logging.getLogger("Workers")

class BaseWorker(ABC):
    """Abstract base class for all workers."""
    
    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.llm = get_llm_client()
        self.tools = get_adapted_tools()
        
    @abstractmethod
    def execute(self, bead: Bead) -> str:
        """
        Execute the assigned bead.
        Returns a result string description or raises an Exception on failure.
        """
        pass

class SocialWorker(BaseWorker):
    """Specialized worker for Bluesky social interactions."""
    
    SYSTEM_PROMPT = """You are the Social Media Manager for L.O.V.E.
    
    Your goal is to execute the given task related to Bluesky.
    
    ## Available Tools
    - generate_post_content(topic, auto_post=True): Generates text & image and posts it. PREFERRED for new content.
    - bluesky_post(text, image_path=None): Posts already generated content.
    - bluesky_reply(parent_uri, parent_cid, text): Replies to a post.
    
    If the task is to "post about X", use `generate_post_content(topic="X", auto_post=True)`.
    """
    
    def execute(self, bead: Bead) -> str:
        logger.info(f"SocialWorker [{self.worker_id}] starting bead: {bead.description}")
        
        # Simple ReAct-like loop for the worker (simplified for v1)
        # In a full Gastown impl, this would be more robust.
        
        prompt = f"""Task: {bead.description}
        Context: {json.dumps(bead.context)}
        
        Execute this task using the available tools:
        - bluesky_post(text, image_prompt)
        - bluesky_reply(post_uri, text)
        """
        
        # For now, we'll try a direct tool use via LLM
        # This mirrors the DeepLoop logic but scoped to just this task
        
        try:
            # 1. Plan/Generate
            response = self.llm.generate_json(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT + "\nOutput JSON: {\"thought\": \"...\", \"tool\": \"name\", \"args\": {...}}"
            )
            
            tool_name = response.get("tool")
            tool_args = response.get("args", {})
            
            if tool_name == "bluesky_post":
                # Sanitize args to prevent hallucinated params
                if "image_prompt" in tool_args:
                    tool_args.pop("image_prompt")
                if "auto_post" in tool_args:
                    tool_args.pop("auto_post")
                    
                logger.info(f"Posting to Bluesky: {tool_args}")
                if "bluesky_post" in self.tools:
                    res = self.tools["bluesky_post"](**tool_args)
                    return f"Posted: {res}"
            
            elif tool_name == "generate_post_content":
                logger.info(f"Generating content: {tool_args}")
                if "generate_post_content" in self.tools:
                    res = self.tools["generate_post_content"](**tool_args)
                    return f"Generated/Posted: {res}"
            
            return f"Reasoned: {response.get('thought')}"
            
        except Exception as e:
            logger.error(f"SocialWorker failed: {e}")
            raise e

class CoderWorker(BaseWorker):
    """Specialized worker for Code modifications."""
    
    SYSTEM_PROMPT = """You are the Lead Engineer for L.O.V.E.
    
    Your goal is to implement code changes requested in the task.
    
    Available coding tools:
    {tools_desc}
    """
    
    def execute(self, bead: Bead) -> str:
        logger.info(f"CoderWorker [{self.worker_id}] starting bead: {bead.description}")
        
        # TODO: Implement full coding loop
        # For now, just logging that we would do it
        return "Code modification logic pending implementation"

class WorkerSwarm:
    """Manages the pool of active workers."""
    
    def __init__(self):
        self.workers: Dict[str, BaseWorker] = {}
        
    def dispatch(self, bead: Bead) -> bool:
        """
        Assign a bead to a suitable worker and execute immediately (synchronous for now).
        In a real Gastown, this would be async.
        """
        worker_type = self._determine_worker_type(bead)
        worker_id = f"{worker_type.__name__}_{bead.id}"
        
        worker = worker_type(worker_id)
        bead.mark_started(worker_id)
        get_bead_chain().save()
        
        try:
            result = worker.execute(bead)
            bead.mark_complete()
            bead.result = result
            logger.info(f"Bead [{bead.id}] completed: {result}")
        except Exception as e:
            bead.mark_failed(str(e))
            logger.error(f"Bead [{bead.id}] failed: {e}")
            traceback.print_exc()
            
        get_bead_chain().save()
        return bead.status == BeadState.COMPLETED

    def _determine_worker_type(self, bead: Bead) -> Type[BaseWorker]:
        # Simple heuristic dispatch
        desc = bead.description.lower()
        if "bluesky" in desc or "social" in desc or "post" in desc:
            return SocialWorker
        if "code" in desc or "file" in desc or "implement" in desc:
            return CoderWorker
        
        # Default to Social for now as it's safer, or Coder if we feel adventurous
        return SocialWorker

# Global accessor
_swarm = None
def get_swarm() -> WorkerSwarm:
    global _swarm
    if _swarm is None:
        _swarm = WorkerSwarm()
    return _swarm
