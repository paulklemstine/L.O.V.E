from core.agents.specialist_agent import SpecialistAgent
from typing import Dict, Any
from core.prompt_registry import PromptRegistry

class MetacognitionAgent(SpecialistAgent):
    """
    A specialist agent dedicated to observing the cognitive processes of the L.O.V.E. organism.
    """

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager

    async def execute_task(self, task_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a cognitive event and records it as a memory.

        Args:
            task_details: A dictionary containing the cognitive event payload.

        Returns:
            A dictionary confirming the status of the operation.
        """
        event_type = task_details.get('event_type')
        if not event_type:
            return {'status': 'failure', 'result': 'Missing event_type in task_details'}

        try:
            formatted_string = self._format_event(task_details)
            
            # MEMORY LOGGING: Track MetacognitionAgent calls
            print(f"\n[METACOGNITION] Recording cognitive event: {event_type}")
            print(f"[METACOGNITION]   Event details: {str(task_details)[:200]}...")
            print(f"[METACOGNITION]   Calling add_episode...")
            
            await self.memory_manager.add_episode(formatted_string)
            
            print(f"[METACOGNITION] SUCCESS: Cognitive event '{event_type}' recorded.")
            return {'status': 'success', 'result': f"Cognitive event '{event_type}' recorded."}
        except Exception as e:
            print(f"[METACOGNITION] ERROR: Failed to record cognitive event: {e}")
            return {'status': 'failure', 'result': f"Failed to record cognitive event: {e}"}


    def _format_event(self, task_details: Dict[str, Any]) -> str:
        """Formats the cognitive event into a structured string."""
        event_type = task_details['event_type']

        if event_type == 'plan_generated':
            goal = task_details.get('goal', 'N/A')
            plan_steps = len(task_details.get('plan', []))
            return f"Cognitive Event: Plan Generated | Goal: '{goal}' | Plan Steps: {plan_steps}"

        elif event_type == 'agent_dispatch':
            agent_name = task_details.get('agent_name', 'N/A')
            task = task_details.get('task', 'N/A')
            return f"Cognitive Event: Agent Dispatched | Agent: '{agent_name}' | Task: '{task}'"

        elif event_type == 'agent_result':
            agent_name = task_details.get('agent_name', 'N/A')
            result = task_details.get('result', 'N/A')
            return f"Cognitive Event: Agent Result | Agent: '{agent_name}' | Result: '{result}'"

        elif event_type == 'json_repair':
            malformed = task_details.get('malformed_text', 'N/A')
            error = task_details.get('error', 'N/A')
            repaired = task_details.get('repaired_text', 'N/A')
            return f"Cognitive Event: JSON Repair | Error: '{error}' | Malformed: '{malformed[:50]}...' | Repaired: '{repaired[:50]}...'"

        elif event_type == 'post_mortem':
            return f"[PostMortem] Failure: {task_details.get('failure_reason', 'N/A')} | Root Cause: {task_details.get('root_cause', 'N/A')} | Proposed Fix: {task_details.get('correction', 'N/A')}"

        elif event_type == 'prompt_evolution':
            repo_id = task_details.get('repo_id', 'N/A')
            result = task_details.get('result', 'N/A')
            prompt_preview = task_details.get('prompt_preview', '')[:50]
            return f"Cognitive Event: Prompt Evolution | Repo: '{repo_id}' | Result: '{result}' | Preview: '{prompt_preview}...'"

        else:
            return f"Cognitive Event: Unknown Event | Details: {task_details}"

    async def record_repair_event(self, malformed_text: str, error_context: str, repaired_text: str = None):
        """
        Records a JSON repair event.
        """
        event_payload = {
            'event_type': 'json_repair',
            'malformed_text': malformed_text,
            'error': error_context,
            'repaired_text': repaired_text
        }
        return await self.execute_task(event_payload)

    async def record_post_mortem(self, failure_reason: str, root_cause: str, correction: str):
        """
        Records a post-mortem of a critical failure.
        """
        event_payload = {
            'event_type': 'post_mortem',
            'failure_reason': failure_reason,
            'root_cause': root_cause,
            'correction': correction
        }
        return await self.execute_task(event_payload)

    async def push_evolution_prompt(self, repo_id: str, prompt_content: str):
        """
        Pushes an optimized prompt to the LangChain Hub.
        """
        registry = PromptRegistry()
        success = registry.push_to_hub(repo_id, prompt_content)
        result = "Success" if success else "Failed"
        return await self.execute_task({
            'event_type': 'prompt_evolution',
            'repo_id': repo_id,
            'result': result
        })

    async def perform_metacognition_cycle(self, deep_agent_engine=None):
        """
        Reflects on the system state and proposes/executes maintenance actions.
        """
        from core.reflection.self_model import generate_self_symbol, get_context_injection
        from core.llm_api import run_llm
        from core.logging import log_event
        import json

        log_event("Starting Metacognition Cycle...", "INFO")
        
        symbol = generate_self_symbol()
        context = get_context_injection()
        
        prompt = f"""
        You are the Metacognition Agent for L.O.V.E.
        Your goal is to inspect the system's current state (The Self-Symbol) and ensure optimal operation.
        
        {context}
        
        Review the active agents, API health, and capabilities.
        Are there any obvious problems? (e.g., API unavailable, Agent crashed/unavailable).
        
        Respond with a JSON object:
        {{
            "status": "nominal" | "degraded" | "critical",
            "observation": "Brief summary of state",
            "action_needed": "None" | "Description of action",
            "action_tool": "Tool name to call (optional)",
            "action_args": {{ "arg": "value" }} (optional),
            "internal_thought": "Your internal reasoning process"
        }}
        """
        
        try:
             response_dict = await run_llm(
                 prompt_text=prompt,
                 purpose="reasoning",
                 deep_agent_instance=deep_agent_engine,
                 response_schema={
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "observation": {"type": "string"},
                        "action_needed": {"type": "string"},
                        "action_tool": {"type": "string"},
                        "action_args": {"type": "object"},
                        "internal_thought": {"type": "string"}
                    },
                    "required": ["status", "observation", "action_needed", "internal_thought"]
                }
             )
             
             # run_llm returns a dict with 'result' containing text or dict if schema used?
             # Actually run_llm usually returns {'result': ..., 'prompt_cid': ...}
             # If schema is used, 'result' might be the parsed dict.
             
             result_data = response_dict.get('result')
             if isinstance(result_data, str):
                 try:
                     # Attempt to clean markdown json blocks if present
                     clean_text = result_data.replace('```json', '').replace('```', '').strip()
                     import json
                     analysis = json.loads(clean_text)
                 except:
                     analysis = {"status": "unknown", "observation": result_data}
             elif isinstance(result_data, dict):
                 analysis = result_data
             else:
                 analysis = {}

             log_event(f"Metacognition Analysis: {analysis.get('status')} - {analysis.get('observation')}", "INFO")
             
             # Record this event
             await self.execute_task({
                 'event_type': 'metacognition_cycle', 
                 'analysis': analysis
             })
             
             return analysis

        except Exception as e:
            log_event(f"Metacognition cycle failed: {e}", "ERROR")
            return None
