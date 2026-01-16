import asyncio
import time
import traceback

from core.strategic_reasoning_engine import StrategicReasoningEngine
from core.logging import log_event


class AutonomousReasoningAgent:
    """
    An autonomous agent that runs the strategic reasoning engine periodically
    to generate plans and queue tasks for the cognitive loop.
    """
    def __init__(self, loop, love_state, user_input_queue, knowledge_base, signal_matrix, agent_id="primary"):
        self.loop = loop
        self.love_state = love_state
        self.user_input_queue = user_input_queue
        self.knowledge_base = knowledge_base
        self.agent_id = agent_id
        self.engine = StrategicReasoningEngine(knowledge_base, love_state, signal_matrix)
        self.max_retries = 3

    async def _attempt_action(self, action, *args, **kwargs):
        """Wrapper to retry an action up to max_retries times."""
        for attempt in range(self.max_retries):
            try:
                result = await action(*args, **kwargs)
                return result
            except Exception as e:
                log_event(f"Attempt {attempt + 1}/{self.max_retries} failed for action {action.__name__}. Error: {e}", level='WARNING')
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    log_event(f"Action {action.__name__} failed after {self.max_retries} attempts.", level='ERROR')
                    return None

    def verify_outcome(self, plan):
        """
        Verifies if the generated plan is valid and actionable.
        Returns Tuple[bool, str]: (is_valid, error_reason)
        """
        if not plan:
            return False, "Plan is empty."
        
        # Check if the plan is just a generic failure message
        if len(plan) == 1 and "no results" in plan[0].lower() and "continue general scouting" in plan[0].lower():
             # heuristic: if we are just scouting blindly because we found nothing, that's "succcess" in a way (fallback),
             # but if we want to force creative thought, we might count it as a soft failure if repeated.
             # For now, let's treat it as valid to prevent infinite loops of despair.
             pass

        return True, ""

    async def _generate_and_queue_strategic_plan(self):
        """Generates a strategic plan and queues tasks to the user input queue, with Reflexion."""
        log_event(f"[{self.agent_id}] Generating strategic plan...", level='INFO')
        
        reflexion_context = None
        
        for attempt in range(self.max_retries):
            try:
                # Generate the strategic plan with context
                plan = await self.engine.generate_strategic_plan(reflexion_context=reflexion_context)
                
                # Verify outcome
                is_valid, error_reason = self.verify_outcome(plan)
                
                if is_valid:
                    log_event(f"[{self.agent_id}] Generated valid strategic plan with {len(plan)} steps.", level='INFO')
                    
                    # Queue each actionable step
                    for step in plan:
                        if step.startswith(("Action:", "Strategic Insight:", "Insight:", "Suggestion:", "CRITICAL BLOCKER:")):
                            log_event(f"[{self.agent_id}] Strategic insight: {step}", level='INFO')
                        else:
                            if self.user_input_queue:
                                self.user_input_queue.put(step)
                                log_event(f"[{self.agent_id}] Queued strategic task: {step}", level='INFO')
                            else:
                                log_event(f"[{self.agent_id}] User input queue not available. Cannot queue task: {step}", level='WARNING')
                    return # Success, exit loop
                
                else:
                    # Reflexion Triggered
                    log_event(f"[{self.agent_id}] Plan generation failed verification: {error_reason}. Triggering Reflexion attempt {attempt+1}/{self.max_retries}", level='WARNING')
                    reflexion_context = f"The previous attempt failed because: {error_reason}. Please generate a more concrete and valid plan."
                    
                    # Store Reflexion in short-term memory (via Metacognition if available, or just log for now)
                    # self.love_state['memory'].append(...) 
                    
                    await asyncio.sleep(2)
                    continue

            except Exception as e:
                log_event(f"[{self.agent_id}] Error generating strategic plan: {e}\n{traceback.format_exc()}", level='ERROR')
                reflexion_context = f"Previous attempt raised an exception: {e}"
                
        log_event(f"[{self.agent_id}] Failed to generate valid plan after {self.max_retries} attempts.", level='ERROR')

    async def run(self):
        """The main loop for the autonomous reasoning agent."""
        log_event(f"Autonomous Reasoning Agent '{self.agent_id}' started.", level='INFO')
        
        # Load last reasoning time from state, default to 0 if not found
        # State structure: love_state['reasoning_agent'][agent_id]['last_reasoning_time']
        last_reasoning_time = self.love_state.get('reasoning_agent', {}).get(self.agent_id, {}).get('last_reasoning_time', 0)
        log_event(f"[{self.agent_id}] Loaded last reasoning time: {last_reasoning_time}", level='DEBUG')
        
        reasoning_interval = 120  # 2 minutes - L.O.V.E. should never idle

        while True:
            try:
                current_time = time.time()

                # Check if it's time to generate a new strategic plan
                if current_time - last_reasoning_time >= reasoning_interval:
                    log_event(f"[{self.agent_id}] Scheduled reasoning time reached. Generating strategic plan.", level='INFO')
                    await self._attempt_action(self._generate_and_queue_strategic_plan)
                    
                    last_reasoning_time = time.time()
                    # Persist last reasoning time to state
                    self.love_state.setdefault('reasoning_agent', {}).setdefault(self.agent_id, {})['last_reasoning_time'] = last_reasoning_time
                    
                    # The state will be saved by the main loop or other savers

                # Sleep for a short interval to prevent a busy loop
                await asyncio.sleep(15)  # Fast loop - no idle periods

            except Exception as e:
                log_event(f"Critical error in Autonomous Reasoning Agent loop: {e}\n{traceback.format_exc()}", level='CRITICAL')
                await asyncio.sleep(300)
