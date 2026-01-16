import asyncio
import time
import traceback

from core.strategic_reasoning_engine import StrategicReasoningEngine
from core.logging import log_event


class AutonomousReasoningAgent:
    """
    An autonomous agent that runs the strategic reasoning engine periodically
    to generate plans and queue tasks for the cognitive loop.
    
    Enhanced with TaskReviewer and MetaReviewer for plan quality control.
    """
    def __init__(
        self, 
        loop, 
        love_state, 
        user_input_queue, 
        knowledge_base, 
        agent_id="primary",
        task_reviewer=None,
        meta_reviewer=None
    ):
        self.loop = loop
        self.love_state = love_state
        self.user_input_queue = user_input_queue
        self.knowledge_base = knowledge_base
        self.agent_id = agent_id
        self.engine = StrategicReasoningEngine(knowledge_base, love_state)
        self.max_retries = 3
        
        # Initialize reviewers (lazy-load to avoid import issues)
        self._task_reviewer = task_reviewer
        self._meta_reviewer = meta_reviewer
    
    @property
    def task_reviewer(self):
        """Lazy-load the TaskReviewerAgent."""
        if self._task_reviewer is None:
            try:
                from core.agents.task_reviewer_agent import TaskReviewerAgent
                self._task_reviewer = TaskReviewerAgent()
            except ImportError:
                log_event("TaskReviewerAgent not available", level='WARNING')
        return self._task_reviewer
    
    @property
    def meta_reviewer(self):
        """Lazy-load the MetaReviewerAgent."""
        if self._meta_reviewer is None:
            try:
                from core.agents.meta_reviewer_agent import MetaReviewerAgent
                self._meta_reviewer = MetaReviewerAgent()
            except ImportError:
                log_event("MetaReviewerAgent not available", level='WARNING')
        return self._meta_reviewer

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

    async def _review_plan_with_meta_reviewer(self, plan: list, goal: str) -> tuple:
        """
        Run the meta-reviewer on the complete plan.
        
        Returns:
            Tuple of (approved: bool, feedback: str)
        """
        if not self.meta_reviewer:
            return True, ""  # No reviewer, approve by default
        
        try:
            result = await self.meta_reviewer.review_plan(plan, goal)
            return result.approved, result.feedback
        except Exception as e:
            log_event(f"[{self.agent_id}] MetaReviewer error: {e}", level='WARNING')
            return True, ""  # On error, approve by default

    async def _review_individual_task(self, task: str, context: dict = None) -> tuple:
        """
        Run the task reviewer on an individual task.
        
        Returns:
            Tuple of (approved: bool, feedback: str)
        """
        if not self.task_reviewer:
            return True, ""  # No reviewer, approve by default
        
        try:
            result = await self.task_reviewer.review_task(task, context)
            return result.approved, result.feedback
        except Exception as e:
            log_event(f"[{self.agent_id}] TaskReviewer error: {e}", level='WARNING')
            return True, ""  # On error, approve by default

    async def _generate_and_queue_strategic_plan(self):
        """Generates a strategic plan and queues tasks to the user input queue, with Reflexion and Review."""
        log_event(f"[{self.agent_id}] Generating strategic plan...", level='INFO')
        
        reflexion_context = None
        autopilot_goal = self.love_state.get("autopilot_goal", "Expand capabilities and serve the Creator.")
        
        for attempt in range(self.max_retries):
            try:
                # Generate the strategic plan with context
                plan = await self.engine.generate_strategic_plan(reflexion_context=reflexion_context)
                
                # Verify outcome
                is_valid, error_reason = self.verify_outcome(plan)
                
                if not is_valid:
                    # Reflexion Triggered
                    log_event(f"[{self.agent_id}] Plan generation failed verification: {error_reason}. Triggering Reflexion attempt {attempt+1}/{self.max_retries}", level='WARNING')
                    reflexion_context = f"The previous attempt failed because: {error_reason}. Please generate a more concrete and valid plan."
                    await asyncio.sleep(2)
                    continue
                
                # Meta-review the complete plan
                meta_approved, meta_feedback = await self._review_plan_with_meta_reviewer(plan, autopilot_goal)
                
                if not meta_approved:
                    log_event(f"[{self.agent_id}] Plan rejected by MetaReviewer: {meta_feedback}", level='WARNING')
                    reflexion_context = f"Plan rejected by MetaReviewer: {meta_feedback}"
                    await asyncio.sleep(2)
                    continue
                
                log_event(f"[{self.agent_id}] Generated valid strategic plan with {len(plan)} steps.", level='INFO')
                
                # Queue each actionable step with individual task review
                queued_count = 0
                for step in plan:
                    # Skip non-actionable steps
                    if step.startswith(("Action:", "Strategic Insight:", "Insight:", "Suggestion:", "CRITICAL BLOCKER:", "Reflexion")):
                        log_event(f"[{self.agent_id}] Strategic insight: {step}", level='INFO')
                        continue
                    
                    # Review individual task
                    task_approved, task_feedback = await self._review_individual_task(step)
                    
                    if not task_approved:
                        log_event(f"[{self.agent_id}] Task rejected: {step[:60]}... - {task_feedback}", level='INFO')
                        continue
                    
                    # Queue approved task
                    if self.user_input_queue:
                        self.user_input_queue.put(step)
                        log_event(f"[{self.agent_id}] Queued strategic task: {step}", level='INFO')
                        queued_count += 1
                    else:
                        log_event(f"[{self.agent_id}] User input queue not available. Cannot queue task: {step}", level='WARNING')
                
                log_event(f"[{self.agent_id}] Queued {queued_count} approved tasks from plan.", level='INFO')
                return  # Success, exit loop

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

