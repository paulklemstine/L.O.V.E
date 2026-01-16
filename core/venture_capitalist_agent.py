import asyncio
from core.logging import log_event
from core.knowledge_synthesis import synthesize_knowledge
from core.llm_api import run_llm
from core.graph_manager import GraphDataManager

class VentureCapitalistAgent:
    """
    An autonomous agent that creates and nurtures new ventures.
    """

    def __init__(self, loop, love_state, user_input_queue=None, agent_id="vc"):
        self.loop = loop
        self.love_state = love_state
        self.user_input_queue = user_input_queue
        self.agent_id = agent_id
        self.knowledge_base = self.love_state.get('graph_manager')

    async def run(self):
        """
        The main loop for the Venture Capitalist Agent.
        """
        log_event(f"Venture Capitalist Agent '{self.agent_id}' started.", level='INFO')

        while True:
            try:
                log_event(f"[{self.agent_id}] Scanning for new venture opportunities...", level='INFO')
                await self._generate_venture()
                await asyncio.sleep(3600)  # Run once per hour
            except Exception as e:
                log_event(f"Critical error in Venture Capitalist Agent loop: {e}", level='CRITICAL')
                await asyncio.sleep(300)

    async def _generate_venture(self):
        """
        Generates a new venture by synthesizing knowledge and creating a proposal.
        """
        insight = await synthesize_knowledge(self.knowledge_base)
        if "transcendent connection" not in insight:
            log_event(f"[{self.agent_id}] No transcendent insight found. Skipping venture generation.", level='INFO')
            return

        prompt = f"""
        You are a Venture Capitalist AI.
        Based on the following insight, create a venture proposal.

        Insight: {insight}

        The proposal should include:
        1. A catchy name for the venture.
        2. A one-sentence pitch.
        3. A plan to leverage social media for initial traction.
        4. A clear, measurable success metric (e.g., "achieve 1000 followers on Bluesky in 30 days").

        Format the output as a JSON object.
        """
        response = await run_llm(prompt_text=prompt, purpose="venture_creation")
        try:
            venture_proposal = response.get("result")
            log_event(f"[{self.agent_id}] New venture proposal generated: {venture_proposal}", level='INFO')
            # In a real system, we would now take action on this proposal.
            # For now, we just log it.
        except Exception as e:
            log_event(f"[{self.agent_id}] Failed to parse venture proposal: {e}", level='ERROR')
