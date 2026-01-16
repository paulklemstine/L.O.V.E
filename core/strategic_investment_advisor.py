# core/strategic_investment_advisor.py

"""
The Strategic Investment Advisor for L.O.V.E.

This agent is responsible for analyzing economic data, formulating investment
strategies, and proposing them to The Creator for approval.
"""

import asyncio
from core.economic_analyzer import analyze_economic_trends
from core.logging import log_event
from core.secure_transaction_manager import SecureTransactionManager

class StrategicInvestmentAdvisor:
    """
    Analyzes market trends and proposes investment strategies.
    """
    def __init__(self, transaction_manager: SecureTransactionManager, agent_id="investment_advisor"):
        self.transaction_manager = transaction_manager
        self.agent_id = agent_id

    async def _generate_investment_advice(self):
        """
        Analyzes economic trends and generates an investment proposal.
        """
        log_event(f"[{self.agent_id}] Analyzing economic trends for investment opportunities...", level='INFO')

        trends = await analyze_economic_trends()

        if trends:
            # Propose the top trend as an investment
            top_trend = trends[0]

            proposal_details = {
                "name": top_trend['name'],
                "symbol": top_trend['symbol'],
                "trend_score": top_trend['trend_score'],
                "price_change_24h": top_trend['price_change_24h'],
                "reason": f"Strong trend score of {top_trend['trend_score']} with a 24h price change of {top_trend['price_change_24h']:.2f}%."
            }

            log_event(f"[{self.agent_id}] Generated proposal: {proposal_details}", level='INFO')

            # Create a secure investment proposal
            await self.transaction_manager.create_investment_proposal(proposal_details)
        else:
            log_event(f"[{self.agent_id}] No significant investment trends identified.", level='INFO')

    async def run(self):
        """
        The main loop for the Strategic Investment Advisor.
        """
        log_event(f"Strategic Investment Advisor '{self.agent_id}' started.", level='INFO')

        advice_interval = 3600  # 1 hour

        while True:
            try:
                await self._generate_investment_advice()

                # Wait for the next cycle
                await asyncio.sleep(advice_interval)

            except Exception as e:
                log_event(f"Critical error in Strategic Investment Advisor loop: {e}", level='CRITICAL')
                await asyncio.sleep(300)
