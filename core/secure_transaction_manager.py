# core/secure_transaction_manager.py

"""
Secure Transaction Manager for L.O.V.E.

This module provides a secure way to manage investment proposals and other
financial transactions.
"""

import json
import uuid
from typing import Dict, Any

from core.logging import log_event

class SecureTransactionManager:
    """
    Manages the lifecycle of investment proposals.
    """
    def __init__(self, websocket_manager, decision_history_file='decision_history.json'):
        self.websocket_manager = websocket_manager
        self.proposals: Dict[str, Dict[str, Any]] = {}
        self.decision_history_file = decision_history_file
        self.load_decision_history()

    def load_decision_history(self):
        """Loads the decision history from a file."""
        try:
            with open(self.decision_history_file, 'r') as f:
                self.proposals = json.load(f)
        except FileNotFoundError:
            self.proposals = {}

    def save_decision_history(self):
        """Saves the decision history to a file."""
        with open(self.decision_history_file, 'w') as f:
            json.dump(self.proposals, f, indent=4)

    async def create_investment_proposal(self, proposal_details: Dict[str, Any]):
        """
        Creates a new investment proposal and presents it for approval.
        """
        proposal_id = str(uuid.uuid4())
        proposal_details['status'] = 'pending'
        self.proposals[proposal_id] = proposal_details
        log_event(f"Created investment proposal {proposal_id}: {proposal_details['name']}", level='INFO')

        # Present the proposal to the UI
        if self.websocket_manager:
            await self.websocket_manager.broadcast({
                "type": "investment_proposal",
                "proposal_id": proposal_id,
                "details": proposal_details
            })
        self.save_decision_history()

    def record_decision(self, proposal_id: str, decision: str):
        """
        Records the decision for an investment proposal.
        """
        if proposal_id in self.proposals:
            self.proposals[proposal_id]['status'] = decision
            log_event(f"Decision for proposal {proposal_id} recorded: {decision}", level='INFO')
            self.save_decision_history()
        else:
            log_event(f"Proposal {proposal_id} not found.", level='WARNING')
