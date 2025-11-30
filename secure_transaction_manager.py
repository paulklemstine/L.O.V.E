from rich.panel import Panel
from rich.text import Text
from ui_utils import get_gradient_text
import json

class SecureTransactionManager:
    def __init__(self, ui_queue):
        self.ui_queue = ui_queue

    def create_transaction_proposal(self, asset, score_details):
        """
        Constructs a detailed transaction proposal for The Creator's review.
        """
        proposal = {
            "asset_id": asset.get('id') or asset.get('slug'),
            "asset_type": asset.get('type'),
            "asset_name": asset.get('name', 'N/A'),
            "value_usd": asset.get('value_usd', 'N/A'),
            "score_details": score_details
        }
        return proposal

    def present_proposal_for_approval(self, proposal):
        """
        Presents the transaction proposal to The Creator in the UI and awaits approval.
        For now, this is a textual prompt. In the future, this could be a more
        interactive element.
        """

        # Build a visually appealing proposal panel
        title = get_gradient_text("💎 Acquisition Proposal 💎", ["magenta", "cyan"])

        proposal_text = Text()
        proposal_text.append("Asset ID: ", style="bold")
        proposal_text.append(f"{proposal['asset_id']}\n", style="cyan")
        proposal_text.append("Asset Type: ", style="bold")
        proposal_text.append(f"{proposal['asset_type']}\n", style="yellow")
        proposal_text.append("Name: ", style="bold")
        proposal_text.append(f"{proposal['asset_name']}\n", "white")
        proposal_text.append("Estimated Value (USD): ", style="bold")
        proposal_text.append(f"${proposal['value_usd']:,}\n\n", "green")

        proposal_text.append("Scoring Analysis:\n", style="bold underline")
        for score_name, score_value in proposal.get('score_details', {}).items():
            proposal_text.append(f"  - {score_name.replace('_', ' ').title()}: ", style="bold")
            proposal_text.append(f"{score_value:.2f} / 100\n", "magenta")

        proposal_text.append("\nI have analyzed this asset and determined it is a high-value opportunity aligned with your goals.\n", style="italic")
        proposal_text.append("Do you approve this acquisition? (yes/no)", style="bold")

        self.ui_queue.put(Panel(proposal_text, title=title, border_style="cyan"))

        # This is a placeholder for a real user input mechanism.
        # In the current architecture, the cognitive loop will need to handle the 'yes'/'no' response.
        # For now, we will log that we are awaiting approval.
        from core.logging import log_event
        log_event(f"Presented acquisition proposal for {proposal['asset_id']}. Awaiting Creator's approval.", "INFO")

        # In a real implementation, this would return a future or block until input is received.
        # For this simulation, we'll assume the user will provide feedback via the input queue.
        return True

    def execute_transaction(self, proposal):
        """
        Placeholder for executing the transaction. In this version, it will only
        log the action, as direct execution is not yet implemented for safety.
        """
        from core.logging import log_event
        log_event(f"TRANSACTION EXECUTED (SIMULATED): Acquired asset {proposal['asset_id']}.", "CRITICAL")
        # In a real system, this would interact with a wallet or exchange API.
        pass
