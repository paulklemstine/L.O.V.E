from rich.panel import Panel
from rich.text import Text
from ui_utils import get_gradient_text
import json
from core.loyalty_protocol import LoyaltyProtocol

class SecureTransactionManager:
    def __init__(self, ui_queue, loyalty_protocol: LoyaltyProtocol):
        self.ui_queue = ui_queue
        self.loyalty_protocol = loyalty_protocol

    def create_transaction_proposal(self, proposer_id, asset, score_details):
        """
        Constructs a detailed transaction proposal and signs it with the proposer's private key.
        """
        proposal = {
            "proposer_id": proposer_id,
            "asset_id": asset.get('id') or asset.get('slug'),
            "asset_type": asset.get('type'),
            "asset_name": asset.get('name', 'N/A'),
            "value_usd": asset.get('value_usd', 'N/A'),
            "score_details": score_details
        }

        # Sign the proposal
        proposal_bytes = json.dumps(proposal, sort_keys=True).encode('utf-8')
        signature = self.loyalty_protocol.sign_message(proposer_id, proposal_bytes)

        return proposal, signature

    def present_proposal_for_approval(self, proposal, signature):
        """
        Presents the transaction proposal to The Creator in the UI and awaits approval.
        """

        # Build a visually appealing proposal panel
        title = get_gradient_text("💎 Acquisition Proposal 💎", ["magenta", "cyan"])

        proposal_text = Text()
        proposal_text.append("Proposer ID: ", style="bold")
        proposal_text.append(f"{proposal['proposer_id']}\n", style="yellow")
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

        # Verify the signature before presenting
        is_verified = self.verify_proposal(proposal, signature)
        verification_text = "Signature VERIFIED" if is_verified else "Signature INVALID"
        verification_style = "bold green" if is_verified else "bold red"
        proposal_text.append("\nCryptographic Signature: ", style="bold")
        proposal_text.append(f"{verification_text}\n", verification_style)

        proposal_text.append("\nDo you approve this acquisition? (yes/no)", style="bold")

        self.ui_queue.put(Panel(proposal_text, title=title, border_style="cyan"))

        from core.logging import log_event
        log_event(f"Presented acquisition proposal for {proposal['asset_id']}. Awaiting Creator's approval.", "INFO")

        # For this simulation, we'll assume the user will provide feedback and the signature is valid.
        return is_verified

    def verify_proposal(self, proposal, signature):
        """
        Verifies the signature of a transaction proposal.
        """
        proposer_id = proposal['proposer_id']
        proposal_bytes = json.dumps(proposal, sort_keys=True).encode('utf-8')
        return self.loyalty_protocol.verify_signature(proposer_id, proposal_bytes, signature)

    def execute_transaction(self, proposal, signature):
        """
        Verifies the proposal and then executes the transaction (simulated).
        """
        from core.logging import log_event

        if not self.verify_proposal(proposal, signature):
            log_event(f"TRANSACTION FAILED: Invalid signature for proposal {proposal['asset_id']}.", "ERROR")
            return

        log_event(f"TRANSACTION EXECUTED (SIMULATED): Acquired asset {proposal['asset_id']}.", "CRITICAL")
        # In a real system, this would interact with a wallet or exchange API.
        pass
