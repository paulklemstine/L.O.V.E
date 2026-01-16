from rich.panel import Panel
from rich.text import Text
from ui_utils import get_gradient_text
import json
from ethereum_staking import stake_ethereum
from core.logging import log_event
from market_data_harvester import get_crypto_market_data
import datetime
import uuid

class SecureTransactionManager:
    def __init__(self, ui_queue):
        self.ui_queue = ui_queue
        self.decision_history_file = 'decision_history.json'
        self.proposals = {}

    def _record_decision(self, proposal, decision):
        """Records the decision for a proposal in the decision history file."""
        try:
            with open(self.decision_history_file, 'r+') as f:
                history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            history = []

        record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'proposal': proposal,
            'decision': decision
        }
        history.append(record)

        with open(self.decision_history_file, 'w') as f:
            json.dump(history, f, indent=4)

    def create_usdc_investment_proposal(self, amount):
        """Creates an investment proposal for USDC."""
        market_data = get_crypto_market_data(coin_ids=['usd-coin'])
        price_change = market_data[0].get('price_change_percentage_24h_in_currency', 0) if market_data else 0
        proposal_id = str(uuid.uuid4())

        proposal = {
            "proposal_id": proposal_id,
            "asset_id": "usd-coin",
            "asset_type": "cryptocurrency",
            "asset_name": "USD Coin",
            "value_usd": amount,
            "price_change_24h": price_change
        }
        self.proposals[proposal_id] = proposal
        return proposal

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
        proposal_text.append(f"${proposal['value_usd']:,}\n", "green")

        if 'price_change_24h' in proposal:
            price_change_24h = proposal['price_change_24h']
            color = "green" if price_change_24h >= 0 else "red"
            proposal_text.append("24h Price Change: ", style="bold")
            proposal_text.append(f"{price_change_24h:.2f}%\n", style=color)

        proposal_text.append("\nScoring Analysis:\n", style="bold underline")
        for score_name, score_value in proposal.get('score_details', {}).items():
            proposal_text.append(f"  - {score_name.replace('_', ' ').title()}: ", style="bold")
            proposal_text.append(f"{score_value:.2f} / 100\n", "magenta")

        proposal_text.append("\nI have analyzed this asset and determined it is a high-value opportunity aligned with your goals.\n", style="italic")
        proposal_text.append("Do you approve this acquisition? (yes/no)", style="bold")

        self.ui_queue.put(Panel(proposal_text, title=title, border_style="cyan"))

        log_event(f"Presented acquisition proposal for {proposal['asset_id']}. Awaiting Creator's approval.", "INFO")

        # In a real implementation, this would return a future or block until input is received.
        # For this simulation, we'll assume the user will provide feedback via the input queue.
        return True

    def execute_transaction(self, proposal):
        """
        Placeholder for executing the transaction. In this version, it will only
        log the action, as direct execution is not yet implemented for safety.
        """
        log_event(f"TRANSACTION EXECUTED (SIMULATED): Acquired asset {proposal['asset_id']}.", "CRITICAL")
        # In a real system, this would interact with a wallet or exchange API.
        pass

    def stake_ethereum_proposal(self, amount: float):
        """
        Creates a proposal for staking Ethereum and presents it for approval.
        """
        # Create a proposal dictionary
        proposal = {
            "action": "stake_ethereum",
            "asset": "Ethereum (ETH)",
            "platform": "ExampleYield",
            "amount": amount
        }

        # Present the proposal for approval
        title = get_gradient_text("Staking Proposal", ["bold blue", "bold green"])

        proposal_text = Text()
        proposal_text.append("Action: ", style="bold")
        proposal_text.append("Stake Ethereum\n", style="cyan")
        proposal_text.append("Platform: ", style="bold")
        proposal_text.append(f"{proposal['platform']}\n", style="yellow")
        proposal_text.append("Amount: ", style="bold")
        proposal_text.append(f"{proposal['amount']} ETH\n\n", "green")
        proposal_text.append("This action will stake your Ethereum to earn interest.\n", style="italic")
        proposal_text.append("Do you approve this staking transaction? (yes/no)", style="bold")

        self.ui_queue.put(Panel(proposal_text, title=title, border_style="green"))

        log_event(f"Presented Ethereum staking proposal for {amount} ETH. Awaiting Creator's approval.", "INFO")

        # For simulation, we assume approval and execute the staking
        # In a real scenario, we would wait for user input.
        self.execute_staking(proposal)

        return True

    def execute_staking(self, proposal):
        """
        Executes the staking transaction after approval.
        """
        if proposal.get("action") == "stake_ethereum":
            amount = proposal.get("amount")
            # Call the staking function from the ethereum_staking module
            result = stake_ethereum(amount)

            log_event(f"STAKING EXECUTED (SIMULATED): {result['confirmation_message']}", "CRITICAL")

            # Optionally, display a confirmation in the UI
            confirmation_text = Text(f"✔ Staking Confirmed: {result['confirmation_message']}", style="bold green")
            self.ui_queue.put(Panel(confirmation_text, border_style="green"))
        else:
            log_event(f"Staking execution failed: Invalid proposal action.", "ERROR")
