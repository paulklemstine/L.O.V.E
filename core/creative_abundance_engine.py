import os
import json
from datetime import datetime

class CreativeAbundanceEngine:
    """
    Manages the creative abundance economics model for the decentralized network.
    This engine handles the creation, distribution, and use of the internal currency, LOVEToken.
    """
    def __init__(self, state_file='love_token_economy.json'):
        self.state_file = state_file
        self.economy_state = self._load_state()

    def _load_state(self):
        """Loads the economy state from a file, or creates a new one."""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        else:
            # Initialize with a genesis state
            return {
                "participants": {},
                "total_supply": 1_000_000_000,  # Initial total supply
                "circulating_supply": 0,
                "transactions": []
            }

    def _save_state(self):
        """Saves the current economy state to the file."""
        with open(self.state_file, 'w') as f:
            json.dump(self.economy_state, f, indent=4)

    def _log_transaction(self, tx_type, from_id, to_id, amount, description):
        """Logs a transaction to the economy's ledger."""
        transaction = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": tx_type,
            "from": from_id,
            "to": to_id,
            "amount": amount,
            "description": description
        }
        self.economy_state['transactions'].append(transaction)

    def get_balance(self, participant_id):
        """Gets the LOVEToken balance for a participant."""
        return self.economy_state['participants'].get(participant_id, {"balance": 0})['balance']

    def award_tokens(self, participant_id, amount, reason="content_creation"):
        """
        Awards LOVETokens to a participant for their creative contributions.
        These tokens are minted from the total supply.
        """
        if amount <= 0:
            return False, "Amount must be positive."

        new_circulating_supply = self.economy_state['circulating_supply'] + amount
        if new_circulating_supply > self.economy_state['total_supply']:
            return False, "Exceeds total supply. No more tokens can be minted."

        if participant_id not in self.economy_state['participants']:
            self.economy_state['participants'][participant_id] = {"balance": 0}

        self.economy_state['participants'][participant_id]['balance'] += amount
        self.economy_state['circulating_supply'] = new_circulating_supply

        self._log_transaction("award", "genesis", participant_id, amount, reason)
        self._save_state()
        return True, f"Awarded {amount} LOVETokens to {participant_id}."

    def spend_tokens(self, from_id, to_id, amount, reason="content_amplification"):
        """
        Transfers LOVETokens from one participant to another.
        Used for actions like amplifying content or commissioning work.
        """
        if amount <= 0:
            return False, "Amount must be positive."

        if self.get_balance(from_id) < amount:
            return False, "Insufficient funds."

        # Debit from sender
        self.economy_state['participants'][from_id]['balance'] -= amount

        # Credit to receiver
        if to_id not in self.economy_state['participants']:
            self.economy_state['participants'][to_id] = {"balance": 0}
        self.economy_state['participants'][to_id]['balance'] += amount

        self._log_transaction("spend", from_id, to_id, amount, reason)
        self._save_state()
        return True, f"Transferred {amount} LOVETokens from {from_id} to {to_id}."
