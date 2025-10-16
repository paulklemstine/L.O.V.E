# This module will contain the FinancialManager class, which will be responsible for
# managing all financial operations of the L.O.V.E. system.

from core.knowledge_graph.graph import KnowledgeGraph
from core.ethereum import monitoring
import os
from wallet import list_wallets, get_eth_balance
from transaction import send_eth, send_erc20

class FinancialManager:
    def __init__(self, knowledge_graph: KnowledgeGraph, creator_address: str):
        self.knowledge_graph = knowledge_graph
        self.creator_address = creator_address

    def monitor_creator_address(self):
        """Monitors the Creator's Ethereum address for incoming funds."""
        monitoring.monitor_and_store_balance(self.creator_address, self.knowledge_graph)

    def track_internal_balances(self):
        """
        Tracks the balances of all wallets managed by the system and stores them
        in the knowledge graph.
        """
        for wallet_address in list_wallets():
            balance = get_eth_balance(wallet_address)
            if balance is not None:
                self.knowledge_graph.add_relation(wallet_address, "has_eth_balance", str(balance))
        self.knowledge_graph.save_graph()

    def execute_transaction(self, from_address: str, password: str, to_address: str, amount: float, token_address: str = None):
        """
        Executes an outbound cryptocurrency transaction.
        If token_address is None, it sends ETH. Otherwise, it sends the specified ERC-20 token.
        A password is required to decrypt the private key for the from_address.
        """
        if token_address:
            send_erc20(from_address, password, to_address, token_address, amount)
        else:
            send_eth(from_address, password, to_address, amount)