# This module will contain the FinancialManager class, which will be responsible for
# managing all financial operations of the L.O.V.E. system.

from core.knowledge_graph.graph import KnowledgeGraph
from core.ethereum import monitoring
import os
from core.wallet import Wallet
from core.transaction import TransactionManager

class FinancialManager:
    def __init__(self, knowledge_graph: KnowledgeGraph, creator_address: str):
        self.knowledge_graph = knowledge_graph
        self.creator_address = creator_address
        self.love_wallet = Wallet()
        self.love_wallet.load_or_create()
        self.transaction_manager = TransactionManager()

    def monitor_creator_address(self):
        """Monitors the Creator's Ethereum address for incoming funds."""
        monitoring.monitor_and_store_balance(self.creator_address, self.knowledge_graph)

    def track_internal_balances(self):
        """
        Tracks the balances of all wallets managed by the system and stores them
        in the knowledge graph.
        """
        balance = self.love_wallet.get_balance()
        if balance is not None:
            self.knowledge_graph.add_relation(self.love_wallet.address, "has_eth_balance", str(balance))
        self.knowledge_graph.save_graph()

    def execute_transaction(self, to_address: str, amount: float):
        """
        Executes an outbound cryptocurrency transaction.
        """
        self.transaction_manager.send_eth(self.love_wallet, to_address, amount)