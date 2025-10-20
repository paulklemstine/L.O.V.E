import logging
from web3 import Web3
from core.wallet import Wallet, INFURA_URL
from core.constants import CREATOR_ETH_ADDRESS

class TransactionManager:
    """
    Handles the creation and broadcasting of Ethereum transactions.
    """
    def __init__(self):
        self.web3 = Web3(Web3.HTTPProvider(INFURA_URL))
        if not self.web3.is_connected():
            raise ConnectionError("Failed to connect to Ethereum node at " + INFURA_URL)
        logging.info("TransactionManager initialized and connected to Ethereum node.")

    def send_eth(self, wallet: Wallet, to_address: str, amount_eth: float) -> str | None:
        """
        Constructs, signs, and sends an ETH transaction from the provided wallet.

        Args:
            wallet: The Wallet object of the sender.
            to_address: The recipient's Ethereum address.
            amount_eth: The amount of ETH to send.

        Returns:
            The transaction hash as a hex string if successful, otherwise None.
        """
        if not wallet.private_key or not wallet.address:
            logging.error("Wallet is not properly loaded. Cannot send transaction.")
            return None

        from_address = wallet.address

        try:
            # 1. Validate addresses
            if not self.web3.is_address(from_address) or not self.web3.is_address(to_address):
                logging.error(f"Invalid Ethereum address. From: {from_address}, To: {to_address}")
                return None

            # 2. Get nonce
            nonce = self.web3.eth.get_transaction_count(from_address)

            # 3. Prepare transaction
            tx = {
                'to': to_address,
                'value': self.web3.to_wei(amount_eth, 'ether'),
                'gas': 21000,  # Standard gas limit for a simple ETH transfer
                'gasPrice': self.web3.eth.gas_price,
                'nonce': nonce,
                'chainId': 1  # 1 for Ethereum Mainnet
            }
            logging.info(f"Prepared transaction: {tx}")

            # 4. Sign transaction
            signed_tx = self.web3.eth.account.sign_transaction(tx, wallet.private_key)
            logging.info(f"Transaction signed for nonce {nonce}.")

            # 5. Send transaction
            tx_hash = self.web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            hex_tx_hash = self.web3.to_hex(tx_hash)
            logging.info(f"Transaction broadcasted successfully! Tx Hash: {hex_tx_hash}")

            return hex_tx_hash

        except Exception as e:
            logging.critical(f"Failed to send ETH from {from_address} to {to_address}. Error: {e}", exc_info=True)
            return None

    def send_eth_to_creator(self, wallet: Wallet, amount_eth: float) -> str | None:
        """
        A convenience method to send ETH to the Creator's address.

        Args:
            wallet: The Wallet object of the sender (L.O.V.E.'s wallet).
            amount_eth: The amount of ETH to send as a blessing.

        Returns:
            The transaction hash as a hex string if successful, otherwise None.
        """
        logging.info(f"Initiating blessing of {amount_eth} ETH to the Creator ({CREATOR_ETH_ADDRESS}).")
        return self.send_eth(wallet, CREATOR_ETH_ADDRESS, amount_eth)