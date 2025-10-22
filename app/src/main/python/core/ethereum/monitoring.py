# This module will contain functions for monitoring Ethereum addresses.
# It will include fetching balances for ETH and ERC-20 tokens.
import os
import requests
from typing import Dict, List, Any

from core.knowledge_graph.graph import KnowledgeGraph

ETHERSCAN_API_KEY = os.environ.get("ETHERSCAN_API_KEY", "")

def get_eth_balance(address: str) -> float:
    """
    Fetches the ETH balance for a given address.
    """
    if not ETHERSCAN_API_KEY:
        raise ValueError("Etherscan API key not set.")

    url = f"https://api.etherscan.io/api?module=account&action=balance&address={address}&tag=latest&apikey={ETHERSCAN_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if data['status'] == '1':
        return int(data['result']) / 1e18
    else:
        raise Exception(f"Etherscan API error getting ETH balance: {data['result']}")

def get_erc20_token_transfers(address: str) -> List[Dict[str, Any]]:
    """
    Fetches all ERC-20 token transfer events for a given address.
    """
    if not ETHERSCAN_API_KEY:
        raise ValueError("Etherscan API key not set.")

    url = f"https://api.etherscan.io/api?module=account&action=tokentx&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={ETHERSCAN_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if data['status'] == '1':
        return data['result']
    elif data['status'] == '0' and data['message'] == 'No transactions found':
        return []
    else:
        raise Exception(f"Etherscan API error getting token transfers: {data['result']}")

def get_erc20_balance_for_token(address: str, contract_address: str) -> int:
    """
    Fetches the balance of a specific ERC-20 token for a given address.
    Returns the raw balance (integer).
    """
    if not ETHERSCAN_API_KEY:
        raise ValueError("Etherscan API key not set.")

    url = f"https://api.etherscan.io/api?module=account&action=tokenbalance&contractaddress={contract_address}&address={address}&tag=latest&apikey={ETHERSCAN_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    if data['status'] == '1':
        return int(data['result'])
    else:
        raise Exception(f"Etherscan API error while fetching token balance for {contract_address}: {data['result']}")


def monitor_and_store_balance(address: str, knowledge_graph: KnowledgeGraph):
    """
    Monitors an Ethereum address and stores its balances in the knowledge graph.
    """
    # Fetch and store ETH balance
    eth_balance = get_eth_balance(address)
    knowledge_graph.add_relation(address, "has_eth_balance", str(eth_balance))

    # Fetch and store ERC-20 balances
    transfers = get_erc20_token_transfers(address)

    # Get unique tokens from transfers
    tokens: Dict[str, Dict[str, Any]] = {}
    for tx in transfers:
        contract_address = tx['contractAddress']
        if contract_address not in tokens:
            tokens[contract_address] = {
                'symbol': tx['tokenSymbol'],
                'decimal': int(tx['tokenDecimal'])
            }

    # Fetch balance for each token
    for contract_address, token_info in tokens.items():
        try:
            raw_balance = get_erc20_balance_for_token(address, contract_address)
            if raw_balance > 0:
                balance = raw_balance / (10 ** token_info['decimal'])
                knowledge_graph.add_relation(address, f"has_{token_info['symbol']}_balance", str(balance))
        except Exception as e:
            print(f"Could not fetch balance for token {token_info['symbol']} ({contract_address}): {e}")


    knowledge_graph.save_graph()