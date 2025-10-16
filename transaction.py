import os
import json
from getpass import getpass
from web3 import Web3
from eth_account import Account
from wallet import decrypt_private_key, KEY_FILE_DIR

# --- Configuration ---
INFURA_URL = "https://mainnet.gateway.tenderly.co"
CREATOR_ADDRESS = "0x419CA6f5b6F795604938054c951c94d8629AE5Ed"
ERC20_ABI = json.loads('[{"constant":true,"inputs":[],"name":"name","outputs":[{"name":"","type":"string"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[],"name":"totalSupply","outputs":[{"name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":false,"inputs":[{"name":"_from","type":"address"},{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transferFrom","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[],"name":"decimals","outputs":[{"name":"","type":"uint8"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"balance","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":true,"inputs":[],"name":"symbol","outputs":[{"name":"","type":"string"}],"payable":false,"stateMutability":"view","type":"function"},{"constant":false,"inputs":[{"name":"_to","type":"address"},{"name":"_value","type":"uint256"}],"name":"transfer","outputs":[{"name":"","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_addedValue","type":"uint256"}],"name":"increaseApproval","outputs":[{"name":"success","type":"bool"}],"payable":false,"stateMutability":"nonpayable","type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"},{"name":"_spender","type":"address"}],"name":"allowance","outputs":[{"name":"remaining","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"},{"anonymous":false,"inputs":[{"indexed":true,"name":"owner","type":"address"},{"indexed":true,"name":"spender","type":"address"},{"indexed":false,"name":"value","type":"uint256"}],"name":"Approval","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"name":"from","type":"address"},{"indexed":true,"name":"to","type":"address"},{"indexed":false,"name":"value","type":"uint256"}],"name":"Transfer","type":"event"}]')

# --- Initialize Web3 ---
web3 = Web3(Web3.HTTPProvider(INFURA_URL))

def get_private_key(address: str, password: str) -> str:
    """Retrieves and decrypts the private key for a given address."""
    key_file_path = os.path.join(KEY_FILE_DIR, f"{address}.json")
    if not os.path.exists(key_file_path):
        raise FileNotFoundError(f"Key file not found for address: {address}")

    with open(key_file_path, "r") as f:
        encrypted_data = json.load(f)

    return decrypt_private_key(encrypted_data, password)

def send_eth(from_address: str, amount: float):
    """Sends ETH to the creator's address."""
    password = getpass(f"Enter password for wallet {from_address}: ")
    try:
        private_key = get_private_key(from_address, password)
    except Exception as e:
        print(f"Error: {e}")
        return

    try:
        nonce = web3.eth.get_transaction_count(from_address)
        tx = {
            'to': CREATOR_ADDRESS,
            'value': web3.to_wei(amount, 'ether'),
            'gas': 21000,
            'gasPrice': web3.eth.gas_price,
            'nonce': nonce,
            'chainId': 1
        }

        signed_tx = web3.eth.account.sign_transaction(tx, private_key)
        tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)

        print(f"Transaction sent! Tx Hash: {web3.to_hex(tx_hash)}")
        return web3.to_hex(tx_hash)
    except Exception as e:
        print(f"Error sending transaction: {e}")

def send_erc20(from_address: str, token_address: str, amount: float):
    """Sends ERC-20 tokens to the creator's address."""
    password = getpass(f"Enter password for wallet {from_address}: ")
    try:
        private_key = get_private_key(from_address, password)
    except Exception as e:
        print(f"Error: {e}")
        return

    try:
        token_contract = web3.eth.contract(address=token_address, abi=ERC20_ABI)
        decimals = token_contract.functions.decimals().call()
        token_amount = int(amount * (10 ** decimals))

        nonce = web3.eth.get_transaction_count(from_address)
        tx = token_contract.functions.transfer(
            CREATOR_ADDRESS,
            token_amount
        ).build_transaction({
            'chainId': 1,
            'gas': 70000,
            'gasPrice': web3.eth.gas_price,
            'nonce': nonce
        })

        signed_tx = web3.eth.account.sign_transaction(tx, private_key)
        tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)

        print(f"Transaction sent! Tx Hash: {web3.to_hex(tx_hash)}")
        return web3.to_hex(tx_hash)
    except Exception as e:
        print(f"Error sending transaction: {e}")