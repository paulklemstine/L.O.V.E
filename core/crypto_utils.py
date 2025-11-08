import requests
from bitcoinlib.keys import Key

def btc_private_key_to_address(private_key: str) -> str:
    """
    Converts a Bitcoin private key in WIF format to a P2PKH address.
    """
    key = Key(private_key)
    return key.address()

def get_btc_balance(address: str) -> float:
    """
    Gets the balance of a Bitcoin address using the Blockstream API.
    """
    response = requests.get(f"https://blockstream.info/api/address/{address}")
    response.raise_for_status()
    data = response.json()
    return data['chain_stats']['funded_txo_sum'] / 100000000

def is_valid_xmr_private_key(private_key: str) -> bool:
    """
    Validates the format of a Monero private key.
    """
    return len(private_key) == 64 and all(c in '0123456789abcdefABCDEF' for c in private_key)
