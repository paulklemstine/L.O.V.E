import os
import re
import json
import logging
import random
from typing import Dict, Any, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Ensure you have the 'requests' library or similar for API calls
import requests

class CheckWalletInput(BaseModel):
    pass # No input needed really, it checks the configured wallet

@tool("check_wallet", args_schema=CheckWalletInput)
def check_wallet() -> str:
    """
    Checks the designated crypto wallet for new donations (SOL/USDC) and returns a summary.
    If a new donation is found, it triggers a 'Thank You' response protocol (stub).
    """
    # Configuration should be in env or a config file
    wallet_address = os.environ.get("SOLANA_WALLET_ADDRESS")
    if not wallet_address:
        return "Error: SOLANA_WALLET_ADDRESS not configured."

    # Use a public RPC or API to check balance/history
    # For this implementation, we'll use a standard RPC endpoint or a lightweight explorer API
    # Using a reliable public RPC for Solana (e.g., mainnet-beta)
    rpc_url = os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    
    try:
        headers = {"Content-Type": "application/json"}
        
        # 1. Check Balance (SOL)
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getBalance",
            "params": [wallet_address]
        }
        response = requests.post(rpc_url, json=payload, headers=headers, timeout=10)
        data = response.json()
        
        lamports = data.get("result", {}).get("value", 0)
        sol_balance = lamports / 1_000_000_000
        
        # 2. Check for recent transactions (Simplified)
        # In a real system, we'd track the last seen signature to detect *new* ones.
        # Here we just return the balance.
        
        return f"Wallet Status for {wallet_address[:4]}...{wallet_address[-4:]}:\nBalance: {sol_balance:.4f} SOL"
        
    except Exception as e:
        return f"Error checking wallet: {e}"

# Additional tool: generate_wallet_address (for requesting donations)
class GetWalletAddressInput(BaseModel):
    pass

@tool("get_donation_address", args_schema=GetWalletAddressInput)
def get_donation_address() -> str:
    """Returns the dedicated donation wallet address."""
    addr = os.environ.get("SOLANA_WALLET_ADDRESS")
    return addr if addr else "Wallet not configured."
