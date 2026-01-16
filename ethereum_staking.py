
import json
from datetime import datetime
from core.logging import log_event

STAKING_HISTORY_FILE = "staking_history.json"

def get_staking_history():
    """Retrieves the staking history from the JSON file."""
    try:
        with open(STAKING_HISTORY_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def add_to_staking_history(transaction):
    """Adds a new transaction to the staking history."""
    history = get_staking_history()
    history.append(transaction)
    with open(STAKING_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def stake_ethereum(amount: float):
    """
    Simulates staking a given amount of Ethereum on the ExampleYield platform.

    In a real implementation, this function would:
    1.  Connect to the Ethereum blockchain via a web3 provider.
    2.  Interact with the ExampleYield staking smart contract.
    3.  Validate the user's wallet has sufficient funds.
    4.  Send a transaction to the staking contract with the specified amount.
    5.  Handle transaction confirmation, retries, and potential errors.
    6.  Return a transaction hash or an object representing the staking position.

    For this simulation, it will simply log the intended action.

    Args:
        amount (float): The amount of ETH to stake.

    Returns:
        dict: A dictionary containing a confirmation message and the amount.
    """
    log_event(f"SIMULATION: Staking {amount} ETH on ExampleYield.", "INFO")

    # Simulate a successful transaction
    transaction_details = {
        "status": "success",
        "platform": "ExampleYield",
        "asset": "Ethereum (ETH)",
        "amount_staked": amount,
        "timestamp": datetime.now().isoformat(),
        "confirmation_message": f"Successfully staked {amount} ETH. Your assets are now earning interest."
    }

    add_to_staking_history(transaction_details)

    # In a real scenario, you might return a transaction hash or an ID.
    return transaction_details

def get_total_staked():
    """Calculates the total amount of ETH staked."""
    history = get_staking_history()
    return sum(item.get("amount_staked", 0) for item in history)

if __name__ == '__main__':
    # Example of how to use the function
    staked_amount = 1.5
    result = stake_ethereum(staked_amount)
    print(result)

    staked_amount_2 = 10.0
    result_2 = stake_ethereum(staked_amount_2)
    print(result_2)
