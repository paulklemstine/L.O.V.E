import os
import sys
from core.logging import log_event
from core.yield_xyz_client import YieldXYZClient

def stake_ethereum(api_key: str, yield_id: str, address: str, amount: float):
    """
    Initiates staking a given amount of Ethereum on the Yield.xyz platform.

    This function interacts with the Yield.xyz API to create an unsigned staking
    transaction. In a real implementation, the returned transaction would need
    to be signed by the user's wallet and broadcast to the Ethereum network.

    Args:
        api_key (str): The API key for authenticating with the Yield.xyz API.
        yield_id (str): The ID for the specific staking opportunity on Yield.xyz
                        (e.g., 'ethereum-lido-staking').
        address (str): The user's Ethereum wallet address.
        amount (float): The amount of ETH to stake.

    Returns:
        dict: A dictionary containing the API response, which includes the
              unsigned transaction details.
    """
    log_event(f"Initiating staking of {amount} ETH for address {address} on yield {yield_id}.", "INFO")

    client = YieldXYZClient(api_key=api_key)
    try:
        # Convert the ETH amount to wei (the smallest unit of Ether)
        # 1 ETH = 10^18 wei
        amount_in_wei = str(int(amount * 1e18))

        # Call the client to get the unsigned transaction
        transaction_details = client.enter_yield(
            yield_id=yield_id,
            address=address,
            amount=amount_in_wei
        )
        log_event(f"Successfully generated staking transaction for {amount} ETH.", "INFO")
        return transaction_details
    except Exception as e:
        log_event(f"Failed to initiate staking transaction: {e}", "ERROR")
        # Return a dictionary with error information
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        client.close()

if __name__ == '__main__':
    # --- Example Usage ---
    # This is a demonstration and requires a valid YIELD_XYZ_API_KEY environment variable.
    # The transaction generated is unsigned and would need to be signed and broadcast
    # in a real-world scenario.

    api_key = os.getenv("YIELD_XYZ_API_KEY")
    if not api_key:
        print("Error: YIELD_XYZ_API_KEY environment variable not set.")
        print("Please set the environment variable and try again.")
        sys.exit(1)

    # Parameters for the staking operation
    example_yield_id = "ethereum-lido-staking"
    example_address = "0x1234567890123456789012345678901234567890"
    eth_to_stake = 0.01

    print(f"--- Attempting to stake {eth_to_stake} ETH ---")
    print(f"Yield ID: {example_yield_id}")
    print(f"Address: {example_address}")
    print("------------------------------------------")

    # Call the staking function
    result = stake_ethereum(
        api_key=api_key,
        yield_id=example_yield_id,
        address=example_address,
        amount=eth_to_stake
    )

    # Print the result
    if result.get("status") == "error":
        print("Staking operation failed.")
        print(f"Error: {result.get('message')}")
    else:
        print("Staking operation initiated successfully.")
        print("API Response (unsigned transaction details):")
        print(result)

    print("------------------------------------------")
