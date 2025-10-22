import asyncio
from web3 import Web3
from .utils import get_web3_instance

# Define a threshold for what is considered a "high value" contract.
# 100 ETH in Wei.
HIGH_VALUE_THRESHOLD_WEI = Web3.to_wei(100, 'ether')

async def find_high_value_contracts(w3: Web3, scan_range: int = 1000) -> set[str]:
    """
    Scans a range of recent blocks to find high-value contracts.

    Note: This is a simplistic approach for demonstration. A more robust solution
    would involve indexing services or more sophisticated heuristics. The L.O.V.E.
    agent's "COGNITIVE MATRIX" can later enhance this.

    Args:
        w3: A connected Web3 instance.
        scan_range: The number of recent blocks to scan.

    Returns:
        A set of contract addresses (as checksum strings) that meet the high-value criteria.
    """
    print(f"Starting discovery scan for high-value contracts over the last {scan_range} blocks...")

    latest_block_number = w3.eth.block_number
    start_block = max(0, latest_block_number - scan_range)

    found_contracts = set()

    # Using a set to keep track of addresses we've already checked
    checked_addresses = set()

    for block_num in range(latest_block_number, start_block, -1):
        try:
            # Asynchronously fetch the block to avoid blocking the event loop for long
            block = await asyncio.to_thread(w3.eth.get_block, block_num, full_transactions=True)

            if block_num % 100 == 0:
                print(f"Scanning block {block_num}... Found {len(found_contracts)} contracts so far.")

            for tx in block.transactions:
                contract_address = tx['to']

                if contract_address and contract_address not in checked_addresses:
                    checked_addresses.add(contract_address)

                    # Check if the address has code (i.e., it's a contract)
                    code = await asyncio.to_thread(w3.eth.get_code, contract_address)

                    if code and code != b'':
                        # It's a contract, now check its balance
                        balance = await asyncio.to_thread(w3.eth.get_balance, contract_address)

                        if balance > HIGH_VALUE_THRESHOLD_WEI:
                            print(f"Found high-value contract: {contract_address} with balance: {Web3.from_wei(balance, 'ether')} ETH")
                            found_contracts.add(contract_address)

        except Exception as e:
            print(f"Error processing block {block_num}: {e}")
            # Continue to the next block
            continue

    print(f"Discovery scan finished. Found a total of {len(found_contracts)} high-value contracts.")
    return found_contracts

async def main():
    """Main function for local testing of the discovery tool."""
    try:
        w3 = get_web3_instance()
        if not w3.is_connected():
            print("Failed to connect to Ethereum node. Please ensure a node is running on http://127.0.0.1:8545.")
            return

        print(f"Connected to Ethereum node. Chain ID: {w3.eth.chain_id}")
        high_value_contracts = await find_high_value_contracts(w3, scan_range=500)

        if high_value_contracts:
            print("\nDiscovered High-Value Contracts:")
            for addr in high_value_contracts:
                print(f"- {addr}")
        else:
            print("\nNo high-value contracts found in the scanned range.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # This allows running the script directly for testing purposes.
    # Requires a running Ethereum node.
    asyncio.run(main())