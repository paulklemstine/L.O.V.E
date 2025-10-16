import os
import requests
from web3 import Web3

# It's good practice to use an environment variable for the API key.
ETHERSCAN_API_KEY = os.environ.get("ETHERSCAN_API_KEY", "")
DEFAULT_NODE_URL = "http://127.0.0.1:8545"

def get_web3_instance(node_url: str = DEFAULT_NODE_URL) -> Web3:
    """
    Returns a connected Web3 instance.

    Args:
        node_url: The URL of the Ethereum node to connect to.

    Returns:
        A Web3 instance connected to the specified node.
    """
    return Web3(Web3.HTTPProvider(node_url))

def get_contract_source_code(address: str, api_key: str = ETHERSCAN_API_KEY) -> str:
    """
    Fetches the verified source code of a contract from the Etherscan API.

    Args:
        address: The Ethereum address of the contract.
        api_key: The Etherscan API key.

    Returns:
        The contract's source code as a string, or an empty string if not found or an error occurs.
    """
    if not api_key:
        print("Warning: Etherscan API key is not set. Cannot fetch source code.")
        return ""

    api_url = f"https://api.etherscan.io/api?module=contract&action=getsourcecode&address={address}&apikey={api_key}"

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()

        if data['status'] == '1' and data['message'] == 'OK':
            source_code = data['result'][0]['SourceCode']
            # Sometimes the source code is wrapped in an extra pair of curly braces
            if source_code.startswith('{') and source_code.endswith('}'):
                import json
                try:
                    # It might be a JSON object containing multiple sources
                    sources = json.loads(source_code[1:-1])['sources']
                    # Concatenate all source files
                    return "\n\n".join(sources[key]['content'] for key in sources)
                except (json.JSONDecodeError, KeyError):
                    # Not a JSON object, return as is
                    pass
            return source_code
        else:
            print(f"Etherscan API error for address {address}: {data['result']}")
            return ""

    except requests.exceptions.RequestException as e:
        print(f"Error fetching source code for {address} from Etherscan: {e}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred while fetching source code: {e}")
        return ""