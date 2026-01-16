
import httpx

class YieldXYZClient:
    """
    A client for interacting with the Yield.xyz API.
    """
    def __init__(self, api_key: str):
        """
        Initializes the Yield.xyz API client.

        Args:
            api_key (str): The API key for authenticating with the Yield.xyz API.
        """
        self.base_url = "https://api.yield.xyz/v2"
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }
        self.client = httpx.Client(headers=headers)

    def get_yield_opportunities(self, network: str = "ethereum", limit: int = 10):
        """
        Fetches yield opportunities from the Yield.xyz API.

        Args:
            network (str): The blockchain network to filter by (e.g., "ethereum").
            limit (int): The maximum number of opportunities to return.

        Returns:
            dict: A dictionary containing the API response.
        """
        url = f"{self.base_url}/yields"
        params = {
            "network": network,
            "limit": limit
        }
        response = self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_yield_balances(self, yield_id: str, address: str):
        """
        Fetches the balances for a specific yield and address.

        Args:
            yield_id (str): The ID of the yield opportunity.
            address (str): The wallet address to check the balance for.

        Returns:
            dict: A dictionary containing the API response.
        """
        url = f"{self.base_url}/yields/{yield_id}/balances"
        data = {
            "address": address
        }
        response = self.client.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def enter_yield(self, yield_id: str, address: str, amount: str):
        """
        Initiates a staking transaction (enter a yield).

        Args:
            yield_id (str): The ID of the yield opportunity.
            address (str): The wallet address to stake from.
            amount (str): The amount to stake in the smallest unit (e.g., wei for ETH).

        Returns:
            dict: A dictionary containing the API response with the unsigned transaction.
        """
        url = f"{self.base_url}/yields/{yield_id}/enter"
        data = {
            "address": address,
            "arguments": {
                "amount": amount
            }
        }
        response = self.client.post(url, json=data)
        response.raise_for_status()
        return response.json()

    def close(self):
        """
        Closes the HTTP client.
        """
        self.client.close()


if __name__ == '__main__':
    # TODO: Replace with a valid API key from a secure configuration.
    api_key = "YOUR_API_KEY"
    client = YieldXYZClient(api_key)
    try:
        opportunities = client.get_yield_opportunities()
        print(opportunities)
    except httpx.HTTPStatusError as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        client.close()
