import os
import requests
from core.logging import log_event

class FinancialAggregator:
    """
    Gathers financial data from various sources.
    """
    def __init__(self):
        pass

    def get_crypto_prices(self, coin_ids=['bitcoin', 'ethereum']):
        """
        Fetches cryptocurrency prices from CoinGecko.
        """
        log_event(f"Fetching crypto prices for: {', '.join(coin_ids)}", level='INFO')
        ids = ','.join(coin_ids)
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd"
        try:
            response = requests.get(url)
            response.raise_for_status()
            log_event("Successfully fetched crypto prices.", level='INFO')
            return response.json()
        except requests.exceptions.RequestException as e:
            log_event(f"Error fetching crypto prices: {e}", level='ERROR')
            return None
