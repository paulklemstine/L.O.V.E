import requests
from core.logging import log_event

# Define the base URLs for the APIs
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
OPENSEA_API_URL = "https://api.opensea.io/api/v1"

def get_crypto_market_data(coin_ids=['bitcoin', 'ethereum']):
    """
    Fetches market data for specified cryptocurrencies from CoinGecko.

    Args:
        coin_ids (list): A list of cryptocurrency IDs to fetch data for.

    Returns:
        dict: A dictionary containing the market data for the specified coins,
              or None if the request fails.
    """
    try:
        url = f"{COINGECKO_API_URL}/coins/markets"
        params = {
            'vs_currency': 'usd',
            'ids': ','.join(coin_ids),
            'price_change_percentage': '24h'
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        log_event(f"Successfully fetched crypto market data for: {', '.join(coin_ids)}", "INFO")
        return response.json()
    except requests.exceptions.RequestException as e:
        log_event(f"Error fetching crypto market data: {e}", "ERROR")
        return None

def get_nft_collection_stats(collection_slug):
    """
    Fetches collection statistics for a specified NFT collection from OpenSea.

    Args:
        collection_slug (str): The slug of the NFT collection to fetch data for.

    Returns:
        dict: A dictionary containing the collection statistics, or None if the
              request fails.
    """
    try:
        url = f"{OPENSEA_API_URL}/collection/{collection_slug}"
        response = requests.get(url)
        response.raise_for_status()
        log_event(f"Successfully fetched NFT collection stats for: {collection_slug}", "INFO")
        return response.json().get('collection', {}).get('stats', {})
    except requests.exceptions.RequestException as e:
        log_event(f"Error fetching NFT collection stats: {e}", "ERROR")
        return None
