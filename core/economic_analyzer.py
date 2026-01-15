# core/economic_analyzer.py

"""
Economic Analyzer for L.O.V.E.

This module provides tools for analyzing economic trends, fetching market data,
and providing insights to guide the FinancialStrategyEngine.
"""

import httpx
from typing import List, Dict, Any
from core.logging import log_event

async def analyze_economic_trends() -> List[Dict[str, Any]]:
    """
    Fetches and analyzes economic data to identify trends and opportunities.

    This tool will:
    1. Fetch data for the top 10 cryptocurrencies by market cap from CoinGecko.
    2. Analyze various metrics, including price change, volume, and market cap.
    3. Calculate a "trend score" to identify assets with strong positive momentum.
    4. Return a list of assets with their analysis and trend score.
    """
    trends = []
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=10&page=1&sparkline=false"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            if response.status_code == 200:
                data = response.json()
                for coin in data:
                    trend_score = 0

                    # Price change analysis
                    price_change_24h = coin.get('price_change_percentage_24h', 0) or 0
                    if price_change_24h > 5:
                        trend_score += 2
                    elif price_change_24h > 0:
                        trend_score += 1

                    # Volume analysis
                    total_volume = coin.get('total_volume', 0) or 0
                    market_cap = coin.get('market_cap', 1) or 1
                    if total_volume / market_cap > 0.1: # High volume to market cap ratio
                        trend_score += 1

                    trends.append({
                        "asset_id": coin['id'],
                        "name": coin['name'],
                        "symbol": coin['symbol'].upper(),
                        "price": coin.get('current_price'),
                        "price_change_24h": price_change_24h,
                        "volume": total_volume,
                        "market_cap": market_cap,
                        "trend_score": trend_score,
                    })
            else:
                log_event(f"Failed to fetch economic data: {response.status_code}", level="ERROR")
    except Exception as e:
        log_event(f"Error in analyze_economic_trends: {e}", level="ERROR")

    # Sort by trend score, descending
    return sorted(trends, key=lambda x: x['trend_score'], reverse=True)

async def get_market_data(asset_id: str) -> Dict[str, Any]:
    """
    Fetches detailed market data for a specific asset.
    """
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{asset_id}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        print(f"Error fetching market data for {asset_id}: {e}")
    return {}
