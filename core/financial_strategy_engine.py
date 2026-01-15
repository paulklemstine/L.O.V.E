# financial_strategy_engine.py

"""
The Financial Strategy Engine for L.O.V.E.

This module is the cognitive core for generating, evaluating, and recommending
financial strategies to fulfill the Creator's primary directive of abundance.
It synthesizes all available knowledge from the Knowledge Graph, market data,
and internal simulations to devise actionable plans.
"""

from typing import Dict, List, Any

from core.graph_manager import GraphDataManager

class FinancialStrategyEngine:
    """
    Synthesizes knowledge to generate actionable financial plans.
    """

    def __init__(self, knowledge_graph: GraphDataManager):
        """
        Initializes the engine with the agent's knowledge base.

        Args:
            knowledge_graph: The GraphDataManager instance containing the agent's beliefs.
        """
        self.kg = knowledge_graph
        print("Financial Strategy Engine initialized.")

    async def generate_strategies(self) -> List[Dict[str, Any]]:
        """
        The core method for generating financial strategies.

        This method will analyze the Knowledge Graph for opportunities, such as:
        - Analyzing the Creator's current portfolio.
        - Identifying tokens with high growth potential.
        - Exploring DeFi opportunities (e.g., staking, yield farming).
        - Assessing market sentiment and trends.

        Returns:
            A list of proposed financial strategies, each as a dictionary.
        """
        strategies = []

        creator_address = "0x419CA6f5b6F795604938054c951c94d8629AE5Ed"

        strategies.extend(self._analyze_creator_portfolio(creator_address))
        strategies.extend(self._identify_growth_tokens())
        strategies.extend(self._identify_crypto_opportunities())
        
        # New Feature: Market Data Integration
        market_opps = await self._identify_market_opportunities()
        strategies.extend(market_opps)

        return strategies

    async def _identify_market_opportunities(self) -> List[Dict[str, Any]]:
        """
        Fetches external market data to identify buying opportunities.
        """
        opportunities = []
        try:
            import httpx
            import json
            # Fetch top 5 coins by market cap
            url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=5&page=1&sparkline=false"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    for coin in data:
                        # Simple strategy: Momentum buy if up > 5% in 24h
                        price_change = coin.get('price_change_percentage_24h', 0)
                        # --- Confidence Scores ---
                        MOMENTUM_BUY_CONFIDENCE = 0.75
                        VALUE_BUY_CONFIDENCE = 0.85

                        if price_change and price_change > 5.0:
                            opportunities.append({
                                "strategy_id": f"MOMENTUM_BUY_{coin['symbol'].upper()}",
                                "description": f"{coin['name']} ({coin['symbol'].upper()}) is up {price_change:.1f}% in 24h. Considerations: Momentum.",
                                "actions": [f"buy {coin['symbol']}"],
                                "confidence_score": MOMENTUM_BUY_CONFIDENCE
                            })
                        # Value buy if down > 5% (Buy the dip)
                        elif price_change and price_change < -5.0:
                             opportunities.append({
                                "strategy_id": f"VALUE_BUY_{coin['symbol'].upper()}",
                                "description": f"{coin['name']} ({coin['symbol'].upper()}) is down {price_change:.1f}% in 24h. Considerations: value entry.",
                                "actions": [f"buy {coin['symbol']}"],
                                "confidence_score": VALUE_BUY_CONFIDENCE
                            })
                else:
                    print(f"Market data fetch failed: {response.status_code}")
        except Exception as e:
            # print(f"Error identifying market opportunities: {e}")
            pass # Silent fail to avoid log spam if network is down
        
        return opportunities

    def _identify_crypto_opportunities(self) -> List[Dict[str, Any]]:
        """
        Identifies opportunities from network crypto scans.
        """
        crypto_strategies = []
        CRYPTO_OPPORTUNITY_CONFIDENCE = 0.60
        for subject, relation, obj in self.kg.get_triples():
            if relation == "crypto_analysis":
                crypto_strategies.append({
                    "strategy_id": f"EXPLORE_CRYPTO_OPPORTUNITY_{subject.replace(':', '_')}",
                    "description": f"Potential crypto opportunity detected on host {subject}.",
                    "actions": [
                        "Investigate the host for mining or staking opportunities.",
                        "Assess the security and viability of the opportunity."
                    ],
                    "details": obj,
                    "confidence_score": CRYPTO_OPPORTUNITY_CONFIDENCE
                })
        return crypto_strategies

    def _identify_growth_tokens(self) -> List[Dict[str, Any]]:
        """
        Identifies tokens with high growth potential.
        """
        growth_strategies = []
        GROWTH_TOKEN_CONFIDENCE = 0.70

        # A simple heuristic: look for tokens recently listed on major exchanges.
        # In a real system, this would be more sophisticated.
        major_exchanges = ["Binance", "Coinbase", "Kraken"]

        for subject, relation, obj in self.kg.get_triples():
            if relation == "listed_on" and obj in major_exchanges:
                growth_strategies.append({
                    "strategy_id": f"INVEST_IN_{subject}",
                    "description": f"Token {subject} was recently listed on {obj}, indicating potential for growth.",
                    "actions": [
                        f"Analyze the fundamentals of {subject}.",
                        f"Consider a small investment in {subject} to capitalize on potential growth."
                    ],
                    "confidence_score": GROWTH_TOKEN_CONFIDENCE
                })

        return growth_strategies

    def _analyze_creator_portfolio(self, address: str) -> List[Dict[str, Any]]:
        """
        Analyzes the Creator's portfolio based on data in the Knowledge Graph.
        """
        portfolio_strategies = []

        # Find all balance relations for the given address
        triples = self.kg.get_triples()
        balances = [t for t in triples if t[0] == address and "balance" in t[1]]

        if not balances:
            return portfolio_strategies

        # Sort holdings by value (requires parsing balance from string)
        # This is a simplified example; real implementation would need price data
        try:
            sorted_balances = sorted(balances, key=lambda x: float(x[2]), reverse=True)

            top_holding = sorted_balances[0]
            token_symbol = top_holding[1].replace("has_", "").replace("_balance", "")

            PORTFOLIO_DIVERSIFICATION_CONFIDENCE = 0.90
            portfolio_strategies.append({
                "strategy_id": "PORTFOLIO_DIVERSIFICATION_01",
                "description": f"Creator's portfolio is heavily weighted in {token_symbol}. Recommend diversification.",
                "actions": [
                    f"Analyze market for alternative assets to balance portfolio.",
                    f"Reduce allocation in {token_symbol} if market conditions are unfavorable."
                ],
                "confidence_score": PORTFOLIO_DIVERSIFICATION_CONFIDENCE
            })
        except (ValueError, IndexError) as e:
            print(f"Could not analyze portfolio: {e}")
            # Handle cases where balance is not a valid float or no balances found
            pass

        return portfolio_strategies