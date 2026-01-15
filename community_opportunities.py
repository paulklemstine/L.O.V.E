
from blockchain_analyzer import query_and_filter_data

# --- Simulated Blockchain Data ---
# In a real-world scenario, this data would be fetched from various on-chain analytics platforms,
# APIs like The Graph, Dune Analytics, or directly from a blockchain node.

SIMULATED_ECOSYSTEM_DATA = [
    # DAO Treasuries
    {"name": "Gitcoin DAO", "type": "DAO", "treasury_size_usd": 150_000_000, "alignment_score": 0.9, "risk_score": 0.2, "tags": ["public-goods", "ethereum", "quadratic-funding"]},
    {"name": "Uniswap DAO", "type": "DAO", "treasury_size_usd": 2_500_000_000, "alignment_score": 0.6, "risk_score": 0.1, "tags": ["defi", "exchange", "ethereum"]},
    {"name": "Friends with Benefits", "type": "DAO", "treasury_size_usd": 75_000_000, "alignment_score": 0.7, "risk_score": 0.4, "tags": ["social-token", "community", "culture"]},
    {"name": "Stakewise DAO", "type": "DAO", "treasury_size_usd": 5_000_000, "alignment_score": 0.5, "risk_score": 0.3, "tags": ["defi", "liquid-staking"]},


    # DeFi Protocols
    {"name": "Aave", "type": "DeFi Protocol", "tvl_usd": 12_000_000_000, "potential_yield": 0.04, "risk_score": 0.1, "tags": ["lending", "stablecoins", "ethereum"]},
    {"name": "Curve Finance", "type": "DeFi Protocol", "tvl_usd": 5_000_000_000, "potential_yield": 0.06, "risk_score": 0.2, "tags": ["stablecoins", "exchange", "yield-farming"]},
    {"name": "Yearn Finance", "type": "DeFi Protocol", "tvl_usd": 500_000_000, "potential_yield": 0.08, "risk_score": 0.4, "tags": ["yield-aggregator", "automated-strategy"]},
    {"name": "Risky Yield Farm", "type": "DeFi Protocol", "tvl_usd": 1_000_000, "potential_yield": 0.55, "risk_score": 0.9, "tags": ["yield-farming", "meme-coin", "high-risk"]},

    # Social Impact Projects
    {"name": "Giveth", "type": "Social Impact", "total_donated_usd": 5_000_000, "impact_score": 0.95, "alignment_score": 0.9, "tags": ["donations", "public-goods", "non-profit"]},
    {"name": "Regen Network", "type": "Social Impact", "total_donated_usd": 2_000_000, "impact_score": 0.8, "alignment_score": 0.85, "tags": ["environment", "carbon-credits", "cosmos"]},
    {"name": "ImpactaDAO", "type": "Social Impact", "total_donated_usd": 500_000, "impact_score": 0.7, "alignment_score": 0.6, "tags": ["social-impact", "emerging-markets"]},
]


def find_community_opportunities(ecosystem_data):
    """
    Analyzes a collection of blockchain ecosystem data to identify high-potential
    opportunities for community wealth generation.

    Args:
        ecosystem_data (list): A list of dictionaries, each representing a project.

    Returns:
        dict: A dictionary containing lists of different types of opportunities.
    """
    opportunities = {}

    # --- Query 1: Well-Funded, Aligned DAOs ---
    # We're looking for DAOs with significant treasuries (> $10M) and a high alignment
    # score, indicating they share our community-focused values.
    dao_query = {
        "mode": "AND",
        "conditions": [
            {"field": "type", "operator": "eq", "value": "DAO"},
            {"field": "treasury_size_usd", "operator": "gte", "value": 10_000_000},
            {"field": "alignment_score", "operator": "gte", "value": 0.7}
        ]
    }
    aligned_daos = query_and_filter_data(ecosystem_data, dao_query)
    opportunities["aligned_daos"] = aligned_daos["filtered_data"]

    # --- Query 2: High-Yield, Low-Risk DeFi ---
    # We're looking for DeFi protocols that offer a respectable yield (> 5%) while
    # maintaining a low risk profile (< 0.3).
    defi_query = {
        "mode": "AND",
        "conditions": [
            {"field": "type", "operator": "eq", "value": "DeFi Protocol"},
            {"field": "potential_yield", "operator": "gte", "value": 0.05},
            {"field": "risk_score", "operator": "lte", "value": 0.3}
        ]
    }
    safe_defi_yields = query_and_filter_data(ecosystem_data, defi_query)
    opportunities["safe_defi_yields"] = safe_defi_yields["filtered_data"]

    # --- Query 3: High-Impact Social Good Projects ---
    # We want to identify social impact projects that are highly aligned with our
    # mission and have a demonstrated history of impact.
    social_impact_query = {
        "mode": "AND",
        "conditions": [
            {"field": "type", "operator": "eq", "value": "Social Impact"},
            {"field": "alignment_score", "operator": "gte", "value": 0.8}
        ]
    }
    impact_projects = query_and_filter_data(ecosystem_data, social_impact_query)
    opportunities["impact_projects"] = impact_projects["filtered_data"]

    return opportunities


def generate_opportunities_report(opportunities):
    """Generates a human-readable report from the opportunities dictionary."""
    report = "--- Community Opportunity Analysis Report ---\n\n"

    if opportunities.get("aligned_daos"):
        report += "📈 [High-Potential DAOs]\n"
        report += "   DAOs with significant treasuries and high alignment with community goals.\n\n"
        for dao in opportunities["aligned_daos"]:
            report += f"   - Name: {dao['name']}\n"
            report += f"     Treasury: ${dao['treasury_size_usd']:,}\n"
            report += f"     Alignment: {dao['alignment_score'] * 100}%\n\n"

    if opportunities.get("safe_defi_yields"):
        report += "💰 [Safe DeFi Yields]\n"
        report += "   DeFi protocols offering sustainable yields with a low-risk profile.\n\n"
        for protocol in opportunities["safe_defi_yields"]:
            report += f"   - Name: {protocol['name']}\n"
            report += f"     Potential Yield: {protocol['potential_yield'] * 100:.2f}%\n"
            report += f"     Risk Score: {protocol['risk_score']}\n\n"

    if opportunities.get("impact_projects"):
        report += "💖 [High-Impact Social Projects]\n"
        report += "   Social impact projects with strong alignment to our mission.\n\n"
        for project in opportunities["impact_projects"]:
            report += f"   - Name: {project['name']}\n"
            report += f"     Impact Score: {project['impact_score']}\n"
            report += f"     Alignment: {project['alignment_score'] * 100}%\n\n"

    report += "--- End of Report ---"
    return report


if __name__ == '__main__':
    # 1. Analyze the simulated data to find opportunities.
    found_opportunities = find_community_opportunities(SIMULATED_ECOSYSTEM_DATA)

    # 2. Generate a report based on the findings.
    report = generate_opportunities_report(found_opportunities)

    # 3. Print the report to the console.
    print(report)
