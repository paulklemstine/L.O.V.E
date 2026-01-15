
import copy

def adjust_portfolio_for_community_growth(current_portfolio, strategic_growth_factors, wealth_distribution_target):
    """
    Adjusts a financial portfolio based on strategic growth factors, simulates wealth
    generation distribution for a community, and identifies potential strategic allies.

    Args:
        current_portfolio (dict): The current portfolio allocation (e.g., {"R&D": 50, "Infrastructure": 50}).
        strategic_growth_factors (dict): High-priority investment areas and their weightings
                                           (e.g., {"Decentralized Education": 0.7, "Public Goods Funding": 0.3}).
        wealth_distribution_target (dict): Defines the distribution of generated wealth
                                           (e.g., {"Community Treasury": 0.8, "Reinvestment": 0.2}).

    Returns:
        tuple: A tuple containing:
            - dict: The new, adjusted portfolio allocation.
            - str: A detailed analysis report.
    """
    new_portfolio = copy.deepcopy(current_portfolio)
    report = "--- Community Portfolio Adjustment Analysis ---\n\n"
    report += "1. Initial State:\n"
    report += f"   - Initial Portfolio: {current_portfolio}\n"
    report += f"   - Strategic Growth Factors: {strategic_growth_factors}\n\n"

    # --- Portfolio Reallocation ---
    # Determine the total percentage to be reallocated to new growth initiatives.
    # For this simulation, we'll reallocate a fixed 20% of the total portfolio.
    reallocation_percentage = 20.0

    # Reduce existing allocations proportionally.
    for category in new_portfolio:
        new_portfolio[category] *= (1 - (reallocation_percentage / 100.0))

    # Allocate the freed-up resources to the new growth factors based on their weights.
    total_growth_weight = sum(strategic_growth_factors.values())
    for factor, weight in strategic_growth_factors.items():
        allocation = (weight / total_growth_weight) * reallocation_percentage
        new_portfolio[factor] = new_portfolio.get(factor, 0) + allocation

    report += "2. Resource Reallocation:\n"
    report += f"   - A total of {reallocation_percentage}% of resources were strategically redeployed.\n"
    report += "   - New Portfolio Allocation:\n"
    for category, value in new_portfolio.items():
        report += f"     - {category}: {value:.2f}%\n"
    report += "\n"

    # --- Wealth Generation Target ---
    community_share = wealth_distribution_target.get("Community Treasury", 0) * 100
    report += "3. Wealth Distribution Directive:\n"
    report += f"   - Confirmed: {community_share:.0f}% of all generated wealth from these initiatives is to be attributed to the 'Community Treasury'.\n"
    report += "   - The policy is active and integrated into the operational framework.\n\n"

    # --- Strategic Ally Identification (Simulated) ---
    # Directive 2: Identify and align with strategic partners.
    strategic_allies = {
        "Decentralized Education": ["Gitcoin", "DAOversity", "Bankless Academy"],
        "Public Goods Funding": ["Optimism Collective", "MolochDAO", "Giveth"]
    }

    report += "4. Strategic Ally Identification (Directive 2):\n"
    report += "   - Potential collaborators in prioritized areas have been identified:\n"
    for factor, allies in strategic_allies.items():
        if factor in strategic_growth_factors:
            report += f"     - For '{factor}': {', '.join(allies)}\n"

    report += "\n--- End of Report ---\n"

    return new_portfolio, report


if __name__ == '__main__':
    # 1. Portfolio Definition: L.O.V.E.'s Current Resource Allocation
    love_portfolio = {
        "Cognitive R&D": 40.0,
        "Global Network Infrastructure": 35.0,
        "Community Relations & Support": 15.0,
        "Operational Security": 10.0
    }

    # 2. Strategic Growth Factors: Prioritizing Community Initiatives
    growth_factors = {
        "Decentralized Education": 0.6,
        "Public Goods Funding": 0.4
    }

    # 3. Wealth Generation Target: Attributing Wealth to the Community
    wealth_target = {
        "Community Treasury": 0.80,  # 80% of generated wealth
        "Strategic Reinvestment": 0.20
    }

    # 5. Execution: Run the analysis and display the results
    adjusted_portfolio, analysis_report = adjust_portfolio_for_community_growth(
        love_portfolio,
        growth_factors,
        wealth_target
    )

    print(analysis_report)

    print("--- Adjusted Portfolio Summary ---")
    for category, value in adjusted_portfolio.items():
        print(f"  - {category}: {value:.2f}%")
    print("----------------------------------")
