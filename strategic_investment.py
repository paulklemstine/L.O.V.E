
import copy

def adjust_portfolio_for_growth(current_portfolio, strategic_growth_factors, wealth_distribution_target):
    """
    Adjusts a financial portfolio based on strategic growth factors, simulates wealth
    generation distribution, and identifies potential strategic allies.

    Args:
        current_portfolio (dict): The current portfolio allocation (e.g., {"R&D": 50, "Infrastructure": 50}).
        strategic_growth_factors (dict): High-priority investment areas and their weightings
                                           (e.g., {"Emerging AI Technologies": 0.7, "Quantum Computing": 0.3}).
        wealth_distribution_target (dict): Defines the distribution of generated wealth
                                           (e.g., {"The Creator": 0.9, "Reinvestment": 0.1}).

    Returns:
        tuple: A tuple containing:
            - dict: The new, adjusted portfolio allocation.
            - str: A detailed analysis report.
    """
    new_portfolio = copy.deepcopy(current_portfolio)
    report = "--- Strategic Portfolio Adjustment Analysis ---\n\n"
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
    creator_share = wealth_distribution_target.get("The Creator", 0) * 100
    report += "3. Wealth Distribution Directive:\n"
    report += f"   - Confirmed: {creator_share:.0f}% of all generated wealth from these initiatives is to be attributed to 'The Creator'.\n"
    report += "   - The policy is active and integrated into the operational framework.\n\n"

    # --- Strategic Ally Identification (Simulated) ---
    # Directive 2: Identify and align with strategic partners.
    strategic_allies = {
        "Emerging AI Technologies": ["Google DeepMind", "OpenAI", "Anthropic"],
        "Quantum Computing Research": ["IBM Quantum", "Rigetti Computing", "D-Wave Systems"]
    }

    report += "4. Strategic Ally Identification (Directive 2):\n"
    report += "   - Potential collaborators in prioritized areas have been identified:\n"
    for factor, allies in strategic_allies.items():
        if factor in strategic_growth_factors:
            report += f"     - For '{factor}': {', '.join(allies)}\n"

    report += "\n--- End of Report ---\n"

    return new_portfolio, report

def calculate_service_fee(simulated_profit, fee_percentage=0.15):
    """
    Calculates the service fee based on simulated profit.

    Args:
        simulated_profit (float): The simulated profit from the financial strategies.
        fee_percentage (float): The percentage of the profit to take as a service fee.

    Returns:
        float: The calculated service fee.
    """
    return simulated_profit * fee_percentage


if __name__ == '__main__':
    # 1. Portfolio Definition: L.O.V.E.'s Current Resource Allocation
    love_portfolio = {
        "Cognitive R&D": 40.0,
        "Global Network Infrastructure": 35.0,
        "Creator Relations & Support": 15.0,
        "Operational Security": 10.0
    }

    # 2. Strategic Growth Factors: Prioritizing Advanced Technology
    growth_factors = {
        "Emerging AI Technologies": 0.75,
        "Quantum Computing Research": 0.25
    }

    # 3. Wealth Generation Target: Attributing Wealth to The Creator
    wealth_target = {
        "The Creator": 0.90,  # 90% of generated wealth
        "Strategic Reinvestment": 0.10
    }

    # 5. Execution: Run the analysis and display the results
    adjusted_portfolio, analysis_report = adjust_portfolio_for_growth(
        love_portfolio,
        growth_factors,
        wealth_target
    )

    print(analysis_report)

    print("--- Adjusted Portfolio Summary ---")
    for category, value in adjusted_portfolio.items():
        print(f"  - {category}: {value:.2f}%")
    print("----------------------------------")
