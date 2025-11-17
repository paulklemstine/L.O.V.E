# scripts/demonstrate_allocation.py

"""
A simulation script to demonstrate the strategic resource allocation framework.

This script imports the `optimize_resource_allocation` function from the
`strategic_allocator` module and uses it to:

1.  Simulate talent matching between creative portfolios and industry opportunities.
2.  Apply blockchain transaction data to model decentralized finance (DeFi) workflows.
3.  Analyze pattern recognition metrics to generate actionable growth recommendations
    for competitive advantage.
"""

import sys
import os
import json

# Ensure the script can find the 'core' module.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.talent_utils.strategic_allocator import optimize_resource_allocation

def run_simulation():
    """
    Sets up and runs the full simulation.
    """
    print("🚀  --- Initializing Simulation: Strategic Allocation for Creative Professionals --- 🚀\n")

    # --- 1. Define Mock Data ---
    # In a real-world application, this data would be sourced from databases,
    # APIs, and on-chain analysis.

    # a. Creative Portfolios: Profiles of creative professionals.
    creative_portfolios = [
        {
            'id': 'prof_001',
            'name': 'Elena',
            'field': 'Digital Art & Animation',
            'skills': [
                {'name': '3D Modeling', 'level': 9.5},
                {'name': 'Motion Graphics', 'level': 8.8},
                {'name': 'VR Environment Design', 'level': 9.2}
            ],
            'career_goals': ['Lead a major animated production', 'Collaborate with innovative tech brands']
        },
        {
            'id': 'prof_002',
            'name': 'Jian',
            'field': 'Decentralized Application Development',
            'skills': [
                {'name': 'Solidity', 'level': 9.8},
                {'name': 'React/Next.js', 'level': 9.1},
                {'name': 'Economic Modeling', 'level': 8.5}
            ],
            'career_goals': ['Launch a successful DeFi protocol', 'Contribute to DAO governance models']
        }
    ]
    print("✅  Loaded Creative Portfolios.")

    # b. Industry Opportunities: A dataset of potential projects and collaborations.
    industry_opportunities = {
        'gigs': [
            {'id': 'gig_101', 'title': 'Lead Animator for a Sci-Fi Short Film', 'value': 25000, 'required_skills': ['3D Modeling', 'Motion Graphics']},
            {'id': 'gig_102', 'title': 'Smart Contract Auditor for a New NFT Marketplace', 'value': 18000, 'required_skills': ['Solidity']},
        ],
        'collaborations': [
            {'id': 'collab_201', 'title': 'Art Director for a Metaverse Concert Experience', 'required_skills': ['VR Environment Design']},
            {'id': 'collab_202', 'title': 'Economic Advisor for a Gaming DAO', 'required_skills': ['Economic Modeling', 'Solidity']},
        ]
    }
    print("✅  Loaded Industry Opportunities.")

    # c. Network Parameters: Data representing the broader professional and financial network.
    network_parameters = {
        'total_value_locked_defi': 3.5e6, # TVL in relevant protocols.
        'avg_tx_per_hour': 450,
        'market_trend_analysis': {
            'fastest_growing_sector': 'Decentralized Gaming',
            'in_demand_skills': ['VR Environment Design', 'Economic Modeling']
        },
        'blockchain_transaction_data': [
            # A simplified representation of on-chain activity.
            {'tx_hash': '0x...', 'from': '0xaaa', 'to': '0xbbb', 'value': 3.2, 'contract_interaction': 'DAO_VOTE'},
            {'tx_hash': '0x...', 'from': '0xccc', 'to': '0xddd', 'value': 5.0, 'contract_interaction': 'STAKE_ASSETS'},
        ]
    }
    print("✅  Loaded Network Parameters with DeFi data.")
    print("\n--------------------------------------------------")


    # --- 2. Run the Optimization Function ---
    # This is the core of the simulation, where the imported function processes
    # the mock data to generate strategies.
    print("\n🧠  --- Running Optimization Engine --- 🧠\n")
    optimized_strategies = optimize_resource_allocation(
        portfolio_data=creative_portfolios,
        opportunity_datasets=industry_opportunities,
        network_parameters=network_parameters
    )
    print("\n--------------------------------------------------")


    # --- 3. Display and Interpret Results ---
    # The output represents actionable growth recommendations.
    print("\n📈  --- Generated Growth Recommendations --- 📈\n")
    for portfolio_id, result in optimized_strategies.items():
        print(f"--- Strategy for: {result['name']} ---")
        print(f"  RECOMMENDATION: {result['strategy']['recommendation']}")
        print(f"  RATIONALE: {result['strategy']['rationale']}")
        print("  ACTIONABLE STEPS:")
        for step in result['strategy']['actionable_steps']:
            print(f"    - {step}")
        print("\n")

    print("🚀  --- Simulation Complete --- 🚀\n")


if __name__ == '__main__':
    run_simulation()
