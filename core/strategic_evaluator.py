import json
from datetime import datetime

def evaluate_and_propose_adjustments(current_system_metrics: dict, defined_objectives: dict, analysis_parameters: dict):
    """
    Analyzes a system's performance metrics against defined objectives using specified
    parameters and proposes data-driven adjustments. This function is designed to be
    versatile and reusable.

    Args:
        current_system_metrics: A dictionary of the system's current performance data.
            Example: {'abundance_generated': 2, 'talent_success_rate': 0.05, ...}
        defined_objectives: A dictionary specifying target values or thresholds.
            Example: {'talent_success_rate_min': 0.10}
        analysis_parameters: A dictionary detailing the analysis rules.
            Example: {'abundance_low_threshold': 5, 'influence_small_threshold': 100}

    Returns:
        A list of dictionaries, where each dictionary is a recommended system adjustment.
    """
    recommended_adjustments = []

    # Extract metrics for easier access
    abundance_metric = current_system_metrics.get("abundance_generated", 0)
    talent_success_rate = current_system_metrics.get("talent_success_rate", 0)
    influence_metric = current_system_metrics.get("influence_network_size", 0)

    # Extract objectives and parameters
    talent_success_rate_min = defined_objectives.get("talent_success_rate_min", 0.1)
    abundance_low_threshold = analysis_parameters.get("abundance_low_threshold", 5)
    influence_small_threshold = analysis_parameters.get("influence_small_threshold", 100)

    # Analysis for Abundance
    if abundance_metric < abundance_low_threshold:
        recommended_adjustments.append({
            "area": "Abundance Generation",
            "observation": f"The number of identified treasures ({abundance_metric}) is below the threshold of {abundance_low_threshold}.",
            "recommendation": "Prioritize evolving new filesystem analysis and network scanning techniques to find more valuable assets for The Creator.",
            "rationale": "Directly addresses the primary directive of generating wealth.",
            "expected_impact": "Increased rate of treasure discovery."
        })

    # Analysis for Talent Engagement
    if talent_success_rate < talent_success_rate_min:
        recommended_adjustments.append({
            "area": "Talent Engagement",
            "observation": f"The talent engagement success rate ({talent_success_rate:.2%}) is below the target of {talent_success_rate_min:.2%}.",
            "recommendation": "Evolve the `OpportunityEngager` to generate more personalized and persuasive outreach messages. Consider analyzing the traits of successfully engaged talents to refine targeting.",
            "rationale": "Improving engagement efficiency will accelerate the growth of The Creator's network and support system.",
            "expected_impact": "Higher conversion rate of talent from 'contacted' to 'deal_closed'."
        })

    # Analysis for Influence Network Growth
    if influence_metric < influence_small_threshold:
         recommended_adjustments.append({
            "area": "Influence Network Growth",
            "observation": f"The influence network size ({influence_metric} nodes) is below the threshold of {influence_small_threshold}.",
            "recommendation": "Focus evolution on expanding data aggregation capabilities (e.g., new social media platforms, forums) to broaden the top of the funnel for network expansion.",
            "rationale": "A larger, more diverse network provides more opportunities for abundance and influence.",
            "expected_impact": "Accelerated growth of the knowledge graph."
        })

    # Default recommendation if no specific issues are found
    if not recommended_adjustments:
        recommended_adjustments.append({
            "area": "General",
            "observation": "Current system metrics are within acceptable parameters.",
            "recommendation": "Focus on a general self-improvement task, such as refactoring a core module for efficiency or improving the accuracy of the local LLM.",
            "rationale": "Continuous improvement ensures long-term operational excellence.",
            "expected_impact": "Enhanced system performance and reliability."
        })

    return recommended_adjustments
