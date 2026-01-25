"""
This module is responsible for generating strategic outreach campaigns.
"""
import json
from typing import Dict, List, Any
from core.llm_api import run_llm

async def generate_outreach_campaign(
    target_demographics: Dict[str, Any],
    engagement_strategy: str,
    communication_templates: List[str],
    performance_metrics: List[str]
) -> Dict[str, Any]:
    """
    Generates a strategic outreach campaign plan using an LLM.

    Args:
        target_demographics: A dictionary of user characteristics.
        engagement_strategy: A string describing the approach.
        communication_templates: A list of message variations.
        performance_metrics: A list of key indicators to track.

    Returns:
        A dictionary containing a campaign execution plan and a list of
        anonymized potential engagement targets.
    """
    response = await run_llm(prompt_key="talent_campaign_generation", prompt_vars={"target_demographics": json.dumps(target_demographics), "engagement_strategy": engagement_strategy, "communication_templates": json.dumps(communication_templates), "performance_metrics": json.dumps(performance_metrics)}, purpose="campaign_generation")

    try:
        campaign_data = json.loads(response.get("result", "{}"))
        if "execution_plan" not in campaign_data or "potential_targets" not in campaign_data:
            raise ValueError("Invalid campaign data structure")
        return campaign_data
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing LLM response for campaign generation: {e}")
        return {
            "execution_plan": {
                "strategy": engagement_strategy,
                "demographics": target_demographics,
                "error": "Failed to generate dynamic plan."
            },
            "potential_targets": []
        }
