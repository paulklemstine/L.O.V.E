import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Campaign:
    """
    Represents a marketing campaign.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled Campaign"
    goals: List[str] = field(default_factory=list)
    target_audience: Dict[str, Any] = field(default_factory=dict)
    channels: List[str] = field(default_factory=list)
    engagement_metrics: Dict[str, Any] = field(default_factory=dict)
    sentiment_scores: Dict[str, float] = field(default_factory=dict)

class CampaignManager:
    """
    Manages marketing campaigns.
    """
    def __init__(self):
        self.campaigns: Dict[str, Campaign] = {}

    def create_campaign(self, name: str, goals: List[str], target_audience: Dict[str, Any], channels: List[str]) -> Campaign:
        """
        Creates a new campaign.
        """
        campaign = Campaign(
            name=name,
            goals=goals,
            target_audience=target_audience,
            channels=channels
        )
        self.campaigns[campaign.id] = campaign
        return campaign

    def get_campaign(self, campaign_id: str) -> Campaign:
        """
        Retrieves a campaign by its ID.
        """
        return self.campaigns.get(campaign_id)

    def list_campaigns(self) -> List[Campaign]:
        """
        Lists all campaigns.
        """
        return list(self.campaigns.values())

    def update_engagement_metrics(self, campaign_id: str, metrics: Dict[str, Any]):
        """
        Updates the engagement metrics for a campaign.
        """
        campaign = self.get_campaign(campaign_id)
        if campaign:
            campaign.engagement_metrics.update(metrics)

    def update_sentiment_scores(self, campaign_id: str, scores: Dict[str, float]):
        """
        Updates the sentiment scores for a campaign.
        """
        campaign = self.get_campaign(campaign_id)
        if campaign:
            campaign.sentiment_scores.update(scores)
