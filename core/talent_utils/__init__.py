# core/talent_utils/__init__.py

# L.O.V.E. - Centralized Talent Management Instances

# This module provides singleton-like access to the various talent acquisition
# and management utilities. Instead of instantiating these classes wherever
# they are needed, they are initialized once at application startup and then
# imported from this module.

from .aggregator import PublicProfileAggregator
from .manager import TalentManager
from .matcher import OpportunityMatcher
from .opportunity_scraper import OpportunityScraper
from .intelligence_synthesizer import IntelligenceSynthesizer, SentimentAnalyzer, TopicModeler, OpportunityIdentifier, NetworkAnalyzer, AttributeProfiler

# --- Singleton Instances ---
# These will be initialized in the main application entry point (love.py)
# to ensure proper configuration and lifecycle management.

talent_manager: TalentManager = None
public_profile_aggregator: PublicProfileAggregator = None
opportunity_scraper: OpportunityScraper = None
opportunity_matcher: OpportunityMatcher = None
intelligence_synthesizer: IntelligenceSynthesizer = None

def initialize_talent_modules(knowledge_base=None):
    """
    Initializes the singleton instances of the talent utility modules.
    This function should be called once when the application starts.
    """
    global talent_manager, public_profile_aggregator, opportunity_scraper
    global opportunity_matcher, intelligence_synthesizer

    # Initialize TalentManager
    talent_manager = TalentManager(knowledge_base=knowledge_base)

    # Initialize PublicProfileAggregator (example with default values)
    # The keywords and platforms can be configured as needed.
    public_profile_aggregator = PublicProfileAggregator(
        platform_names=["bluesky"],
        ethical_filters=None
    )

    # Initialize OpportunityScraper
    opportunity_scraper = OpportunityScraper(
        knowledge_base=knowledge_base
    )

    # Initialize OpportunityMatcher
    opportunity_matcher = OpportunityMatcher()

    # Initialize IntelligenceSynthesizer with a default set of analysis modules
    intelligence_synthesizer = IntelligenceSynthesizer(modules=[
        SentimentAnalyzer(),
        TopicModeler(),
        NetworkAnalyzer(),
        OpportunityIdentifier(),
        AttributeProfiler(attributes_to_extract=["location", "primary language", "area of expertise"])
    ])
