from abc import ABC, abstractmethod
import asyncio
from core.llm_api import run_llm

class AnalysisModule(ABC):
    """Abstract base class for all analysis modules."""
    @abstractmethod
    async def analyze(self, dataset):
        pass

class IntelligenceSynthesizer:
    """
    A generic synthesizer that applies a series of analytical modules to a dataset.
    """
    def __init__(self, modules):
        self.modules = modules

    async def run(self, dataset):
        """
        Applies each specified module to the dataset and returns the enriched dataset.
        """
        enriched_dataset = dataset
        for module in self.modules:
            enriched_dataset = await module.analyze(enriched_dataset)
        return enriched_dataset

class SentimentAnalyzer(AnalysisModule):
    """Analyzes text fields to generate sentiment scores."""

    async def _get_sentiment(self, text):
        """Uses an LLM to get the sentiment of a given text."""
        if not text:
            return "neutral"

        prompt = f"""
        Analyze the sentiment of the following text.
        Respond with a single word: positive, negative, or neutral.

        Text:
        ---
        {text[:4000]}
        ---

        Sentiment (positive, negative, or neutral):
        """
        try:
            response_dict = await run_llm(prompt, purpose="sentiment_analysis")
            response = response_dict.get("result", "neutral").lower().strip()
            if response in ["positive", "negative", "neutral"]:
                return response
            return "neutral"
        except Exception:
            return "neutral"

    async def analyze(self, dataset):
        """Adds sentiment scores to each profile in the dataset."""
        for profile in dataset:
            full_text = profile.get('bio', '') or ''
            posts = profile.get('posts', [])
            if posts:
                full_text += "\n" + "\n".join([post.get('text', '') for post in posts if post.get('text')])

            sentiment = await self._get_sentiment(full_text)

            if 'analysis' not in profile:
                profile['analysis'] = {}
            profile['analysis']['sentiment'] = sentiment

        return dataset

class TopicModeler(AnalysisModule):
    """Identifies latent topics from text fields."""

    async def _get_topics(self, text):
        """Uses an LLM to identify topics in a given text."""
        if not text:
            return []

        prompt = f"""
        Identify the main topics from the following text.
        Please provide a comma-separated list of 2-5 topics.

        Text:
        ---
        {text[:4000]}
        ---

        Topics (comma-separated):
        """
        try:
            response_dict = await run_llm(prompt, purpose="topic_modeling")
            response = response_dict.get("result", "")
            topics = [topic.strip() for topic in response.split(',') if topic.strip()]
            return topics
        except Exception:
            return []

    async def analyze(self, dataset):
        """Adds a list of topics to each profile in the dataset."""
        for profile in dataset:
            full_text = profile.get('bio', '') or ''
            posts = profile.get('posts', [])
            if posts:
                full_text += "\n" + "\n".join([post.get('text', '') for post in posts if post.get('text')])

            topics = await self._get_topics(full_text)

            if 'analysis' not in profile:
                profile['analysis'] = {}
            profile['analysis']['topics'] = topics

        return dataset

class OpportunityIdentifier(AnalysisModule):
    """Identifies emergent themes and opportunities from analysis results."""

    async def _get_opportunities(self, sentiment, topics):
        """Uses an LLM to identify opportunities from sentiment and topics."""
        if not topics:
            return "No specific opportunities identified."

        prompt = f"""
        Given the following sentiment and topics, identify potential areas for growth,
        valuable connections, or emergent themes. Provide a brief, actionable summary.

        Sentiment: {sentiment}
        Topics: {', '.join(topics)}

        Opportunity Summary:
        """
        try:
            response_dict = await run_llm(prompt, purpose="opportunity_identification")
            return response_dict.get("result", "No specific opportunities identified.").strip()
        except Exception:
            return "Error in opportunity identification."

    async def analyze(self, dataset):
        """Adds an opportunity summary to each profile."""
        for profile in dataset:
            analysis_data = profile.get('analysis', {})
            sentiment = analysis_data.get('sentiment', 'neutral')
            topics = analysis_data.get('topics', [])

            opportunities = await self._get_opportunities(sentiment, topics)

            if 'analysis' not in profile:
                profile['analysis'] = {}
            profile['analysis']['opportunities'] = opportunities

        return dataset

class NetworkAnalyzer(AnalysisModule):
    """Processes network or relational data to identify key entities and their attributes."""

    async def analyze(self, dataset):
        """Calculates network metrics like influence score for each profile."""
        for profile in dataset:
            followers = profile.get('followers_count') or 0
            following = profile.get('follows_count') or 0

            # Avoid division by zero
            if following > 0:
                influence_score = round(followers / following, 2)
            else:
                influence_score = followers  # High score if they follow no one but have followers

            if 'analysis' not in profile:
                profile['analysis'] = {}

            profile['analysis']['network_analysis'] = {
                'influence_score': influence_score,
                'reach': followers
            }

        # This is an async function, but the current implementation is synchronous.
        # It's declared async to conform to the base class.
        await asyncio.sleep(0)
        return dataset

class AttributeProfiler(AnalysisModule):
    """Extracts and categorizes specific attributes of entities."""

    def __init__(self, attributes_to_extract):
        """
        Initializes the profiler with a list of attributes to extract.
        For example: ["age range", "stated interests", "social behavior indicators"]
        """
        self.attributes_to_extract = attributes_to_extract

    async def _get_attributes(self, text):
        """Uses an LLM to extract and categorize attributes from text."""
        if not text:
            return {}

        prompt = f"""
        From the text below, extract the following attributes:
        {', '.join(self.attributes_to_extract)}.

        Return the answer as a JSON object with keys for each attribute.
        If an attribute is not found, its value should be "Not found".

        Text:
        ---
        {text[:4000]}
        ---

        JSON Output:
        """
        try:
            response_dict = await run_llm(prompt, purpose="attribute_profiling")
            response_text = response_dict.get("result", "{}")
            # Basic cleanup for JSON that might be in a code block
            if "```json" in response_text:
                response_text = response_text.split("```json\n")[1].split("```")[0]

            import json
            attributes = json.loads(response_text)
            return attributes
        except Exception:
            return {attr: "Error" for attr in self.attributes_to_extract}

    async def analyze(self, dataset):
        """Adds a profile of attributes to each entity."""
        for profile in dataset:
            full_text = profile.get('bio', '') or ''
            posts = profile.get('posts', [])
            if posts:
                full_text += "\n" + "\n".join([post.get('text', '') for post in posts if post.get('text')])

            attributes = await self._get_attributes(full_text)

            if 'analysis' not in profile:
                profile['analysis'] = {}
            profile['analysis']['attributes'] = attributes

        return dataset
