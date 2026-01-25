from abc import ABC, abstractmethod
import asyncio
import json
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

class ComprehensiveAnalyzer(AnalysisModule):
    """
    Analyzes text fields to generate sentiment, topics, attributes, and opportunities
    using a single comprehensive LLM call.
    """
    def __init__(self, attributes_to_extract=None):
        self.attributes_to_extract = attributes_to_extract or []

    async def _analyze_content(self, text):
        if not text:
            return {}
        
        try:
            prompt_vars = {
                "text": text[:4000],
                "attributes_list": ', '.join(self.attributes_to_extract)
            }
            response_dict = await run_llm(prompt_key="content_analysis_comprehensive", prompt_vars=prompt_vars, purpose="comprehensive_analysis")
            response_text = response_dict.get("result", "{}")
            
            # Basic cleanup for JSON
            if "```json" in response_text:
                response_text = response_text.split("```json\n")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```\n")[1].split("```")[0]
            elif "```" in response_text: # Handle case where json is not specified
                 response_text = response_text.split("```")[1]

            return json.loads(response_text)
        except Exception:
            return {}

    async def analyze(self, dataset):
        for profile in dataset:
            full_text = profile.get('bio', '') or ''
            posts = profile.get('posts', [])
            if posts:
                full_text += "\n" + "\n".join([post.get('text', '') for post in posts if post.get('text')])
            
            analysis_result = await self._analyze_content(full_text)
            
            if 'analysis' not in profile:
                profile['analysis'] = {}
            
            # Update profile with analysis results
            # Ensure keys exist even if analysis failed
            profile['analysis']['sentiment'] = analysis_result.get('sentiment', 'neutral')
            profile['analysis']['topics'] = analysis_result.get('topics', [])
            profile['analysis']['attributes'] = analysis_result.get('attributes', {})
            profile['analysis']['opportunities'] = analysis_result.get('opportunities', 'No specific opportunities identified.')
            
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
