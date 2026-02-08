import logging
from typing import List, Optional, Dict
from huggingface_hub import HfApi

logger = logging.getLogger("VariantFinder")

class VariantFinder:
    def __init__(self):
        self.api = HfApi()

    def find_best_variant(self, repo_id: str, search_type: str = "AWQ") -> Optional[str]:
        """
        Searches for a variant of the given repo_id.
        search_type: "AWQ" or "ABLITERATED"
        Returns the repo_id of the best match or None.
        """
        if not repo_id or '/' not in repo_id:
            return None
            
        base_name = repo_id.split('/')[-1]
        
        # Define search keywords based on type
        # Define search keywords based on type
        if search_type == "AWQ":
            keywords = ["AWQ", "4bit", "4-bit"]
        elif search_type == "GPTQ":
             keywords = ["GPTQ", "4bit", "Int4", "4-bit"]
        elif search_type == "ABLITERATED":
            keywords = ["abliterated", "uncensored"]
        else:
            return None
            
        logger.info(f"Searching for {search_type} variant of {base_name}...")
        
        candidates = []
        
        import time
        from huggingface_hub.utils import HfHubHTTPError
        
        for kw in keywords:
            query = f"{base_name} {kw}"
            try:
                # Rate limit protection: Sleep briefly before request
                time.sleep(0.5) 
                
                models = self.api.list_models(search=query, limit=5, sort="downloads", direction=-1)
                for m in models:
                    # Basic filtering: strictly ensure the keyword is in the ID (case insensitive)
                    if kw.lower() in m.modelId.lower():
                        candidates.append(m)
            except HfHubHTTPError as e:
                if e.response.status_code == 429:
                    logger.warning(f"⚠️ HF Rate Limit (429) hit checking {query}. Backing off.")
                    time.sleep(5) # Wait longer if we hit a limit
                    # Optional: Break completely if we really can't continue, but sleep might handle transient spikes
                else:
                    logger.warning(f"Search failed for {query}: {e}")
            except Exception as e:
                logger.warning(f"Search failed for {query}: {e}")
                
        if not candidates:
            return None
            
        # Deduplicate candidates
        unique_candidates = {m.modelId: m for m in candidates}.values()
        
        # Sort by downloads again to pick the most popular valid one
        sorted_candidates = sorted(unique_candidates, key=lambda x: x.downloads, reverse=True)
        
        best = sorted_candidates[0]
        logger.info(f"✨ Found Best {search_type} Variant: {best.modelId} ({best.downloads} downloads)")
        return best.modelId
