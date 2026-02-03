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
        if search_type == "AWQ":
            keywords = ["AWQ", "4bit", "4-bit"]
            # We prefer models that differ mostly by these tags
        elif search_type == "ABLITERATED":
            keywords = ["abliterated", "uncensored"]
        else:
            return None
            
        logger.info(f"Searching for {search_type} variant of {base_name}...")
        
        # We search specifically for models containing the base name and the keyword
        # This is a bit loose, so we need to verify similarity?
        # A good heuristic: filtered search by name
        
        candidates = []
        
        for kw in keywords:
            query = f"{base_name} {kw}"
            try:
                models = self.api.list_models(search=query, limit=5, sort="downloads", direction=-1)
                for m in models:
                    # Basic filtering: strictly ensure the keyword is in the ID (case insensitive)
                    if kw.lower() in m.modelId.lower():
                        candidates.append(m)
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
