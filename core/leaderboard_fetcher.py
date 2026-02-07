"""
leaderboard_fetcher.py

Fetches and parses model data from the Open LLM Leaderboard.
Source: https://huggingface.co/datasets/open-llm-leaderboard/contents
"""

import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger("LeaderboardFetcher")

LEADERBOARD_DATASET = "open-llm-leaderboard/contents"

@dataclass
class LeaderboardModel:
    name: str
    params_b: str
    score: float
    is_open_source: bool
    verified: bool
    repo_id: Optional[str] = None

class LeaderboardFetcher:
    def __init__(self, dataset_name: str = LEADERBOARD_DATASET):
        self.dataset_name = dataset_name
        self._cache = None

    def fetch_data(self) -> List[LeaderboardModel]:
        """
        Fetches the leaderboard data from HuggingFace datasets and parses into model objects.
        Uses the Open LLM Leaderboard dataset which contains text-only LLM evaluations.
        """
        if self._cache is not None:
            return self._cache
            
        try:
            print(f"⬇️  Fetching leaderboard data from HuggingFace: {self.dataset_name}...")
            logger.info(f"Fetching leaderboard data from {self.dataset_name}...")
            
            # Try to use datasets library
            try:
                from datasets import load_dataset
                ds = load_dataset(self.dataset_name, split="train")
                return self._parse_hf_dataset(ds)
            except ImportError:
                logger.warning("datasets library not available, falling back to API")
                return self._fetch_via_api()
            except Exception as e:
                logger.warning(f"Failed to load via datasets library: {e}, falling back to API")
                return self._fetch_via_api()
                
        except Exception as e:
            print(f"❌ Failed to fetch leaderboard data: {e}")
            logger.error(f"Failed to fetch leaderboard data: {e}")
            return []

    def _parse_hf_dataset(self, ds) -> List[LeaderboardModel]:
        """Parse the HuggingFace dataset into LeaderboardModel objects."""
        models_map = {}
        
        print(f"   Processing {len(ds)} entries from leaderboard...")
        
        for row in ds:
            try:
                # The Open LLM Leaderboard dataset has these columns:
                # - fullname or model_name: The model identifier
                # - Average or score: The average score
                # - Params or params: Parameter count
                # - Type: Model type (pretrained, fine-tuned, etc.)
                
                model_name = row.get('fullname') or row.get('model_name') or row.get('Model', '')
                if not model_name:
                    continue
                
                # Extract repo_id (should be in fullname as org/model format)
                repo_id = model_name if '/' in model_name else None
                
                # Get score - try different column names
                score = 0.0
                for score_col in ['Average', 'score', 'average', 'Average ⬆️']:
                    if score_col in row and row[score_col] is not None:
                        try:
                            score = float(row[score_col])
                            break
                        except (ValueError, TypeError):
                            continue
                
                # Get params
                params = 'Unknown'
                for param_col in ['#Params (B)', 'Params', 'params', 'Parameters']:
                    if param_col in row and row[param_col] is not None:
                        params = str(row[param_col])
                        break
                
                # Get architecture/type info
                model_type = row.get('Type', row.get('type', ''))
                
                # All models in the Open LLM Leaderboard are open source
                is_open = True
                
                # Clean the name for display
                clean_name = self._clean_name(model_name)
                
                if clean_name not in models_map:
                    models_map[clean_name] = LeaderboardModel(
                        name=clean_name,
                        params_b=params,
                        score=score,
                        is_open_source=is_open,
                        verified=True,  # All are verified on this leaderboard
                        repo_id=repo_id
                    )
                    
            except Exception as e:
                logger.debug(f"Error parsing row: {e}")
                continue

        results = list(models_map.values())
        print(f"✅ Parsed {len(results)} unique models from Open LLM Leaderboard.")
        logger.info(f"Parsed {len(results)} unique models from leaderboard.")
        
        self._cache = results
        return results
    
    def _fetch_via_api(self) -> List[LeaderboardModel]:
        """Fallback: Fetch via HuggingFace API if datasets library unavailable."""
        import requests
        
        # Try to get the parquet file directly
        api_url = f"https://huggingface.co/api/datasets/{self.dataset_name}"
        
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            
            # This gives us dataset metadata, not actual data
            # For actual data we need the datasets library
            print("⚠️ API fallback: Limited data available. Install 'datasets' for full access.")
            logger.warning("Using API fallback - limited functionality")
            
            # Return empty list if datasets library not available
            # The model selector will fall back to hardcoded defaults
            return []
            
        except Exception as e:
            logger.error(f"API fallback also failed: {e}")
            return []

    def _clean_name(self, raw_name: str) -> str:
        """Removes HTML/Markdown artifacts from model name."""
        # Simple extraction
        if '<a' in raw_name and '>' in raw_name:
            try:
                part = raw_name.split('>')[1]
                return part.split('<')[0].strip()
            except:
                pass
        
        if '[' in raw_name and '](' in raw_name:
            try:
                return raw_name.split('[')[1].split(']')[0].strip()
            except:
                pass
                
        return raw_name.strip()
