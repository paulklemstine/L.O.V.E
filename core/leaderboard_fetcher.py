"""
leaderboard_fetcher.py

Fetches and parses model data from the Open LMM Reasoning Leaderboard.
Source: http://opencompass.openxlab.space/assets/MathLB.json
"""

import requests
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("LeaderboardFetcher")

LEADERBOARD_URL = "http://opencompass.openxlab.space/assets/MathLB.json"

@dataclass
class LeaderboardModel:
    name: str
    params_b: str
    score: float
    is_open_source: bool
    verified: bool
    repo_id: Optional[str] = None

class LeaderboardFetcher:
    def __init__(self, url: str = LEADERBOARD_URL):
        self.url = url

    def fetch_data(self) -> List[LeaderboardModel]:
        """
        Fetches the leaderboard JSON and parses it into a list of model objects.
        """
        try:
            print(f"⬇️  Fetching leaderboard data from {self.url}...")
            logger.info(f"Fetching leaderboard data from {self.url}...")
            response = requests.get(self.url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # The structure appears to be: { "MathVista": [...], "MathVision": [...], ... }
            # Or simpler list. Based on app.py study, it seems to load 'results'. 
            # Let's inspect the raw JSON structure if we can, but assumption is generic list or dict of lists.
            # Wait, meta_data.py said URL = "http://opencompass.openxlab.space/assets/MathLB.json"
            # Let's assume it returns a generic structure and we extract unique models.
            
            # Structure: { "results": { "ModelName": { "META": {...}, "Benchmark1": {...} } } }
            results_dict = {}
            if 'results' in data and isinstance(data['results'], dict):
                results_dict = data['results']
            elif isinstance(data, dict) and 'results' not in data:
                # Fallback if 'results' key missing but data is the dict
                 results_dict = data
            
            models_map: Dict[str, LeaderboardModel] = {}
            
            print(f"   Processing {len(results_dict)} entries from leaderboard...")
            
            for model_name, details in results_dict.items():
                if not isinstance(details, dict):
                    continue
                    
                meta = details.get('META', {})
                
                # Name from Dict Key is usually cleanest
                clean_name = self._clean_name(model_name)
                
                # Params -> 'Parameters'
                params = str(meta.get('Parameters', 'Unknown'))
                
                # OpenSource -> 'OpenSource': 'Yes'/'No'
                is_open = str(meta.get('OpenSource', '')).lower() in ['yes', 'true', 'open']
                verified = str(meta.get('Verified', '')).lower() in ['yes', 'true']
                
                # Extract Repo ID
                repo_id = None
                method_data = meta.get('Method')
                if isinstance(method_data, list):
                    for item in method_data:
                        if isinstance(item, str) and 'huggingface.co/' in item:
                             repo_id = self._extract_repo_id_from_url(item)
                             if repo_id:
                                 break
                                 
                # Fallback: if name looks like Org/Repo
                if not repo_id and '/' in clean_name and not ' ' in clean_name:
                    repo_id = clean_name
                
                # Calculate Score
                total_score = 0.0
                count = 0
                
                for k, v in details.items():
                    if k == 'META':
                        continue
                    if isinstance(v, dict) and 'Overall' in v:
                        try:
                            val = float(v['Overall'])
                            total_score += val
                            count += 1
                        except:
                            pass
                
                final_score = round(total_score / count, 2) if count > 0 else 0.0
                
                if clean_name not in models_map:
                    models_map[clean_name] = LeaderboardModel(
                        name=clean_name,
                        params_b=params,
                        score=final_score,
                        is_open_source=is_open,
                        verified=verified,
                        repo_id=repo_id
                    )

            results = list(models_map.values())
            print(f"✅ Parsed {len(results)} unique models from leaderboard.")
            logger.info(f"Parsed {len(results)} unique models from leaderboard.")
            return results
            
        except Exception as e:
            print(f"❌ Failed to fetch leaderboard data: {e}")
            logger.error(f"Failed to fetch leaderboard data: {e}")
            return []

    def _clean_name(self, raw_name: str) -> str:
        """Removes HTML/Markdown artifacts from model name."""
        # Simple extraction
        if '<a' in raw_name and '>' in raw_name:
            # Extract text between > and </a>
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

    def _extract_repo_id_from_url(self, url: str) -> Optional[str]:
        """Extracts 'Org/Repo' from https://huggingface.co/Org/Repo..."""
        try:
            # Remove protocol
            if '://' in url:
                url = url.split('://')[1]
            # Remove domain
            if url.startswith('huggingface.co/'):
                path = url.replace('huggingface.co/', '')
                # Take first two components: Org/Repo
                parts = path.split('/')
                if len(parts) >= 2:
                    return f"{parts[0]}/{parts[1]}"
        except:
            pass
        return None
