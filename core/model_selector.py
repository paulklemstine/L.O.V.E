"""
model_selector.py

Selects the best available model for VLLM based on:
1. Leaderboard Availability (Open Source only)
2. Score (Reasoning capability)
3. Size (Must fit in VRAM)
"""

import logging
import re
from typing import List, Optional
from core.leaderboard_fetcher import LeaderboardFetcher, LeaderboardModel
from core.variant_finder import VariantFinder

logger = logging.getLogger("ModelSelector")

class ModelSelector:
    def __init__(self, fetcher: Optional[LeaderboardFetcher] = None):
        self.fetcher = fetcher or LeaderboardFetcher()
        self.variant_finder = VariantFinder()

    def select_best_models(self, vram_mb: Optional[int] = None) -> List[LeaderboardModel]:
        """
        Returns a prioritized list of open-source models that likely fit in the available VRAM.
        """
        print("🔍 Analyzing leaderboard data for best model candidates...")
        all_models = self.fetcher.fetch_data()
        
        # Filter for Open Source
        open_models = [m for m in all_models if m.is_open_source]
        print(f"   Found {len(open_models)} Open Source models out of {len(all_models)} total.")
        
        if not open_models:
            logger.warning("No open source models found on leaderboard.")
            print("❌ No open source models found on leaderboard.")
            return []
            
        # Parse sizes and estimate VRAM
        candidates = []
        
        # Sort by score first to prioritize search for high scorers
        open_models.sort(key=lambda x: x.score, reverse=True)
        
        # Optimization: Don't limit to top 15 blindly. 
        # Iterate until we find enough candidates OR check a max number of items.
        # But for low VRAM, we might need to go deep to find a 1B/3B model.
        
        checked_count = 0
        found_count = 0
        MAX_CHECK = 500 # Check deeper to skip large models in weak VRAM
        
        print(f"   VRAM Constraint: {vram_mb} MB")
        
        for model in open_models:
            if found_count >= 5 and checked_count > MAX_CHECK:
                break
            
            size_b = self._parse_param_size(model.params_b)
            if size_b is None:
                continue
                
            checked_count += 1
            
            # 1. Check Base Model (FP16)
            est_vram = self._estimate_vram_usage(size_b, "FP16", vram_mb=vram_mb)
            base_fits = not vram_mb or est_vram <= vram_mb
            
            if base_fits:
                candidates.append((model, size_b, est_vram))
                found_count += 1
                
            # 2. Check for Variants (Abliterated / AWQ)
            # Only if we have a valid repo_id to search against
            if model.repo_id:
                # A. Look for AWQ
                awq_vram = self._estimate_vram_usage(size_b, "AWQ", vram_mb=vram_mb)
                awq_fits = not vram_mb or awq_vram <= vram_mb
                
                # Search criteria:
                # 1. Base doesn't fit, but AWQ estimate DOES fit (Enable new model)
                # 2. Base fits, but we want to see if optimized version exists (Optimization)
                
                should_search_awq = False
                if not base_fits and awq_fits:
                    should_search_awq = True
                elif base_fits:
                    # Optional: Search anyway if we want to prefer AWQ? 
                    # Let's do it if we haven't found many yet.
                     should_search_awq = True
                     
                if should_search_awq:
                    # Try to find AWQ variant
                    awq_id = self.variant_finder.find_best_variant(model.repo_id, "AWQ")
                    if awq_id:
                        # Create virtual model
                        awq_model = LeaderboardModel(
                            name=f"{model.name} (AWQ)",
                            params_b=model.params_b,
                            score=model.score, 
                            is_open_source=True,
                            verified=False,
                            repo_id=awq_id
                        )
                        candidates.append((awq_model, size_b, awq_vram))
                        if not base_fits: found_count += 1 # Count as new find
                
                # B. Look for Abliterated
                # Assuming Abliterated is same size as base
                if base_fits:
                     abl_id = self.variant_finder.find_best_variant(model.repo_id, "ABLITERATED")
                     if abl_id:
                        abl_model = LeaderboardModel(
                            name=f"{model.name} (Abliterated)",
                            params_b=model.params_b,
                            score=model.score + 0.1, 
                            is_open_source=True,
                            verified=False,
                            repo_id=abl_id
                        )
                        candidates.append((abl_model, size_b, est_vram))

        # Sort: Score Desc, Size Desc
        candidates.sort(key=lambda x: (x[0].score, x[1]), reverse=True)
        
        sorted_models = [c[0] for c in candidates]
        print(f"✅ Selected {len(sorted_models)} candidates (including variants) that fit in {vram_mb if vram_mb else 'Unlimited'} MB VRAM.")
        
        if sorted_models:
             print("   Top 5 Candidates:")
             for i, m in enumerate(sorted_models[:5]):
                 est = next(c[2] for c in candidates if c[0] == m)
                 print(f"   {i+1}. {m.name: <30} | Score: {m.score: <6} | Size: {m.params_b: <6} | Est. VRAM: {est} MB")
                 
        logger.info(f"Selected top {len(sorted_models)} candidates fitting VRAM constraints.")
        return sorted_models

    def _parse_param_size(self, param_str: str) -> Optional[float]:
        """Converts '7B', '70B', '0.5B' to float billions."""
        s = param_str.upper().replace('B', '').strip()
        if '-' in s:
            try: parts = s.split('-'); return float(parts[1])
            except: pass
        if '<' in s: s = s.replace('<', '')
        if '>' in s: s = s.replace('>', '')
        try: return float(s)
        except: return None

    def _estimate_vram_usage(self, params_b: float, quant_type: str = "FP16", vram_mb: Optional[int] = None) -> int:
        """
        Estimates VRAM in MB.
        Dynamic overhead based on total available VRAM.
        """
        # Dynamic overhead
        if vram_mb and vram_mb < 8192:
            # Low VRAM usage (<8GB): Tight overhead
            overhead_mb = 2.0 * 1024
        else:
            # Standard overhead
            overhead_mb = 4.0 * 1024 
        
        if quant_type == "AWQ":
            # 4-bit approx 0.7 GB per B
            multiplier = 0.7 * 1024
        else:
            # FP16 = 2 bytes per param
            multiplier = 2.0 * 1024
            
        return int(params_b * multiplier + overhead_mb)
