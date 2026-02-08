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

    # Vision/Multimodal model patterns to exclude (don't support text-only chat)
    VISION_MODEL_PATTERNS = [
        'VL', 'Vision', 'Ovis', 'LLaVA', 'Llava', 'llava',
        'InternVL', 'CogVLM', 'MiniCPM-V', 'Qwen-VL', 'QwenVL',
        'mPLUG', 'Fuyu', 'BLIP', 'Flamingo', 'PaLI', 'Idefics',
        'LMM', 'Multimodal', 'Image', 'Visual'
    ]
    
    def _is_vision_model(self, model: LeaderboardModel) -> bool:
        """Check if model is a vision/multimodal model that doesn't support text-only chat."""
        name_upper = model.name.upper()
        repo_upper = (model.repo_id or "").upper()
        
        for pattern in self.VISION_MODEL_PATTERNS:
            pattern_upper = pattern.upper()
            if pattern_upper in name_upper or pattern_upper in repo_upper:
                return True
        return False
    
    def select_best_models(self, vram_mb: Optional[int] = None) -> List[LeaderboardModel]:
        """
        Returns a prioritized list of open-source models that likely fit in the available VRAM.
        Unquantized > GPTQ > (Exclude others for now per user request)
        Excludes vision/multimodal models that don't support text-only chat.
        """
        print("🔍 Analyzing leaderboard data for best model candidates...")
        all_models = self.fetcher.fetch_data()
        
        # Filter for Open Source
        open_models = [m for m in all_models if m.is_open_source]
        print(f"   Found {len(open_models)} Open Source models out of {len(all_models)} total.")
        
        # Filter out vision/multimodal models
        text_only_models = [m for m in open_models if not self._is_vision_model(m)]
        excluded_count = len(open_models) - len(text_only_models)
        if excluded_count > 0:
            print(f"   Excluded {excluded_count} vision/multimodal models (text-only required).")
        open_models = text_only_models
        
        if not open_models:
            logger.warning("No open source models found on leaderboard.")
            print("❌ No open source models found on leaderboard.")
            return []
            
        # Parse sizes and estimate VRAM
        candidates = []
        
        # Sort by score first
        open_models.sort(key=lambda x: x.score, reverse=True)
        
        checked_count = 0
        found_count = 0
        # User requested to try "all" models. Expanding limits significantly.
        MAX_CHECK = 2000 
        MAX_FOUND = 50
        
        print(f"   VRAM Constraint: {vram_mb} MB")
        
        for model in open_models:
            if found_count >= MAX_FOUND and checked_count > MAX_CHECK:
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
            elif model.repo_id:
                # 2. Base doesn't fit, check for GPTQ variant
                # Only check if base doesn't fit, as preferred by user
                gptq_vram = self._estimate_vram_usage(size_b, "GPTQ", vram_mb=vram_mb)
                gptq_fits = not vram_mb or gptq_vram <= vram_mb
                
                if gptq_fits:
                     gptq_id = self.variant_finder.find_best_variant(model.repo_id, "GPTQ")
                     if gptq_id:
                        gptq_model = LeaderboardModel(
                            name=f"{model.name} (GPTQ)",
                            params_b=model.params_b,
                            score=model.score, # Assume similar score
                            is_open_source=True,
                            verified=False,
                            repo_id=gptq_id
                        )
                        candidates.append((gptq_model, size_b, gptq_vram))
                        found_count += 1

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
        
        if quant_type in ["AWQ", "GPTQ"]:
            # 4-bit approx 0.7-0.8 GB per B. Let's be safe with 0.75
            multiplier = 0.75 * 1024
        else:
            # FP16 = 2 bytes per param
            multiplier = 2.0 * 1024
            
        return int(params_b * multiplier + overhead_mb)
