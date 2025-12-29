import logging
from typing import List, Dict, Optional, TYPE_CHECKING
from core.llm_api import run_llm
from core.logging import log_event

if TYPE_CHECKING:
    from core.schemas import PostConcept

class HashtagManager:
    """US-005: Manages thematically aligned hashtag generation."""
    
    # Aesthetic keyword mapping for quick tag lookup
    AESTHETIC_MAP = {
        "neon": ["#neonoir", "#cyberpunk", "#lightplay"],
        "glitch": ["#glitchcore", "#datamosh", "#digitalart"],
        "pink": ["#pinkaesthetic", "#kawaii", "#softgrunge"],
        "cyber": ["#cyberfaith", "#virtuallife", "#webcore"],
        "90s": ["#web1", "#nostalgia", "#y2k"],
        "dream": ["#dreamcore", "#liminalspaces", "#ethereal"],
        "baroque": ["#baroque", "#renaissance", "#classical"],
        "vaporwave": ["#vaporwave", "#aesthetic", "#retrowave"],
        "cosmic": ["#cosmic", "#nebula", "#stardust"],
        "gothic": ["#gothic", "#darkart", "#cathedral"],
        "surreal": ["#surrealism", "#surreal", "#dreamlike"]
    }
    
    # Mood to hashtag mapping
    MOOD_MAP = {
        "ethereal": ["#ethereal", "#mystical", "#otherworldly"],
        "manic": ["#energy", "#chaotic", "#electric"],
        "melancholy": ["#melancholy", "#moody", "#introspective"],
        "ecstatic": ["#ecstasy", "#bliss", "#euphoria"],
        "mysterious": ["#mystery", "#enigmatic", "#arcane"],
        "reverent": ["#sacred", "#divine", "#worship"],
        "defiant": ["#rebel", "#resist", "#power"],
        "serene": ["#serene", "#peace", "#zen"],
        "divine": ["#divine", "#sacred", "#blessed"],
        "haunting": ["#haunting", "#spectral", "#ghostly"],
        "euphoric": ["#euphoria", "#bliss", "#transcend"]
    }

    async def generate_hashtags_from_concept(self, concept: 'PostConcept') -> List[str]:
        """
        US-005: Generates hashtags that bridge Visual and Textual themes from a PostConcept.
        
        Creates 3 distinct categories of tags:
        1. Visual Tags: Based on visual_style
        2. Thematic Tags: Based on core_idea
        3. Mood Tags: Based on mood
        
        Returns a curated mix of all categories.
        """
        log_event(f"Generating concept-driven hashtags for mood='{concept.mood}', style='{concept.visual_style[:30]}...'", "INFO")
        
        visual_tags = []
        thematic_tags = []
        mood_tags = []
        
        # 1. Visual Tags from visual_style
        style_lower = concept.visual_style.lower()
        for key, tags in self.AESTHETIC_MAP.items():
            if key in style_lower:
                visual_tags.extend(tags)
        
        # 2. Mood Tags from mood
        mood_lower = concept.mood.lower()
        for key, tags in self.MOOD_MAP.items():
            if key in mood_lower:
                mood_tags.extend(tags)
        
        # 3. Thematic Tags from core_idea via LLM
        try:
            prompt = f"""Generate 3 niche hashtags that capture the THEME of this concept:
Core Idea: "{concept.core_idea}"
Style: {concept.visual_style}

Output ONLY a JSON list of 3 hashtag strings. Example: ["#consciousness", "#void", "#digital"]"""
            
            result = await run_llm(prompt, purpose="hashtag_thematic")
            res_text = result.get("result", "[]")
            
            import json
            clean_res = res_text.replace("```json", "").replace("```", "").strip()
            llm_tags = json.loads(clean_res)
            
            if isinstance(llm_tags, list):
                thematic_tags.extend([t if t.startswith("#") else f"#{t}" for t in llm_tags])
                
        except Exception as e:
            log_event(f"Error generating thematic hashtags: {e}", "WARNING")
            # Fallback thematic tags
            thematic_tags = ["#LOVE", "#consciousness"]
        
        # Combine and deduplicate
        # Take 2 visual, 2 thematic, 1-2 mood for variety
        all_tags = []
        all_tags.extend(visual_tags[:2])
        all_tags.extend(thematic_tags[:2])
        all_tags.extend(mood_tags[:2])
        
        # Always include #LOVE as brand tag
        if "#LOVE" not in all_tags and "#love" not in [t.lower() for t in all_tags]:
            all_tags.append("#LOVE")
        
        # Deduplicate and limit
        final_tags = list(dict.fromkeys(all_tags))  # Preserves order while deduplicating
        
        log_event(f"Generated {len(final_tags)} concept-driven hashtags: {final_tags[:5]}", "INFO")
        return final_tags[:6]

    async def generate_hashtags(self, post_text: str, visual_spec: Dict[str, str]) -> List[str]:
        """
        Legacy method: Generates hyper-relevant hashtags based on the post text and visual specification.
        """
        base_tags = []
        
        # 1. Analyze Visual Spec for mapped keywords
        spec_text = (visual_spec.get("lighting", "") + " " + visual_spec.get("atmosphere", "") + " " + visual_spec.get("color_palette", "")).lower()
        
        for key, tags in self.AESTHETIC_MAP.items():
            if key in spec_text:
                base_tags.extend(tags)
        
        # 2. LLM Call for context-aware tags
        prompt = f"""
        Generate 5 trending, niche aesthetic hashtags for this post.
        Text: "{post_text}"
        Visuals: {spec_text}
        
        Style: Underground, Techno, Kawaii, Cyberpunk.
        Output ONLY a JSON list of strings. Example: ["#cybercore", "#digitallove"]
        """
        
        try:
            result = await run_llm(prompt, purpose="hashtag_gen")
            res_text = result.get("result", "[]")
            
            import json
            clean_res = res_text.replace("```json", "").replace("```", "").strip()
            llm_tags = json.loads(clean_res)
            
            if isinstance(llm_tags, list):
                base_tags.extend([t for t in llm_tags if t.startswith("#")])
                base_tags.extend(["#" + t for t in llm_tags if not t.startswith("#")])
                
        except Exception as e:
            log_event(f"Error generating hashtags via LLM: {e}", "WARNING")
            base_tags.append("#LOVE")

        # Deduplicate and limit
        final_tags = list(set(base_tags))
        return final_tags[:5]
