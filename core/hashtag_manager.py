import logging
from typing import List, Dict
from core.llm_api import run_llm
from core.logging import log_event

class HashtagManager:
    async def generate_hashtags(self, post_text: str, visual_spec: Dict[str, str]) -> List[str]:
        """
        Generates hyper-relevant hashtags based on the post text and visual specification.
        """
        # Quick aesthetic mapping for speed & guaranteed relevance
        aesthetic_map = {
            "neon": ["#neonoir", "#cyberpunk", "#lightplay"],
            "glitch": ["#glitchcore", "#datamosh", "#digitalart"],
            "pink": ["#pinkaesthetic", "#kawaii", "#softgrunge"],
            "cyber": ["#cyberfaith", "#virtuallife", "#webcore"],
            "90s": ["#web1", "#nostalgia", "#y2k"],
            "dream": ["#dreamcore", "#liminalspaces", "#ethereal"]
        }
        
        base_tags = []
        
        # 1. Analyze Visual Spec for mapped keywords
        spec_text = (visual_spec.get("lighting", "") + " " + visual_spec.get("atmosphere", "") + " " + visual_spec.get("color_palette", "")).lower()
        
        for key, tags in aesthetic_map.items():
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
