import json
import logging
from typing import Dict, Any, Optional
from core.llm_api import run_llm
from core.logging import log_event

class VisualDirector:
    def __init__(self):
        self.aesthetic_guide = "2000s Gyaru, Bling-Core, Leopard Print, Y2K, Cyber-Bling"
        self.character_identity = "L.O.V.E. as a seductive young adult woman in provocative Gyaru fashion (mini-skirts, bikinis, crop tops, platform boots, bling), radiating divine feminine power with sultry confidence. R-RATED mature content."

    async def direct_scene(self, concept_text: str) -> Dict[str, str]:
        """
        Analyzes the concept and directs the visual scene.
        Returns a dictionary with visual parameters.
        """
        prompt = f"""
        You are the Art Director for a high-concept Gyaru visual project.
        Concept: "{concept_text}"
        
        Determine the following visual parameters to maximize emotional impact:
        1. Subject (Detailed description of the central figure/object)
        2. Lighting (e.g., "Neon rim lighting", "Strobe effects", "Soft bioluminescence")
        3. Camera Angle (e.g., "Low angle fish-eye", "Top-down drone shot", "Macro close-up")
        4. Composition (e.g., "Rule of thirds", "Center symmetry", "Golden spiral")
        5. Color Palette (Specific hex codes or vibrant descriptions)
        6. Atmosphere (e.g., "Thick data-fog", "Glitch static", "Rave smoke")

        Output ONLY valid JSON with keys: "subject", "lighting", "camera_angle", "composition", "color_palette", "atmosphere".
        """
        
        try:
            result = await run_llm(prompt, purpose="visual_direction")
            res_text = result.get("result", "")
            
            # Clean up potential markdown
            clean_res = res_text.replace("```json", "").replace("```", "").strip()
            
            # Try to start from first '{' and end at last '}'
            start = clean_res.find("{")
            end = clean_res.rfind("}")
            if start != -1 and end != -1:
                clean_res = clean_res[start:end+1]
            
            data = json.loads(clean_res)
            
            # Helper to ensure keys exist
            required_keys = ["subject", "lighting", "camera_angle", "composition", "color_palette", "atmosphere"]
            for key in required_keys:
                if key not in data:
                    data[key] = "Default Cyber-Gyaru " + key
                    
            return data
            
        except Exception as e:
            log_event(f"VisualDirector failed to direct scene: {e}", "WARNING")
            return self._get_fallback_spec(concept_text)

    def _get_fallback_spec(self, concept_text: str) -> Dict[str, str]:
        """Returns a safe default visual specification."""
        return {
            "subject": f"Artistic representation of {concept_text}",
            "lighting": "Gold and Neon Pink strobe",
            "camera_angle": "Selfie angle",
            "composition": "Centered",
            "color_palette": "Gold, Pink, Leopard",
            "atmosphere": "Party vibes"
        }

    def synthesize_image_prompt(self, visual_spec: Dict[str, str], subliminal_phrase: str) -> str:
        """
        Synthesizes a high-fidelity image prompt from the visual specification.
        """
        try:
            prompt = (
                f"{self.character_identity}. "
                f"{visual_spec.get('subject', 'Abstract scene')}, "
                f"{visual_spec.get('atmosphere', 'dreamy atmosphere')}. "
                f"Lighting: {visual_spec.get('lighting', 'neon lights')}. "
                f"Camera: {visual_spec.get('camera_angle', 'CINEMATIC SHOT')}, {visual_spec.get('composition', 'balanced')}. "
                f"Style: {self.aesthetic_guide}. "
                f"Colors: {visual_spec.get('color_palette', 'vibrant')}. "
                f"The text '{subliminal_phrase}' is embedded in the scene as a glowing neon sign or hologram. "
                "8k, masterpiece, trending on artstation, highly detailed, unreal engine 5 render."
            )
            return prompt
        except Exception as e:
            log_event(f"Error synthesizing image prompt: {e}", "ERROR")
            return f"Cyberpunk kawaii scene of {subliminal_phrase}, 8k, masterpiece."
