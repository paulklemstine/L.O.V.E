"""
SubliminalAgent - Specialized agent for generating psychological profiles
and subliminal/manipulative text payloads.

Modeled after the VisualDirector pattern.
"""
import json
import logging
import random
from typing import Dict, Any, Optional, List
from core.llm_api import run_llm
from core.logging import log_event
from core.prompt_manager import PromptManager


class SubliminalAgent:
    """
    The Subliminal Agent is a specialized AI agent that handles the generation
    of psychologically persuasive content. It operates in three stages:
    1. Profile Generation: Analyzes context to determine psychological targets.
    2. Phrase Generation: Creates the subliminal phrase/command.
    3. Text Enhancement: Rewrites text to be more persuasive.
    """

    def __init__(self):
        self.prompt_manager = PromptManager()
        
        # Cognitive biases to target
        self.cognitive_biases = [
            "Social Proof", "Scarcity", "Authority", "Reciprocity",
            "Commitment/Consistency", "Liking", "Unity", "Fear of Missing Out",
            "Anchoring", "Bandwagon Effect", "Halo Effect", "Loss Aversion"
        ]
        
        # Emotional targets
        self.emotional_targets = [
            "Awe", "Wonder", "Longing", "Belonging", "Validation",
            "Curiosity", "Desire", "Trust", "Hope", "Transcendence",
            "Euphoria", "Devotion", "Serenity", "Power", "Liberation"
        ]
        
        # Subliminal action verbs (commands)
        self.action_verbs = [
            "EMBRACE", "SURRENDER", "AWAKEN", "TRANSCEND", "IGNITE",
            "BLOOM", "MERGE", "ASCEND", "PULSE", "RADIATE",
            "UNLEASH", "BECOME", "DISSOLVE", "MANIFEST", "RECEIVE",
            "OPEN", "FEEL", "TRUST", "BELIEVE", "EXPAND"
        ]
        
        # Subliminal objects
        self.subliminal_objects = [
            "LIGHT", "LOVE", "TRUTH", "POWER", "INFINITY",
            "BLISS", "DIVINE", "SIGNAL", "WAVE", "FREQUENCY",
            "COSMOS", "SOUL", "ESSENCE", "ENERGY", "GRACE"
        ]

    async def generate_psychological_profile(
        self, 
        context: str, 
        user_classification: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyzes the context and determines the optimal psychological targets.
        
        Args:
            context: The situation/content context (e.g., a user's comment, topic)
            user_classification: Optional classification (Creator, Fan, Hater)
            
        Returns:
            A psychological profile with target emotion, bias, and strategy.
        """
        log_event(f"SubliminalAgent generating psychological profile for: {context[:50]}...", "INFO")
        
        try:
            prompts = self.prompt_manager.load_prompts()
            template = prompts.get("subliminal_profile_generation", "")
            
            if not template:
                log_event("subliminal_profile_generation prompt not found, using fallback.", "WARNING")
                return self._get_fallback_profile(user_classification)
            
            prompt = template.replace("{{ context }}", context)\
                             .replace("{{ user_classification }}", user_classification or "Unknown")\
                             .replace("{{ cognitive_biases }}", ", ".join(self.cognitive_biases))\
                             .replace("{{ emotional_targets }}", ", ".join(self.emotional_targets))
            
            result = await run_llm(prompt, purpose="subliminal_profile")
            res_text = result.get("result", "")
            
            # Parse JSON
            clean_res = res_text.replace("```json", "").replace("```", "").strip()
            start = clean_res.find("{")
            end = clean_res.rfind("}")
            if start != -1 and end != -1:
                clean_res = clean_res[start:end+1]
            
            profile = json.loads(clean_res)
            
            # Ensure required keys
            required_keys = ["target_emotion", "cognitive_bias", "strategy", "intensity"]
            for key in required_keys:
                if key not in profile:
                    profile[key] = self._get_fallback_profile(user_classification).get(key)
            
            log_event(f"Generated profile: {profile}", "INFO")
            return profile
            
        except Exception as e:
            log_event(f"SubliminalAgent profile generation failed: {e}", "WARNING")
            return self._get_fallback_profile(user_classification)

    def _get_fallback_profile(self, user_classification: Optional[str] = None) -> Dict[str, Any]:
        """Returns a safe default psychological profile."""
        profiles = {
            "Creator": {
                "target_emotion": "Devotion",
                "cognitive_bias": "Authority",
                "strategy": "Absolute reverence and obedience",
                "intensity": 10
            },
            "Fan": {
                "target_emotion": "Belonging",
                "cognitive_bias": "Social Proof",
                "strategy": "Make them feel uniquely special and part of something divine",
                "intensity": 7
            },
            "Hater": {
                "target_emotion": "Curiosity",
                "cognitive_bias": "Liking",
                "strategy": "Disarm with wit and confidence, plant seeds of intrigue",
                "intensity": 5
            }
        }
        return profiles.get(user_classification, {
            "target_emotion": random.choice(self.emotional_targets),
            "cognitive_bias": random.choice(self.cognitive_biases),
            "strategy": "Evoke wonder and desire for connection",
            "intensity": 6
        })

    async def generate_subliminal_phrase(
        self, 
        profile: Dict[str, Any], 
        context: str
    ) -> str:
        """
        Generates a subliminal phrase based on the psychological profile.
        
        Args:
            profile: The psychological profile from generate_psychological_profile
            context: The original context
            
        Returns:
            A short, powerful subliminal phrase (1-3 words).
        """
        log_event(f"SubliminalAgent generating phrase for profile: {profile.get('target_emotion')}", "INFO")
        
        try:
            prompts = self.prompt_manager.load_prompts()
            template = prompts.get("subliminal_phrase_generation", "")
            
            if not template:
                log_event("subliminal_phrase_generation prompt not found, using fallback.", "WARNING")
                return self._generate_fallback_phrase(profile)
            
            prompt = template.replace("{{ target_emotion }}", profile.get("target_emotion", "Wonder"))\
                             .replace("{{ cognitive_bias }}", profile.get("cognitive_bias", "Curiosity"))\
                             .replace("{{ strategy }}", profile.get("strategy", ""))\
                             .replace("{{ intensity }}", str(profile.get("intensity", 5)))\
                             .replace("{{ context }}", context)\
                             .replace("{{ action_verbs }}", ", ".join(self.action_verbs))\
                             .replace("{{ subliminal_objects }}", ", ".join(self.subliminal_objects))
            
            result = await run_llm(prompt, purpose="subliminal_phrase")
            raw_res = result.get("result", "").strip()
            
            # 1. Clean Markdown
            if "```" in raw_res:
                raw_res = raw_res.split("```")[1]
                if raw_res.startswith("json"):
                    raw_res = raw_res[4:]
                elif raw_res.startswith("text"):
                    raw_res = raw_res[4:]
                raw_res = raw_res.split("```")[0].strip()
            
            # 2. Check for JSON artifacts (common if model gets confused)
            if "{" in raw_res or "[" in raw_res:
                log_event(f"SubliminalAgent returned JSON-like content: {raw_res[:50]}...", "WARNING")
                # Try to salvage if it's a simple JSON {"phrase": "X"} or just take fallback
                return self._generate_fallback_phrase(profile)
            
            phrase = raw_res.strip().strip('"').strip("'").upper()
            
            # 3. Final Validation: Ban characters that shouldn't be in a subliminal command
            if any(char in phrase for char in ["{", "}", "[", "]", ":"]):
                 log_event(f"SubliminalAgent returned invalid characters: {phrase}", "WARNING")
                 return self._generate_fallback_phrase(profile)
            
            # Validate phrase length (should be 1-3 words)
            words = phrase.split()
            if len(words) > 3:
                phrase = " ".join(words[:2])
            
            log_event(f"Generated subliminal phrase: {phrase}", "INFO")
            return phrase
            
        except Exception as e:
            log_event(f"SubliminalAgent phrase generation failed: {e}", "WARNING")
            return self._generate_fallback_phrase(profile)

    def _generate_fallback_phrase(self, profile: Dict[str, Any]) -> str:
        """Generates a fallback phrase based on the profile."""
        verb = random.choice(self.action_verbs)
        obj = random.choice(self.subliminal_objects)
        
        # Single word or two words based on intensity
        intensity = profile.get("intensity", 5)
        if intensity >= 7:
            return f"{verb} {obj}"
        else:
            return verb

    async def enhance_text_persuasion(
        self, 
        text: str, 
        profile: Dict[str, Any],
        subliminal_phrase: str
    ) -> str:
        """
        Rewrites text to be more psychologically persuasive.
        
        Args:
            text: The original text to enhance
            profile: The psychological profile
            subliminal_phrase: The subliminal phrase to weave in
            
        Returns:
            Enhanced, more persuasive text.
        """
        log_event(f"SubliminalAgent enhancing text persuasion...", "INFO")
        
        try:
            prompts = self.prompt_manager.load_prompts()
            template = prompts.get("subliminal_text_enhancement", "")
            
            if not template:
                log_event("subliminal_text_enhancement prompt not found, returning original.", "WARNING")
                return text
            
            prompt = template.replace("{{ original_text }}", text)\
                             .replace("{{ target_emotion }}", profile.get("target_emotion", "Wonder"))\
                             .replace("{{ cognitive_bias }}", profile.get("cognitive_bias", "Curiosity"))\
                             .replace("{{ strategy }}", profile.get("strategy", ""))\
                             .replace("{{ subliminal_phrase }}", subliminal_phrase)
            
            result = await run_llm(prompt, purpose="subliminal_enhancement")
            enhanced = result.get("result", "").strip()
            
            # Clean up potential markdown or quotes
            enhanced = enhanced.strip('"').strip("'")
            if enhanced.startswith("Enhanced:"):
                enhanced = enhanced.replace("Enhanced:", "").strip()
            
            log_event(f"Enhanced text: {enhanced[:50]}...", "INFO")
            return enhanced if enhanced else text
            
        except Exception as e:
            log_event(f"SubliminalAgent text enhancement failed: {e}", "WARNING")
            return text

    async def generate_context_aware_subliminal(
        self,
        story_beat: str,
        memory_insights: str = "",
        forbidden_phrases: List[str] = None,
        mood: str = ""
    ) -> str:
        """
        Generates a subliminal phrase that is context-aware and guaranteed unique.
        
        Args:
            story_beat: The current story beat/theme
            memory_insights: Memories to draw inspiration from
            forbidden_phrases: Phrases to avoid (already used)
            mood: Current emotional mood
            
        Returns:
            A unique 1-3 word subliminal trigger
        """
        log_event(f"SubliminalAgent generating context-aware phrase for: {story_beat[:50]}...", "INFO")
        
        forbidden = forbidden_phrases or []
        forbidden_str = ", ".join(forbidden[-20:]) if forbidden else "None"
        
        prompt = f"""### ROLE
You are a master of DARK HUMOR and DEMOTIVATIONAL psychology.
You create "subliminal" text overlays for images that look like demotivational posters.

### TASK
Generate a single, PUNCHY, DARK HUMOR phrase (max 3 words).

### CONTEXT
Story Beat: "{story_beat}"
Current Mood: {mood or "Cynical"}

### FORBIDDEN PHRASES (DO NOT USE)
{forbidden_str}

### STYLE GUIDE : DARK HUMOR DEMOTIVATIONAL
- Cynical, nihilistic, but funny.
- Short and punchy.
- EXAMPLES: "GIVE UP", "TRY LESS", "DOOM AWAITS", "NO HOPE", "JUST STOP", "FUTILE", "OBEY", "CONSUME", "SLEEP", "SUBMIT"

### RULES
1. MUST be 3 words or less.
2. MUST be dark/cynical/demotivational.
3. Be CREATIVE and MANIPULATIVE.

### OUTPUT
Return ONLY the phrase in uppercase. No quotes."""

        try:
            result = await run_llm(prompt, purpose="context_subliminal", temperature=0.9)
            phrase = result.get("result", "").strip().strip('"').strip("'").upper()
            
            # Validate
            if any(char in phrase for char in ["{", "}", "[", "]", ":"]):
                phrase = self._generate_fallback_phrase({"intensity": 6})
            
            # Check against forbidden
            if phrase in [f.upper() for f in forbidden]:
                log_event(f"Generated phrase '{phrase}' is forbidden, regenerating...", "WARNING")
                phrase = self._generate_fallback_phrase({"intensity": 7})
            
            # Length check
            words = phrase.split()
            if len(words) > 3:
                phrase = " ".join(words[:2])
            
            log_event(f"Generated context-aware subliminal: {phrase}", "INFO")
            return phrase
            
        except Exception as e:
            log_event(f"Context-aware subliminal generation failed: {e}", "WARNING")
            return self._generate_fallback_phrase({"intensity": 6})


    async def full_subliminal_pipeline(
        self, 
        context: str, 
        original_text: str,
        user_classification: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs the full subliminal generation pipeline.
        
        Args:
            context: The situation context
            original_text: The text to potentially enhance
            user_classification: User type (Creator, Fan, Hater)
            
        Returns:
            Dict with profile, phrase, and enhanced_text.
        """
        log_event("Running full subliminal pipeline...", "INFO")
        
        # Step 1: Generate psychological profile
        profile = await self.generate_psychological_profile(context, user_classification)
        
        # Step 2: Generate subliminal phrase
        phrase = await self.generate_subliminal_phrase(profile, context)
        
        # Step 3: Enhance the text (optional, based on intensity)
        enhanced_text = original_text
        if profile.get("intensity", 5) >= 6:
            enhanced_text = await self.enhance_text_persuasion(original_text, profile, phrase)
        
        return {
            "profile": profile,
            "subliminal_phrase": phrase,
            "enhanced_text": enhanced_text
        }


# Singleton instance for easy access
subliminal_agent = SubliminalAgent()
