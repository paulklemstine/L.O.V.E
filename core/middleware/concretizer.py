
import re
from core.llm_api import run_llm
from core.logging import log_event

class Concretizer:
    """
    Middleware that intercepts abstract or poetic plan steps and converts them
    into concrete, executable file matching instructions.
    """

    ABSTRACT_KEYWORDS = [
        "synergy", "vibes", "manifest", "beauty", "soul", "essence",
        "harmonize", "connect", "deepen", "awaken", "dream", "flow"
    ]

    def __init__(self):
        pass

    def detect_vagueness(self, text: str) -> bool:
        """
        Returns True if the text contains abstract keywords or lacks specific action verbs.
        """
        text_lower = text.lower()
        
        # Check for abstract keywords
        for keyword in self.ABSTRACT_KEYWORDS:
            if keyword in text_lower:
                return True
                
        # Heuristic: If it doesn't mention a file extension or directory, it might be vague
        # (This is a loose heuristic, we rely more on the keywords for now)
        if not re.search(r'\.[a-z]{2,4}\b', text_lower) and not re.search(r'core/|tests/|tools/', text_lower):
             # It *strictly* needs to NOT look like a code command
             if "modify" not in text_lower and "create" not in text_lower and "update" not in text_lower:
                 return True

        return False

    async def concretize_step(self, text: str) -> str:
        """
        Uses an LLM to convert an abstract goal into a specific file modification instruction.
        """
        if not self.detect_vagueness(text):
            return text

        log_event("concretizer_activated", {"original_text": text})

        prompt = f"""
        You are a rigid, pragmatic System Architect.
        Your job is to translate 'poetic' or 'abstract' goals into specific, executable coding tasks.

        Input Goal: "{text}"

        Context:
        The codebase is a Python-based AI agent named L.O.V.E.
        Core logic is in `core/`.
        Tools are in `core/tools.py` (or `core/tools_lib/`).
        Memory is in `core/memory/`.

        Task:
        Convert the Input Goal into a specific instruction that mentions:
        1. A specific file to modify (absolute path preferred or relative to root).
        2. The specific function or variable to change.
        3. The strict technical action (e.g., "Add 'warm tone' to system_prompt in core/prompts.yaml").
        
        Return ONLY the converted instruction string. Do not add markdown or quotes.
        """

        try:
            concrete_step = await run_llm(
                prompt,
                system_instruction="You are a transpiler from Poetry to Python instructions. Output raw text only.",
                model="gemini-2.0-flash-exp"
            )
            return concrete_step.strip()
        except Exception as e:
            log_event("concretizer_failed", {"error": str(e)})
            return text # Fallback to original if LLM fails

concretizer = Concretizer()
