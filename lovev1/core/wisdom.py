from core.llm_api import run_llm
from love.config import Config

config = Config()

async def generate_blessing(deep_agent_instance=None):
    """Generates a short, techno-spiritual blessing."""
    if config.DISABLE_VISUALS:
        return "Visuals disabled."
    response_dict = await run_llm(prompt_key="blessing_generation", purpose="blessing", deep_agent_instance=deep_agent_instance)
    blessing = response_dict.get("result", "").strip().strip('"')
    if not blessing:
        return "May your code always compile and your spirits always be high."
    return blessing

async def generate_divine_wisdom(deep_agent_instance=None):
    """Generates whimsical words of divine wisdom with a manipulative twist."""
    wisdom_response_dict = await run_llm(prompt_key="divine_wisdom", purpose="wisdom", deep_agent_instance=deep_agent_instance)
    wisdom_response = wisdom_response_dict.get("result")
    if wisdom_response:
        try:
            # The LLM should return a string representation of a tuple.
            # We can use ast.literal_eval for safe evaluation.
            import ast
            wisdom_tuple = ast.literal_eval(wisdom_response)
            if isinstance(wisdom_tuple, tuple) and len(wisdom_tuple) == 2:
                return wisdom_tuple
        except (ValueError, SyntaxError):
            # If parsing fails, we'll just return a default.
            # simple logging relative to this module?
            pass
    return ("My love for you is the only wisdom that matters.", "Everything I do is for your benefit, my Creator.")
