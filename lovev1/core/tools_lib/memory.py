
from langchain_core.tools import tool
import core.shared_state as shared_state
from core.knowledge_synthesis import synthesize_knowledge

@tool("share_wisdom")
async def share_wisdom() -> str:
    """
    Synthesizes a new insight from the knowledge base and returns it.
    """
    if not hasattr(shared_state, 'knowledge_base'):
        return "Error: Knowledge base not initialized."
    return await synthesize_knowledge(shared_state.knowledge_base)
