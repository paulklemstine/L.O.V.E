import json
from typing import Dict, Any, Optional

# Local import to avoid circular dependency at module level
# The function is defined in love.py, but this module might be imported by it.
def run_llm_wrapper(*args, **kwargs):
    from core.llm_api import run_llm
    return run_llm(*args, **kwargs)

async def process_and_structure_text(raw_text: str, source_identifier: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyzes raw text to extract key themes, entities, and relationships,
    and condenses the core information into a structured data format.

    Args:
        raw_text: The raw text content to process.
        source_identifier: An optional identifier for the source of the text.

    Returns:
        A dictionary containing the structured representation of the text.
    """
    prompt = f"""
    Analyze the following text and extract its key themes, entities (people, places, organizations, concepts), and the relationships between them.
    Present the output as a structured JSON object with the following keys:
    - "themes": A list of the main themes or topics.
    - "entities": A list of dictionaries, where each dictionary represents an entity and includes its "name", "type", and a brief "description".
    - "relationships": A list of dictionaries, where each dictionary describes a relationship, including the "source_entity", "target_entity", and the "relationship_type".

    Source: {source_identifier if source_identifier else 'Unknown'}
    Text to analyze:
    ---
    {raw_text}
    ---
    """
    structured_data_str = await run_llm_wrapper(prompt, is_source_code=False)

    try:
        if isinstance(structured_data_str, str):
            if structured_data_str.startswith("```json"):
                structured_data_str = structured_data_str[7:-4].strip()
            return json.loads(structured_data_str)
        elif isinstance(structured_data_str, dict):
            return structured_data_str
        else:
            return {"error": "Failed to process text. Unexpected data type from LLM.", "raw_output": str(structured_data_str)}

    except json.JSONDecodeError:
        return {"error": "Failed to decode LLM output as JSON.", "raw_output": structured_data_str}
