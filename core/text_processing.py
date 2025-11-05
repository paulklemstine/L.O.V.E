import json
from typing import Dict, Any, Optional, Union


async def process_content_with_directives(
    content: Union[str, Dict[str, Any]],
    directives: Union[str, Dict[str, Any]],
    source_identifier: Optional[str] = None
) -> Dict[str, Any]:
    """
    Asynchronously dispatches content and processing directives to an external
    engine (e.g., LLM) and returns the structured output.

    Args:
        content: The text or structured data to be processed. If a dictionary,
                 it will be converted to a JSON string.
        directives: The processing instructions, such as a natural language
                    prompt, a JSON schema, or a transformation template.
        source_identifier: An optional identifier for the source of the content.

    Returns:
        A dictionary containing the structured output from the processing engine.
        Includes robust error handling for API failures and JSON parsing issues.
    """
    if isinstance(content, dict):
        content_str = json.dumps(content, indent=2)
    else:
        content_str = str(content)

    prompt = f"""
    You are a configurable data processing engine. Follow the directives below to process the provided content and return the output in the specified format.

    **Directives:**
    ---
    {directives}
    ---

    **Content to Process:**
    ---
    Source: {source_identifier if source_identifier else 'Unknown'}
    Content:
    {content_str}
    ---
    """

    processed_output_dict = await run_llm_wrapper(prompt, is_source_code=False)
    processed_output = processed_output_dict.get("result", "")

    try:
        if isinstance(processed_output, str):
            # Clean up potential markdown code blocks
            if processed_output.strip().startswith("```json"):
                processed_output = processed_output.strip()[7:-4].strip()
            elif processed_output.strip().startswith("```"):
                 processed_output = processed_output.strip()[3:-3].strip()
            return json.loads(processed_output)
        elif isinstance(processed_output, dict):
            # If the LLM wrapper already parsed it (e.g., from gemini-cli)
            return processed_output
        else:
            return {
                "error": "Failed to process content. Unexpected data type from processing engine.",
                "raw_output": str(processed_output)
            }
    except json.JSONDecodeError:
        return {
            "error": "Failed to decode the processing engine's output as JSON.",
            "raw_output": processed_output
        }
    except Exception as e:
        return {
            "error": f"An unexpected error occurred during processing: {str(e)}",
            "raw_output": str(processed_output)
        }


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
    structured_data_dict = await run_llm_wrapper(prompt, is_source_code=False)
    structured_data_str = structured_data_dict.get("result", "")

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
