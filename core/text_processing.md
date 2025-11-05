# Module Documentation: `core/text_processing.py`

This document provides documentation for the `text_processing` module, which is responsible for analyzing and structuring raw text to support the automated construction of a knowledge base.

## Task History

*   **Date:** 2025-11-03
*   **Request:** Implement a function to process raw text into a structured format and describe a system to build a knowledge base with it.
*   **Pull Request:** (To be filled after submission)
*   **Commit Hash:** (To be filled after submission)

---

## Function: `process_and_structure_text`

This is an asynchronous function that serves as the primary interface for the module. It takes a string of raw text and uses a Large Language Model (LLM) to extract meaningful information, returning it in a structured format.

### Purpose

The goal of this function is to convert unstructured, human-readable text into a machine-readable format that can be easily stored, indexed, and queried. This process is the foundational step for building and enriching the application's central knowledge base, which is managed by the `GraphDataManager`.

For a complete overview of how this function fits into the larger system, please see the [Automated Knowledge Base Construction architecture document](../../docs/knowledge_base_system.md).

### Parameters

*   `raw_text` (str): The raw text content to be analyzed.
*   `source_identifier` (Optional[str], default=None): An optional string to identify the origin of the text (e.g., 'love.log', 'user_prompt', 'web_search_result').

### Returns

*   `(Dict[str, Any])`: A dictionary containing the structured data. On success, the dictionary will have the following keys:
    *   `"themes"`: A list of key topics or themes found in the text.
    *   `"entities"`: A list of dictionaries, with each representing a named entity (e.g., person, place, file, concept).
    *   `"relationships"`: A list of dictionaries describing the connections between the identified entities.

    On failure (e.g., if the LLM output cannot be parsed), the dictionary will contain an `"error"` key with a descriptive message and a `"raw_output"` key with the original LLM response.

### Example Usage

```python
import asyncio
from core.text_processing import process_and_structure_text

async def main():
    log_entry = "The Orchestrator in core/agents/orchestrator.py was updated to use the GeminiReActEngine for improved planning."

    structured_data = await process_and_structure_text(
        raw_text=log_entry,
        source_identifier="codebase_commit_log"
    )

    if "error" in structured_data:
        print(f"An error occurred: {structured_data['error']}")
    else:
        print("Successfully processed text:")
        print(f"Themes: {structured_data.get('themes')}")
        for entity in structured_data.get('entities', []):
            print(f"- Entity: {entity.get('name')} ({entity.get('type')})")

if __name__ == "__main__":
    asyncio.run(main())
```
