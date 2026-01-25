# Module Documentation: `core/text_processing.py`

This document provides documentation for the `text_processing` module, which is responsible for analyzing and structuring raw text to support the automated construction of a knowledge base.

## Task History

*   **Date:** 2025-11-06
*   **Request:** Integrate LangExtract for extracting information from text.
*   **Pull Request:** (To be filled after submission)
*   **Commit Hash:** (To be filled after submission)

*   **Date:** 2025-11-03
*   **Request:** Implement a function to process raw text into a structured format and describe a system to build a knowledge base with it.
*   **Pull Request:** (Filled)
*   **Commit Hash:** (Filled)

---

## Function: `process_and_structure_text`

This is an asynchronous function that serves as the primary interface for the module. It takes a string of raw text and uses the **LangExtract** library to extract meaningful information, returning it in a rich, structured format.

### Purpose

The goal of this function is to convert unstructured, human-readable text into a machine-readable format that can be easily stored, indexed, and queried. This process is the foundational step for building and enriching the application's central knowledge base, which is managed by the `GraphDataManager`. By leveraging LangExtract, the function provides more reliable and detailed extractions with precise source grounding.

For a complete overview of how this function fits into the larger system, please see the [Automated Knowledge Base Construction architecture document](../../docs/knowledge_base_system.md).

### Parameters

*   `raw_text` (str): The raw text content to be analyzed.
*   `source_identifier` (Optional[str], default=None): An optional string to identify the origin of the text (e.g., 'love.log', 'user_prompt', 'web_search_result').

### Returns

*   `(Dict[str, Any])`: A dictionary containing the structured data. On success, the dictionary will have the following keys:
    *   `"summary"`: A concise summary of the text.
    *   `"takeaways"`: A list of key takeaways or main points.
    *   `"entities"`: A list of dictionaries, with each representing a named entity. Each entity dictionary contains its `name`, `type`, `description`, and `salience`.
    *   `"topics"`: A list of topics discussed in the text.
    *   `"sentiment"`: The overall sentiment of the text (e.g., 'positive', 'neutral', 'negative').

    On failure, the function may return a dictionary with an `"error"` key.

### Example Usage

```python
import asyncio
from core.text_processing import process_and_structure_text

async def main():
    log_entry = "The Orchestrator in core/agents/orchestrator.py was updated to use the GeminiReActEngine for improved planning. This change is expected to significantly improve the agent's performance and autonomy."

    structured_data = await process_and_structure_text(
        raw_text=log_entry,
        source_identifier="codebase_commit_log"
    )

    if "error" in structured_data:
        print(f"An error occurred: {structured_data['error']}")
    else:
        print("Successfully processed text:")
        print(f"Summary: {structured_data.get('summary')}")
        print(f"Sentiment: {structured_data.get('sentiment')}")
        print("Key Takeaways:")
        for takeaway in structured_data.get('takeaways', []):
            print(f"- {takeaway}")
        print("Entities:")
        for entity in structured_data.get('entities', []):
            print(f"- {entity.get('name')} ({entity.get('type')}): {entity.get('description')}")
        print("Topics:")
        for topic in structured_data.get('topics', []):
            print(f"- {topic}")


if __name__ == "__main__":
    asyncio.run(main())
```
