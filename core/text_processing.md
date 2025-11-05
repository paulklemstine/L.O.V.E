# Module Documentation: `core/text_processing.py`

This document provides documentation for the `text_processing` module, which is responsible for analyzing and structuring raw text to support the automated construction of a knowledge base.

## Task History

*   **Date:** 2025-11-03
*   **Request:** Implement a function to process raw text into a structured format and describe a system to build a knowledge base with it.
*   **Pull Request:** (To be filled after submission)
*   **Commit Hash:** (To be filled after submission)

---

## Function: `process_content_with_directives`

This is a powerful and flexible asynchronous function designed to process arbitrary blocks of text or structured data according to a set of explicit directives. It acts as a generic interface to an external processing engine (e.g., a Large Language Model) and is built for extensibility.

### Purpose

The goal of this function is to provide a standardized way to offload complex interpretation, transformation, or analysis tasks. Instead of having hardcoded prompts or parsing logic, this function allows the caller to define the processing task at runtime by supplying specific `directives`. This makes it ideal for a wide range of applications, from simple data extraction to complex, multi-step reasoning.

### Parameters

*   `content` (Union[str, Dict[str, Any]]): The input data to be processed. This can be a simple string (like a log entry or document) or a Python dictionary, which will be automatically serialized to a JSON string.
*   `directives` (Union[str, Dict[str, Any]]): The instructions for the processing engine. This can be a natural language prompt (e.g., "Summarize the key takeaways from the following text") or a structured format like a JSON object defining a schema for the output.
*   `source_identifier` (Optional[str], default=None): An optional string to identify the origin of the content.

### Returns

*   `(Dict[str, Any])`: A dictionary containing the structured data returned by the processing engine.
    *   On a successful run where the engine provides valid JSON, this will be the parsed Python dictionary.
    *   On failure (e.g., if the engine's output is not valid JSON or an API error occurs), the dictionary will contain an `"error"` key with a descriptive message and a `"raw_output"` key with the original, unprocessed response from the engine.

### Application Examples

The following examples demonstrate how to apply this function to achieve specific outcomes.

#### 1. Data Ingestion & Directive Formulation

This example shows how to ingest different types of unstructured text and formulate directives to extract specific information in a desired JSON format.

```python
import asyncio
from core.text_processing import process_content_with_directives

async def main():
    # --- Example 1: Synthesizing a summary from a log entry ---
    log_entry = "2025-11-05 01:58:30,151 - INFO - Git reset successful: a1b2c3d - User 'jules' initiated evolution."
    summary_directives = "Extract the timestamp, log level, and a brief summary of the event. Return as a JSON object."

    log_summary = await process_content_with_directives(
        content=log_entry,
        directives=summary_directives,
        source_identifier="love.log"
    )
    print("--- Log Summary ---")
    print(log_summary)
    # Expected output: {'timestamp': '2025-11-05 01:58:30,151', 'level': 'INFO', 'summary': 'User jules initiated a successful git reset for evolution.'}


    # --- Example 2: Generating a structured assessment based on a schema ---
    web_summary = "The company's latest quarterly report shows a 15% increase in revenue, driven by the new AI division, but a 5% decrease in user engagement on the legacy platform."
    assessment_directives = {
        "task": "Analyze the financial summary and provide a structured assessment.",
        "output_schema": {
            "positive_key_takeaway": "string",
            "negative_key_takeaway": "string",
            "priority": "one of ['high', 'medium', 'low']",
            "recommended_action": "string"
        }
    }

    strategic_assessment = await process_content_with_directives(
        content=web_summary,
        directives=assessment_directives,
        source_identifier="web_scrape_quarterly_report"
    )
    print("\n--- Strategic Assessment ---")
    print(strategic_assessment)
    # Expected output: {'positive_key_takeaway': 'Revenue increased by 15% due to the new AI division.', 'negative_key_takeaway': 'User engagement dropped by 5% on the legacy platform.', 'priority': 'high', 'recommended_action': 'Investigate the cause of the user engagement drop on the legacy platform.'}

if __name__ == "__main__":
    asyncio.run(main())
```

#### 2. Asynchronous Workflow Integration with `LocalJobManager`

The `process_content_with_directives` function is `async`, but the `LocalJobManager` in `love.py` runs its jobs in synchronous background threads. To integrate them, you need a wrapper function that can run the `async` function within the synchronous thread's event loop.

*Note: This example assumes it is run in a context where `love.py`'s `LocalJobManager` is initialized and running.*

```python
import asyncio
import functools
from love import LocalJobManager # Hypothetical import for demonstration

# --- Wrapper to run the async function in a sync context ---
def sync_process_wrapper(content, directives, source_identifier=None, progress_callback=None):
    """
    A synchronous wrapper to call the async process_content_with_directives function.
    The progress_callback is unused here but is part of the job manager's signature.
    """
    # This is a simplified way to run an async function from a sync thread.
    # In a real application with a running asyncio loop, you might use
    # `asyncio.run_coroutine_threadsafe`.
    try:
        return asyncio.run(process_content_with_directives(content, directives, source_identifier))
    except Exception as e:
        return {"error": f"Failed to run async job: {e}"}


def schedule_processing_job(job_manager: LocalJobManager):
    """
    Demonstrates scheduling a processing task as a background job.
    """
    document_to_analyze = "..." # Assume this is a very long document
    analysis_directives = "Identify all named entities (people, places, organizations) and list them."

    job_id = job_manager.add_job(
        description="Analyze long document for entities",
        target_func=functools.partial(
            sync_process_wrapper,
            content=document_to_analyze,
            directives=analysis_directives,
            source_identifier="document_batch_1.txt"
        )
    )

    print(f"Successfully scheduled background analysis job with ID: {job_id}")
    # The job manager will now run this in the background, and the results
    # can be retrieved later by checking the job's status.
```

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
