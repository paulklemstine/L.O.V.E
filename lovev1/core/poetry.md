# L.O.V.E. Module Documentation: Poetry Generation

**Date:** Sat Jan 10 02:29:39 UTC 2026

## Task History

### Original Request

The Creator initiated a high-level objective for "Automated Codebase Health Monitoring and Feature Enhancement." When prompted to select a concrete starting point, The Creator chose to prioritize "dynamic content" generation. This module was created as the first step in fulfilling that directive.

### Pull Request & Commit Details

-   **Pull Request:** (Will be filled in upon submission)
-   **Commit Hash:** (Will be filled in upon submission)

## Implementation Details

### `core/poetry.py`

This file introduces a new, focused module for creative text generation, specifically poetry. This keeps creative tooling separate from the main application logic in `love.py` for better organization.

#### `generate_poem(topic: str, deep_agent_instance=None) -> str`

-   **Purpose:** This `async` function serves as the core of the poetry generation feature. It takes a string `topic` as input and uses the central `run_llm` function to generate a poem.
-   **Methodology:**
    -   It calls `run_llm` with the `prompt_key="poetry_generation"` and a `purpose="poetry"`. This allows the underlying LLM API to select the most appropriate model and prompt for the creative task.
    -   It follows the established pattern from `generate_divine_wisdom` in `love.py` by including robust error handling within a `try...except` block.
    -   If the LLM call fails or returns an empty string, it provides a graceful fallback message to the user, ensuring the UI does not display an empty or broken output.
-   **Integration:** This function is designed to be exposed as a command-line tool within the L.O.V.E. terminal, allowing The Creator to directly invoke it.
