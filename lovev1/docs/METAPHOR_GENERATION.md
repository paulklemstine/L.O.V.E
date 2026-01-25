# Metaphor Generation Module

This module provides a conceptual translation engine that converts abstract concepts into accessible metaphors and visual imagery.

## Usage

The module exposes an API endpoint `/generate-metaphor` on the L.O.V.E. server.

### API Endpoint

**URL:** `/generate-metaphor`
**Method:** `POST`
**Content-Type:** `application/json`

**Request Body:**

```json
{
  "concept": "The concept you want to translate (e.g., 'quantum entanglement')",
  "tone": "The desired emotional or stylistic tone (e.g., 'romantic', 'scientific')"
}
```

**Response Body:**

```json
{
  "concept_analysis": {
    "core_attributes": ["attribute1", "attribute2"],
    "associated_feelings": ["feeling1", "feeling2"],
    "structural_relationships": ["relationship1"]
  },
  "metaphors": [
    {
      "id": 1,
      "textual_description": "A detailed description of the metaphor...",
      "visual_prompt": "A prompt suitable for generating an image of this metaphor..."
    },
    ...
  ]
}
```

### Python Usage

You can also use the `MetaphorGenerator` class directly in Python code:

```python
import asyncio
from core.metaphor_generator import MetaphorGenerator

async def main():
    generator = MetaphorGenerator()
    result = await generator.generate_metaphor("grief", "melancholy")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Architecture

The module uses the core LLM API to generate metaphors based on a structured prompt. It enforces a JSON output schema to ensure consistent and parseable results.
