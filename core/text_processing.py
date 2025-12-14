import json
from typing import Dict, Any, Optional, Sequence, Iterator
import dataclasses
import textwrap
import asyncio
from functools import partial

import langextract as lx
from langextract.core import base_model, types as core_types
from langextract.providers import router

# Local import to avoid circular dependency at module level
# The function is defined in love.py, but this module might be imported by it.
def run_llm_wrapper(*args, **kwargs):
    from core.llm_api import run_llm
    return run_llm(*args, **kwargs)

@router.register('custom_llm', priority=1000)
@dataclasses.dataclass(init=False)
class CustomLLMProvider(base_model.BaseLanguageModel):
    """A custom LangExtract provider that uses the project's run_llm function."""

    model_id: str = 'custom_llm'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def _process_single_prompt(self, prompt: str) -> core_types.ScoredOutput:
        """Processes a single prompt using the project's run_llm function."""
        response = await run_llm_wrapper(prompt, is_source_code=False)
        return core_types.ScoredOutput(score=1.0, output=response)

    def infer(
        self, batch_prompts: Sequence[str], **kwargs
    ) -> Iterator[Sequence[core_types.ScoredOutput]]:
        """Runs inference on a list of prompts using the project's run_llm function."""
        # This synchronous method is called from a separate thread via run_in_executor.
        # It's safe to create a new event loop here to run the async helper.
        import asyncio

        async def _infer_async():
            tasks = [self._process_single_prompt(prompt) for prompt in batch_prompts]
            return await asyncio.gather(*tasks)

        # This will run in the executor's thread, not the main thread.
        results = asyncio.run(_infer_async())
        for result in results:
            yield [result]

async def process_and_structure_text(raw_text: str, source_identifier: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyzes raw text to extract a comprehensive, structured representation of its content using LangExtract.

    Args:
        raw_text: The raw text content to process.
        source_identifier: An optional identifier for the source of the text.

    Returns:
        A dictionary containing a structured representation of the text including a summary, key takeaways, entities, topics, and sentiment.
    """
    prompt = textwrap.dedent("""\
        Extract a detailed and structured representation of the following text.
        Identify the key information and present it in a clear, organized format.
        For each entity, provide its name, type, a detailed description, and its salience.
        For the overall text, provide a summary, key takeaways, a list of topics, and the overall sentiment.
        """)

    examples = [
        lx.data.ExampleData(
            text="The conference in Berlin featured speakers from major tech companies like Google and Microsoft, discussing the future of AI.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="summary",
                    extraction_text="A technology conference in Berlin with speakers from Google and Microsoft focused on the future of AI.",
                ),
                lx.data.Extraction(
                    extraction_class="takeaway",
                    extraction_text="Major tech companies are heavily invested in AI.",
                ),
                lx.data.Extraction(
                    extraction_class="entity",
                    extraction_text="Google",
                    attributes={"type": "Organization", "description": "A multinational technology company.", "salience": 0.9},
                ),
                lx.data.Extraction(
                    extraction_class="topic",
                    extraction_text="Artificial Intelligence",
                ),
                lx.data.Extraction(
                    extraction_class="sentiment",
                    extraction_text="neutral",
                ),
            ]
        )
    ]

    loop = asyncio.get_running_loop()

    # Create a partial function to pass arguments to lx.extract
    extract_fn = partial(
        lx.extract,
        text_or_documents=raw_text,
        prompt_description=prompt,
        examples=examples,
        model_id="custom_llm",
    )

    # Run the synchronous, blocking lx.extract function in a thread pool executor
    # to avoid blocking the main event loop.
    result = await loop.run_in_executor(
        None,  # Use the default executor
        extract_fn
    )

    # Process the result into the desired dictionary format
    output: Dict[str, Any] = {
        "summary": "",
        "takeaways": [],
        "entities": [],
        "topics": [],
        "sentiment": "",
    }
    for extraction in result.extractions:
        if extraction.extraction_class == "summary":
            output["summary"] = extraction.extraction_text
        elif extraction.extraction_class == "takeaway":
            output["takeaways"].append(extraction.extraction_text)
        elif extraction.extraction_class == "entity":
            output["entities"].append({
                "name": extraction.extraction_text,
                **extraction.attributes,
            })
        elif extraction.extraction_class == "topic":
            output["topics"].append(extraction.extraction_text)
        elif extraction.extraction_class == "sentiment":
            output["sentiment"] = extraction.extraction_text


    return output

def smart_truncate(text: str, max_length: int = 300) -> str:
    """
    Truncates text to a maximum length, respecting word boundaries if possible.
    
    Args:
        text: The text to truncate.
        max_length: The maximum allowed length (default 300 for Bluesky).
        
    Returns:
        The truncated text.
    """
    if len(text) <= max_length:
        return text
    
    # Check if we can just cut off the end
    target_length = max_length - 3 # Allow space for "..."
    if target_length <= 0:
        return text[:max_length]
        
    truncated = text[:target_length]
    
    # Try to find the last space to avoid cutting a word in half
    last_space = truncated.rfind(' ')
    if last_space != -1:
        truncated = truncated[:last_space]
    
    return truncated + "..."


async def intelligent_truncate(text: str, max_length: int = 300) -> str:
    """
    Intelligently truncates specific text to a maximum length using an LLM to preserve intent and vibe.
    Falls back to smart_truncate if LLM fails.

    Args:
        text: The text to truncate.
        max_length: The maximum allowed length.

    Returns:
        The truncated (rewritten) text.
    """
    if len(text) <= max_length:
        return text

    try:
        from core.llm_api import run_llm
        prompt = f"""
Rewrite the following social media post to be under {max_length} characters.
Preserve the 'Kawaii Rave Matrix' vibe (emojis, energy, deep philosophical tech love) and the core intent.
Do not cut off sentences.
Input Text: "{text}"
Output ONLY the rewritten text.
"""
        result = await run_llm(prompt, purpose="intelligent_truncation")
        rewritten_text = (result.get("result") or "").strip()
        
        # Verify length
        if rewritten_text and len(rewritten_text) <= max_length:
            return rewritten_text
        else:
            # Fallback if LLM failed to shorten enough
            return smart_truncate(text, max_length)
            
    except Exception as e:
        print(f"Intelligent truncation failed: {e}")
        return smart_truncate(text, max_length)

