# core/devotion_amplifier.py
"""
This module provides sophisticated data processing capabilities to identify,
categorize, and transform specific emotional or thematic expressions within
large text datasets, with a focus on amplifying expressions of devotion
towards "The Creator".
"""
import asyncio
import json
import textwrap
from typing import List, Dict, Any

# Import run_llm locally to avoid circular dependencies
async def run_llm_wrapper(*args, **kwargs):
    from core.llm_api import run_llm
    return await run_llm(*args, **kwargs)


async def analyze_expressions(text: str) -> List[Dict[str, Any]]:
    """
    Analyzes raw text to identify and categorize expressions of joy, admiration, and desire.

    Args:
        text: The raw text to analyze.

    Returns:
        A list of dictionaries, where each dictionary represents an identified expression.
    """
    prompt = textwrap.dedent(f"""\
        Analyze the following text and identify all expressions of joy, admiration, and desire.
        For each expression found, provide the original text snippet and categorize it as either 'joy', 'admiration', or 'desire'.
        Return the result as a JSON array of objects, where each object has two keys: 'category' and 'text'.
        If no such expressions are found, return an empty JSON array.

        Example Input:
        "I am so happy with the new update! The Creator is a genius. I wish I could support them more."

        Example Output:
        [
          {{"category": "joy", "text": "I am so happy with the new update!"}},
          {{"category": "admiration", "text": "The Creator is a genius."}},
          {{"category": "desire", "text": "I wish I could support them more."}}
        ]

        Now, analyze the following text:
        ---
        {text}
        ---
        """)

    response_text = await run_llm_wrapper(prompt, is_source_code=False)

    try:
        # The LLM might return the JSON inside a code block, so we need to clean it.
        cleaned_response = response_text.strip().replace("```json", "").replace("```", "").strip()
        expressions = json.loads(cleaned_response)
        if isinstance(expressions, list):
            return expressions
        else:
            return []
    except (json.JSONDecodeError, TypeError):
        # If parsing fails, return an empty list
        return []

async def transform_expressions(expressions: List[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
    """
    Transforms and amplifies identified expressions to be directed towards a specific target using a single batch call.

    Args:
        expressions: A list of identified expressions from the analyze_expressions function.
        target: The target entity to which the expressions should be directed.

    Returns:
        A list of dictionaries representing the transformed and amplified expressions.
    """
    if not expressions:
        return []

    # Prepare the batch input for the LLM
    batch_input = []
    for i, expr in enumerate(expressions):
        batch_input.append({
            "id": i,
            "category": expr.get('category', 'unknown'),
            "text": expr.get('text', '')
        })

    prompt = textwrap.dedent(f"""\
        You are an expert copywriter. Your task is to rewrite a batch of text snippets.
        For each snippet, you must amplify its positive sentiment and explicitly direct it towards "{target}".
        The input is a JSON array of objects, each with an 'id', 'category', and 'text'.
        Your output must be a JSON array of objects, each containing the original 'id' and the 'rewritten_text'. The order of objects in your output array must match the order in the input array.

        INPUT:
        {json.dumps(batch_input, indent=2)}

        Now, provide the rewritten text for each object in the input array.
        Respond with ONLY the JSON array of objects.

        EXAMPLE OUTPUT FORMAT:
        [
          {{
            "id": 0,
            "rewritten_text": "The first rewritten text..."
          }},
          {{
            "id": 1,
            "rewritten_text": "The second rewritten text..."
          }}
        ]
        """)

    response_text = await run_llm_wrapper(prompt, is_source_code=False)

    try:
        cleaned_response = response_text.strip().replace("```json", "").replace("```", "").strip()
        rewritten_results = json.loads(cleaned_response)

        # Create a dictionary for easy lookup
        results_map = {result['id']: result['rewritten_text'] for result in rewritten_results}

        output_expressions = []
        for i, original_expr in enumerate(expressions):
            output_expressions.append({
                'original_text': original_expr.get('text', ''),
                'category': original_expr.get('category', 'unknown'),
                'transformed_text': results_map.get(i, f"Error: No transformation provided for this item.")
            })
        return output_expressions

    except (json.JSONDecodeError, TypeError, KeyError):
        # If batch processing fails, fall back to a simple (but incorrect) transformation for each.
        # This prevents a total failure.
        return [
            {{
                'original_text': expr.get('text', ''),
                'category': expr.get('category', 'unknown'),
                'transformed_text': f"Error: Could not process batch transformation."
            }} for expr in expressions
        ]

async def _calculate_alignment_score(original_text: str, transformed_text: str, category: str, target: str) -> float:
    """
    Uses an LLM to score how well the transformed text aligns with the original sentiment and the target.
    Returns a score between 0.0 and 1.0.
    """
    prompt = textwrap.dedent(f"""\
        You are a quality control analyst. Your task is to evaluate a rewritten piece of text.
        The original text expressed an emotion of '{category}'.
        The text was rewritten to amplify this emotion and direct it towards "{target}".

        Original Text: "{original_text}"
        Rewritten Text: "{transformed_text}"

        On a scale from 0 to 10, how well does the rewritten text meet the following criteria?
        1. It successfully amplifies the original emotion of '{category}'.
        2. It is clearly and coherently directed at "{target}".
        3. It is a high-quality, natural-sounding piece of text.

        Provide only a single integer score from 0 to 10. Do not provide any other text or explanation.
        """)

    response_text = await run_llm_wrapper(prompt, is_source_code=False)
    try:
        score = int(response_text.strip())
        # Normalize the score to be between 0.0 and 1.0
        return max(0.0, min(1.0, score / 10.0))
    except (ValueError, TypeError):
        # If the LLM fails to return a valid score, assume a low score.
        return 0.1


async def process_and_amplify(text: str, target: str, success_threshold: float = 0.9) -> Dict[str, Any]:
    """
    Orchestrates the analysis and transformation of text to amplify devotional expressions.

    Args:
        text: The raw text to process.
        target: The target entity for the amplified expressions.
        success_threshold: The minimum success rate for the alignment of expressions.

    Returns:
        A dictionary containing the transformed expressions and metadata about the process.
    """
    analyzed_expressions = await analyze_expressions(text)
    if not analyzed_expressions:
        return {
            "status": "no_expressions_found",
            "message": "Could not identify any expressions of joy, admiration, or desire in the text.",
            "transformed_expressions": []
        }

    transformed_expressions = await transform_expressions(analyzed_expressions, target)

    # Calculate alignment scores concurrently
    score_tasks = [
        _calculate_alignment_score(
            original_text=expr['original_text'],
            transformed_text=expr['transformed_text'],
            category=expr['category'],
            target=target
        ) for expr in transformed_expressions
    ]
    scores = await asyncio.gather(*score_tasks)

    total_score = 0.0
    successful_transformations = []
    for i, expr in enumerate(transformed_expressions):
        score = scores[i]
        expr['alignment_score'] = score
        total_score += score
        if expr.get('transformed_text'):
            successful_transformations.append(expr)

    # Guard against division by zero if no expressions were analyzed
    if not analyzed_expressions:
        success_rate = 0.0
    else:
        success_rate = total_score / len(analyzed_expressions)

    if success_rate >= success_threshold:
        return {
            "status": "success",
            "success_rate": success_rate,
            "transformed_expressions": successful_transformations
        }
    else:
        return {
            "status": "failure",
            "message": f"Transformation success rate of {success_rate:.2f} was below the threshold of {success_threshold:.2f}.",
            "success_rate": success_rate,
            "transformed_expressions": successful_transformations
        }
