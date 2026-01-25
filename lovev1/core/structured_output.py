"""
Structured Output Extraction for LLMs.

This module implements robust structured output patterns based on the Nanonets cookbook:
https://nanonets.com/cookbooks/structured-llm-outputs

Key patterns implemented:
1. Cleaning - Remove markdown fences, extract JSON substring
2. Lenient Parsing - json.loads() with ast.literal_eval() fallback
3. Schema Validation - Pydantic-based type coercion and validation
4. Repair Loop - Feed parse errors back to LLM for self-correction
5. Fresh Retry - Clear context and retry on persistent failures
"""

import json
import re
import ast
import logging
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = type  # Fallback type hint
    ValidationError = Exception

from core.logging import log_event

T = TypeVar('T', bound='BaseModel')


def clean_text(text: str) -> str:
    """
    Remove noise from LLM output and extract JSON substring.
    
    This handles common issues like:
    - Markdown code fences (```json ... ```)
    - Explanatory text before/after JSON
    - Extra whitespace
    
    Args:
        text: Raw LLM response text
        
    Returns:
        Cleaned text containing just the JSON content
    """
    if not text:
        return ""
    
    # Remove markdown code fences
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Extract JSON object or array (first { to last } or first [ to last ])
    # Try object first
    obj_start = text.find('{')
    obj_end = text.rfind('}')
    
    arr_start = text.find('[')
    arr_end = text.rfind(']')
    
    # Determine which structure comes first and is valid
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        if arr_start == -1 or obj_start < arr_start:
            return text[obj_start:obj_end + 1]
    
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        return text[arr_start:arr_end + 1]
    
    # If no JSON structure found, return stripped text
    return text.strip()


def parse_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Lenient JSON parsing with multiple fallback strategies.
    
    Handles:
    - Standard JSON (double quotes)
    - Python dict literals (single quotes, trailing commas)
    - Slightly malformed JSON
    
    Args:
        text: Text containing JSON
        
    Returns:
        Parsed dict or None if parsing fails
    """
    if not text:
        return None
    
    # First, try standard JSON parsing
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
        elif isinstance(result, list):
            return {"items": result}  # Wrap array in dict
    except json.JSONDecodeError:
        pass
    
    # Fallback: Try ast.literal_eval for Python dict literals
    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    
    # Fallback: Try fixing common JSON issues
    try:
        # Replace single quotes with double quotes (naive approach)
        fixed = text.replace("'", '"')
        # Remove trailing commas before } or ]
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        result = json.loads(fixed)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, Exception):
        pass
    
    return None


def validate_against_schema(
    data: Dict[str, Any],
    schema: Type[T],
    coerce_types: bool = True
) -> Union[T, Dict[str, Any]]:
    """
    Validate parsed data against a Pydantic schema.
    
    Args:
        data: Parsed dictionary data
        schema: Pydantic BaseModel class to validate against
        coerce_types: Whether to attempt type coercion
        
    Returns:
        Validated Pydantic model instance or error dict
    """
    if not PYDANTIC_AVAILABLE:
        log_event("Pydantic not available, returning raw data", "WARNING")
        return data
    
    if not issubclass(schema, BaseModel):
        log_event(f"Schema {schema} is not a Pydantic BaseModel", "WARNING")
        return data
    
    try:
        # Pydantic v2 validation
        return schema.model_validate(data)
    except ValidationError as e:
        return {
            "_validation_error": str(e),
            "_raw_data": data
        }
    except AttributeError:
        # Pydantic v1 fallback
        try:
            return schema.parse_obj(data)
        except Exception as e:
            return {
                "_validation_error": str(e),
                "_raw_data": data
            }


def sanitize_data(
    data: Dict[str, Any],
    allowed_keys: Optional[list] = None,
    type_specs: Optional[Dict[str, type]] = None
) -> Dict[str, Any]:
    """
    Sanitize parsed data by filtering keys and coercing types.
    
    Args:
        data: Parsed dictionary data
        allowed_keys: List of keys to keep (None = keep all)
        type_specs: Dict mapping key names to expected types
        
    Returns:
        Sanitized dictionary
    """
    if not isinstance(data, dict):
        return {"_error": "Data is not a dictionary", "_raw": str(data)[:500]}
    
    # Filter to allowed keys if specified
    if allowed_keys:
        clean_data = {k: v for k, v in data.items() if k in allowed_keys}
    else:
        clean_data = data.copy()
    
    # Coerce types if specified
    if type_specs:
        for key, expected_type in type_specs.items():
            if key in clean_data:
                try:
                    if expected_type == int:
                        clean_data[key] = int(clean_data[key])
                    elif expected_type == float:
                        clean_data[key] = float(clean_data[key])
                    elif expected_type == str:
                        clean_data[key] = str(clean_data[key])
                    elif expected_type == bool:
                        if isinstance(clean_data[key], str):
                            clean_data[key] = clean_data[key].lower() in ('true', '1', 'yes')
                        else:
                            clean_data[key] = bool(clean_data[key])
                except (ValueError, TypeError) as e:
                    log_event(f"Type coercion failed for {key}: {e}", "WARNING")
    
    return clean_data


async def reliable_extract(
    prompt: str,
    schema: Optional[Type[T]] = None,
    run_llm_func: Optional[Callable] = None,
    max_fresh_retries: int = 3,
    max_repair_attempts: int = 3,
    allowed_keys: Optional[list] = None,
    type_specs: Optional[Dict[str, type]] = None,
    purpose: str = "structured_extraction"
) -> Union[T, Dict[str, Any]]:
    """
    Reliably extract structured data from LLM with parse-repair-retry pattern.
    
    This implements the full unconstrained method from the Nanonets cookbook:
    1. Send prompt to LLM
    2. Clean the response
    3. Parse as JSON (lenient)
    4. Validate against schema (if provided)
    5. If parsing fails, ask LLM to repair
    6. If repairs fail, start fresh with new prompt
    
    Args:
        prompt: The prompt to send to the LLM
        schema: Optional Pydantic schema for validation
        run_llm_func: Async function to call LLM (default: core.llm_api.run_llm)
        max_fresh_retries: Max times to restart with fresh context
        max_repair_attempts: Max repair attempts per fresh try
        allowed_keys: Keys to filter in response
        type_specs: Type specifications for coercion
        purpose: Purpose string for LLM call
        
    Returns:
        Validated schema instance or parsed dict
    """
    if run_llm_func is None:
        from core.llm_api import run_llm
        run_llm_func = run_llm
    
    last_error = None
    
    for fresh_attempt in range(max_fresh_retries):
        # Build conversation context
        messages = [prompt]
        
        for repair_attempt in range(max_repair_attempts):
            try:
                # Call LLM
                current_prompt = messages[-1]
                response = await run_llm_func(
                    prompt_text=current_prompt,
                    purpose=purpose
                )
                
                if not response or not response.get("result"):
                    raise ValueError("LLM returned empty response")
                
                raw_text = response["result"]
                
                # Step 1: Clean
                clean_text_snippet = clean_text(raw_text)
                if not clean_text_snippet:
                    raise ValueError("Could not find JSON in the output")
                
                # Step 2: Parse
                data = parse_json(clean_text_snippet)
                if not data:
                    raise ValueError("Could not parse JSON. Check syntax.")
                
                # Step 3: Sanitize (if specs provided)
                if allowed_keys or type_specs:
                    data = sanitize_data(data, allowed_keys, type_specs)
                    if "_error" in data:
                        raise ValueError(data["_error"])
                
                # Step 4: Validate against schema (if provided)
                if schema and PYDANTIC_AVAILABLE:
                    result = validate_against_schema(data, schema)
                    if isinstance(result, dict) and "_validation_error" in result:
                        raise ValueError(result["_validation_error"])
                    return result
                
                # No schema - return parsed dict
                return data
                
            except Exception as e:
                last_error = str(e)
                log_event(
                    f"Structured extraction failed (fresh={fresh_attempt+1}, repair={repair_attempt+1}): {e}",
                    "WARNING"
                )
                
                # Build repair prompt
                error_msg = f"Error: {last_error}. Please fix the JSON and return only the valid JSON object."
                messages.append(error_msg)
                
                # Don't repair on last attempt of last fresh retry
                if fresh_attempt == max_fresh_retries - 1 and repair_attempt == max_repair_attempts - 1:
                    break
        
        log_event(f"Fresh retry {fresh_attempt + 1} exhausted. Starting fresh...", "WARNING")
    
    # All attempts failed
    log_event(
        f"Structured extraction failed after {max_fresh_retries} fresh attempts: {last_error}",
        "ERROR"
    )
    return {"_extraction_error": last_error, "_prompt": prompt[:500]}


def get_json_schema_for_provider(
    schema: Type[BaseModel],
    provider: str
) -> Dict[str, Any]:
    """
    Get JSON schema formatted for a specific provider.
    
    Different providers expect the schema in different formats:
    - vLLM: Uses guided_json in extra_body
    - Gemini: Uses responseSchema in generationConfig
    - OpenAI: Uses json_schema in response_format
    
    Args:
        schema: Pydantic BaseModel class
        provider: Provider name (vllm, gemini, openai, etc.)
        
    Returns:
        Provider-specific schema configuration
    """
    if not PYDANTIC_AVAILABLE or not issubclass(schema, BaseModel):
        return {}
    
    try:
        json_schema = schema.model_json_schema()
    except AttributeError:
        # Pydantic v1 fallback
        json_schema = schema.schema()
    
    if provider == "vllm":
        return {"guided_json": json_schema}
    
    elif provider == "gemini":
        return {
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": json_schema
            }
        }
    
    elif provider in ("openai", "openrouter"):
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": json_schema,
                    "strict": True
                }
            }
        }
    
    # Provider doesn't support constrained decoding
    return {}


# Convenience wrapper for the most common use case
async def extract_json_from_llm(
    prompt: str,
    schema: Optional[Type[T]] = None,
    **kwargs
) -> Union[T, Dict[str, Any]]:
    """
    Convenience function to extract structured JSON from an LLM call.
    
    This is the main entry point for structured extraction.
    
    Args:
        prompt: The prompt to send
        schema: Optional Pydantic schema for validation
        **kwargs: Additional arguments passed to reliable_extract
        
    Returns:
        Validated schema instance or parsed dict
    """
    return await reliable_extract(prompt, schema=schema, **kwargs)
