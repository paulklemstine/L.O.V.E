import json
import re
import ast
import logging
from typing import Dict, Any, Optional, Union


def extract_clean_text_content(response: str) -> str:
    """
    Extracts clean text content from LLM responses that may contain unwanted metadata.
    
    Some free OpenRouter models may return responses that:
    - Echo back the original API request structure as JSON
    - Wrap content in JSON with keys like 'requests', 'model', 'messages'
    - Include explanatory text after the actual content
    
    This function attempts to extract just the useful text content.
    
    Args:
        response: Raw LLM response string
        
    Returns:
        Cleaned text content, or empty string if the response is malformed/echoed
    """
    if not response or not isinstance(response, str):
        return ""
    
    response = response.strip()
    
    # Check if it looks like an echoed API request (JSON with 'requests' key)
    if response.startswith("{") or response.startswith("["):
        try:
            parsed = json.loads(response)
            
            if isinstance(parsed, dict):
                # Detect echoed API request patterns
                if "requests" in parsed and isinstance(parsed["requests"], list):
                    # This is an echoed API request, not a valid response
                    logging.warning("Detected echoed API request in LLM response - model may be malfunctioning")
                    return ""
                
                # Extract from standard response formats
                if "choices" in parsed and isinstance(parsed["choices"], list):
                    if parsed["choices"]:
                        msg = parsed["choices"][0].get("message", {})
                        if isinstance(msg, dict) and "content" in msg:
                            return msg.get("content", "")
                
                # Direct content extraction
                for key in ["content", "text", "result", "response", "output"]:
                    if key in parsed and isinstance(parsed[key], str):
                        return parsed[key]
                        
        except json.JSONDecodeError:
            # Not valid JSON, continue with the original content
            pass
    
    # Remove common explanatory suffixes that models add
    explanation_patterns = [
        r"\n\nThe (?:ANSI art|above|output).*$",
        r"\n\nThis (?:ANSI art|representation).*$", 
        r"\n\nI (?:created|generated|made).*$",
        r"\n\nHere (?:is|are).*$",
        r"\n\n---.*$",
        r"\n\nNote:.*$",
    ]
    
    for pattern in explanation_patterns:
        response = re.sub(pattern, "", response, flags=re.DOTALL | re.IGNORECASE)
    
    # Also strip thinking tags from text content
    response = strip_thinking_tags(response)
    
    return response.strip()


def strip_thinking_tags(response: str) -> str:
    """
    Removes <think>...</think> blocks and common conversational prefixes.
    """
    if not response:
        return ""
        
    # 1. Remove <think> blocks (handling unclosed tags at end of string)
    response = re.sub(r'<think>.*?(?:</think>|$)', '', response, flags=re.DOTALL | re.IGNORECASE)
    
    # 2. Remove "Thinking Process:" style blocks (common in some fine-tunes)
    response = re.sub(r'Thinking Process:.*?\n\n', '', response, flags=re.DOTALL | re.IGNORECASE)
    
    # 3. Clean up leading conversational filler *if* it precedes a JSON block or just generally
    # This matches "Okay, here is..." at the start of the string
    # We be careful not to strip legitimate content if it's a text-only response, 
    # but strictly speaking, "Okay, here is" is usually noise.
    conversational_prefixes = [
        r"^Okay, (?:let's|here|I).+?[:\.\n]\s*",
        r"^Sure, (?:here|I).+?[:\.\n]\s*",
        r"^Here is the (?:JSON|code|output).+?[:\.\n]\s*",
        r"^I have generated.+?[:\.\n]\s*",
    ]
    
    clean_response = response.strip()
    for pattern in conversational_prefixes:
        clean_response = re.sub(pattern, "", clean_response, flags=re.DOTALL | re.IGNORECASE) # DOTALL important for multi-line prefixes
        
    return clean_response.strip()


def smart_parse_llm_response(response: str, expected_keys: Optional[list] = None) -> Dict[str, Any]:
    """
    Intelligently parses LLM responses with multiple fallback strategies.
    
    Strategies (in order):
    1. Extract from <json> tags
    2. Parse key-value format (Thought: "..." Action: {...})
    3. Extract from markdown code blocks (```json ... ```)
    4. Parse as JSON (double quotes)
    5. Parse as Python dict (single quotes) using ast.literal_eval
    6. Extract JSON from mixed text using regex
    7. Return error dict with raw response
    
    Args:
        response: Raw LLM response string
        expected_keys: Optional list of expected keys for validation
        
    Returns:
        Parsed dict or error dict with '_parse_error' key
    """
    if not response or not isinstance(response, str):
        return {
            "_parse_error": "Empty or invalid response",
            "_raw_response": str(response)[:500]
        }
    
    response = response.strip()
    
    # Pre-processing: Strip <think> tags and conversational prefixes
    response = strip_thinking_tags(response)

    # Strategy 0: Extract from <json> tags (New Priority)
    result = _extract_from_xml(response)
    if result and not result.get('_parse_error'):return result

    # Strategy 1: Parse as JSON (Priority for valid JSON)
    try:
        result = json.loads(response)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: Parse key-value format (Thought: "..." Action: {...})
    try:
        result = _parse_key_value_format(response)
        if result and not result.get('_parse_error'):
            return result
    except Exception:
        pass
    
    # Strategy 3: Extract from markdown code blocks
    try:
        result = _extract_from_markdown(response)
        if result and not result.get('_parse_error'):
            return result
    except Exception:
        pass
    
    # Strategy 4: Parse as Python dict using ast.literal_eval
    try:
        result = ast.literal_eval(response)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    
    # Strategy 5: Extract JSON from mixed text using regex
    try:
        result = _extract_json_from_text(response)
        if result and not result.get('_parse_error'):
            return result
    except Exception:
        pass
    
    # Strategy 6: Return error
    return {
        "_parse_error": "Failed to parse LLM response with any strategy",
        "_raw_response": response[:500]
    }


def _parse_key_value_format(response: str) -> Optional[Dict[str, Any]]:
    """
    Parses key-value format like:
    Thought: "..."
    Action: {"tool_name": "...", "arguments": {...}}
    
    Returns dict with lowercase keys: {"thought": "...", "action": {...}}
    """
    result = {}
    
    # Pattern to match Key: Value pairs
    # Handles both quoted strings and JSON objects
    # Enforce start of string or newline to avoid matching text inside values
    pattern = r'(?:^|\n)(\w+):\s*(.+?)(?=\n\w+:|$)'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
    
    if not matches:
        return None
    
    for key, value in matches:
        key_lower = key.lower().strip()
        value_stripped = value.strip()
        
        # Try to parse value as JSON first
        try:
            # Remove surrounding quotes if present
            if value_stripped.startswith('"') and value_stripped.endswith('"'):
                # It's a quoted string
                parsed_value = value_stripped[1:-1]
            elif value_stripped.startswith('{') or value_stripped.startswith('['):
                # It's a JSON object or array
                parsed_value = json.loads(value_stripped)
            else:
                # Try parsing as JSON anyway
                try:
                    parsed_value = json.loads(value_stripped)
                except:
                    # Keep as string
                    parsed_value = value_stripped
        except json.JSONDecodeError:
            # Try ast.literal_eval for Python dict format
            try:
                parsed_value = ast.literal_eval(value_stripped)
            except:
                # Keep as string, removing quotes if present
                if value_stripped.startswith('"') and value_stripped.endswith('"'):
                    parsed_value = value_stripped[1:-1]
                else:
                    parsed_value = value_stripped
        
        result[key_lower] = parsed_value
    
    return result if result else None


def _extract_from_markdown(response: str) -> Optional[Dict[str, Any]]:
    """
    Extracts JSON from markdown code blocks.
    Handles: ```json\n{...}\n``` or ```\n{...}\n```
    """
    # Pattern to match markdown code blocks
    pattern = r'```(?:json)?\s*\n(.*?)\n```'
    match = re.search(pattern, response, re.DOTALL)
    
    if not match:
        return None
    
    content = match.group(1).strip()
    
    # Try parsing as JSON
    try:
        result = json.loads(content)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    
    # Try parsing as Python dict
    try:
        result = ast.literal_eval(content)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    
    return None


def _extract_json_from_text(response: str) -> Optional[Dict[str, Any]]:
    """
    Extracts JSON object or array from mixed text using regex.
    Looks for the first complete JSON structure.
    """
    # Pattern to match JSON objects or arrays
    # This is a simplified pattern - may not handle all edge cases
    pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
    match = re.search(pattern, response, re.DOTALL)
    
    if not match:
        return None
    
    json_str = match.group(1)
    
    # Try parsing as JSON
    try:
        result = json.loads(json_str)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    
    # Try parsing as Python dict
    try:
        result = ast.literal_eval(json_str)
        if isinstance(result, dict):
            return result
    except (ValueError, SyntaxError):
        pass
    
    return None


def validate_parsed_response(parsed: Dict[str, Any], expected_keys: list) -> bool:
    """
    Validates that parsed response contains expected keys.
    
    Args:
        parsed: Parsed response dict
        expected_keys: List of required keys
        
    Returns:
        True if all expected keys are present, False otherwise
    """
    if parsed.get('_parse_error'):
        return False
    
    return all(key in parsed for key in expected_keys)


def _extract_from_xml(response: str) -> Optional[Dict[str, Any]]:
    """
    Extracts JSON from <json> tags.
    Supports <json>...</json>
    """
    pattern = r'<json>\s*(.*?)\s*</json>'
    match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
    
    if not match:
        return None
        
    content = match.group(1).strip()
    
    # Reuse existing extractors or parse directly
    # Try parsing as JSON
    try:
        result = json.loads(content)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
        
    # Try ast
    try:
        result = ast.literal_eval(content)
        if isinstance(result, dict):
            return result
    except:
        pass
        
    # If standard parse fails, try the regex extractor on the content
    return _extract_json_from_text(content)
