"""
Tests for structured output extraction module.

Tests the clean_text, parse_json, validate_against_schema, and reliable_extract
functions from core/structured_output.py.
"""

import unittest
import json
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.structured_output import (
    clean_text,
    parse_json,
    validate_against_schema,
    sanitize_data,
    get_json_schema_for_provider,
    PYDANTIC_AVAILABLE
)

if PYDANTIC_AVAILABLE:
    from core.schemas import TaskAction, ThoughtAction, ReviewDecision


class TestCleanText(unittest.TestCase):
    """Tests for clean_text function."""
    
    def test_clean_text_removes_markdown_fences(self):
        """Test that markdown code fences are removed."""
        text = '```json\n{"key": "value"}\n```'
        result = clean_text(text)
        self.assertEqual(result, '{"key": "value"}')
    
    def test_clean_text_extracts_json_object(self):
        """Test JSON object extraction from mixed text."""
        text = 'Here is the result: {"foo": "bar"} Hope this helps!'
        result = clean_text(text)
        self.assertEqual(result, '{"foo": "bar"}')
    
    def test_clean_text_extracts_json_array(self):
        """Test JSON array extraction from mixed text."""
        text = 'The items are: [1, 2, 3] End.'
        result = clean_text(text)
        self.assertEqual(result, '[1, 2, 3]')
    
    def test_clean_text_handles_empty(self):
        """Test that empty input returns empty string."""
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")
    
    def test_clean_text_preserves_valid_json(self):
        """Test that valid JSON is preserved."""
        text = '{"nested": {"key": "value"}}'
        result = clean_text(text)
        self.assertEqual(result, text)


class TestParseJson(unittest.TestCase):
    """Tests for parse_json function."""
    
    def test_parse_standard_json(self):
        """Test parsing standard JSON with double quotes."""
        text = '{"key": "value", "number": 42}'
        result = parse_json(text)
        self.assertEqual(result, {"key": "value", "number": 42})
    
    def test_parse_python_dict_single_quotes(self):
        """Test parsing Python dict with single quotes."""
        text = "{'key': 'value', 'number': 42}"
        result = parse_json(text)
        self.assertEqual(result, {"key": "value", "number": 42})
    
    def test_parse_with_trailing_comma(self):
        """Test parsing JSON with trailing commas."""
        text = '{"key": "value", "number": 42,}'
        result = parse_json(text)
        self.assertEqual(result, {"key": "value", "number": 42})
    
    def test_parse_returns_none_for_invalid(self):
        """Test that invalid JSON returns None."""
        text = "not json at all"
        result = parse_json(text)
        self.assertIsNone(result)
    
    def test_parse_wraps_array(self):
        """Test that arrays are wrapped in dict."""
        text = '[1, 2, 3]'
        result = parse_json(text)
        self.assertEqual(result, {"items": [1, 2, 3]})


class TestSanitizeData(unittest.TestCase):
    """Tests for sanitize_data function."""
    
    def test_filter_allowed_keys(self):
        """Test filtering to allowed keys only."""
        data = {"a": 1, "b": 2, "c": 3}
        result = sanitize_data(data, allowed_keys=["a", "b"])
        self.assertEqual(result, {"a": 1, "b": 2})
    
    def test_type_coercion_int(self):
        """Test integer type coercion."""
        data = {"count": "42"}
        result = sanitize_data(data, type_specs={"count": int})
        self.assertEqual(result["count"], 42)
    
    def test_type_coercion_float(self):
        """Test float type coercion."""
        data = {"score": "0.75"}
        result = sanitize_data(data, type_specs={"score": float})
        self.assertEqual(result["score"], 0.75)
    
    def test_type_coercion_bool(self):
        """Test boolean type coercion."""
        data = {"enabled": "true"}
        result = sanitize_data(data, type_specs={"enabled": bool})
        self.assertEqual(result["enabled"], True)


@unittest.skipIf(not PYDANTIC_AVAILABLE, "Pydantic not installed")
class TestSchemaValidation(unittest.TestCase):
    """Tests for schema validation with Pydantic."""
    
    def test_validate_task_action(self):
        """Test validating TaskAction schema."""
        data = {"tool_name": "search", "arguments": {"query": "test"}}
        result = validate_against_schema(data, TaskAction)
        self.assertIsInstance(result, TaskAction)
        self.assertEqual(result.tool_name, "search")
    
    def test_validate_review_decision(self):
        """Test validating ReviewDecision schema."""
        data = {
            "approved": True,
            "feedback": "Looks good",
            "confidence": 0.9
        }
        result = validate_against_schema(data, ReviewDecision)
        self.assertIsInstance(result, ReviewDecision)
        self.assertTrue(result.approved)
    
    def test_validation_error_returns_dict(self):
        """Test that validation errors return error dict."""
        data = {"not_a_valid_field": "value"}
        result = validate_against_schema(data, TaskAction)
        self.assertIsInstance(result, dict)
        self.assertIn("_validation_error", result)


@unittest.skipIf(not PYDANTIC_AVAILABLE, "Pydantic not installed")
class TestGetJsonSchemaForProvider(unittest.TestCase):
    """Tests for provider-specific schema formatting."""
    
    def test_vllm_schema_format(self):
        """Test vLLM guided_json schema format."""
        result = get_json_schema_for_provider(TaskAction, "vllm")
        self.assertIn("guided_json", result)
        self.assertIsInstance(result["guided_json"], dict)
    
    def test_gemini_schema_format(self):
        """Test Gemini generationConfig schema format."""
        result = get_json_schema_for_provider(TaskAction, "gemini")
        self.assertIn("generationConfig", result)
        self.assertIn("responseMimeType", result["generationConfig"])
        self.assertEqual(result["generationConfig"]["responseMimeType"], "application/json")
    
    def test_openai_schema_format(self):
        """Test OpenAI response_format schema format."""
        result = get_json_schema_for_provider(TaskAction, "openai")
        self.assertIn("response_format", result)
        self.assertEqual(result["response_format"]["type"], "json_schema")
    
    def test_unknown_provider_returns_empty(self):
        """Test that unknown provider returns empty dict."""
        result = get_json_schema_for_provider(TaskAction, "unknown_provider")
        self.assertEqual(result, {})


class TestIntegration(unittest.TestCase):
    """Integration tests combining clean_text + parse_json."""
    
    def test_full_pipeline_markdown_json(self):
        """Test full cleaning and parsing pipeline."""
        raw = '```json\n{"thought": "Thinking...", "action": {"tool_name": "search"}}\n```'
        cleaned = clean_text(raw)
        parsed = parse_json(cleaned)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["thought"], "Thinking...")
    
    def test_full_pipeline_messy_output(self):
        """Test pipeline with LLM-style messy output."""
        raw = '''Sure! Here's your JSON:
        
{"result": "success", "count": 5}

I hope this helps!'''
        cleaned = clean_text(raw)
        parsed = parse_json(cleaned)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["result"], "success")


if __name__ == '__main__':
    unittest.main()
