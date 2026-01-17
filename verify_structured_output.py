#!/usr/bin/env python3
"""Quick verification script for structured output module."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.structured_output import (
    clean_text,
    parse_json,
    sanitize_data,
    validate_against_schema,
    get_json_schema_for_provider,
    PYDANTIC_AVAILABLE
)

print("=" * 60)
print("Structured Output Module Verification")
print("=" * 60)

# Test 1: clean_text
print("\n[TEST 1] clean_text")
test_input = '```json\n{"key": "value"}\n```'
result = clean_text(test_input)
expected = '{"key": "value"}'
status = "PASS" if result == expected else "FAIL"
print(f"  Input: {repr(test_input)}")
print(f"  Output: {repr(result)}")
print(f"  Status: {status}")

# Test 2: parse_json standard
print("\n[TEST 2] parse_json (standard JSON)")
test_input = '{"name": "test", "count": 42}'
result = parse_json(test_input)
status = "PASS" if result == {"name": "test", "count": 42} else "FAIL"
print(f"  Input: {test_input}")
print(f"  Output: {result}")
print(f"  Status: {status}")

# Test 3: parse_json single quotes
print("\n[TEST 3] parse_json (Python dict with single quotes)")
test_input = "{'name': 'test', 'count': 42}"
result = parse_json(test_input)
status = "PASS" if result == {"name": "test", "count": 42} else "FAIL"
print(f"  Input: {test_input}")
print(f"  Output: {result}")
print(f"  Status: {status}")

# Test 4: sanitize_data
print("\n[TEST 4] sanitize_data")
test_input = {"a": 1, "b": 2, "c": 3}
result = sanitize_data(test_input, allowed_keys=["a", "b"])
status = "PASS" if result == {"a": 1, "b": 2} else "FAIL"
print(f"  Input: {test_input}")
print(f"  Allowed keys: ['a', 'b']")
print(f"  Output: {result}")
print(f"  Status: {status}")

# Test 5: type coercion
print("\n[TEST 5] sanitize_data (type coercion)")
test_input = {"count": "42", "score": "0.75"}
result = sanitize_data(test_input, type_specs={"count": int, "score": float})
status = "PASS" if result["count"] == 42 and result["score"] == 0.75 else "FAIL"
print(f"  Input: {test_input}")
print(f"  Output: {result}")
print(f"  Status: {status}")

# Test 6: Pydantic integration
print("\n[TEST 6] Pydantic integration")
print(f"  PYDANTIC_AVAILABLE: {PYDANTIC_AVAILABLE}")
if PYDANTIC_AVAILABLE:
    from core.schemas import TaskAction, ThoughtAction
    
    test_data = {"tool_name": "search", "arguments": {"query": "test"}}
    result = validate_against_schema(test_data, TaskAction)
    is_valid = isinstance(result, TaskAction)
    status = "PASS" if is_valid else "FAIL"
    print(f"  Input: {test_data}")
    print(f"  Schema: TaskAction")
    print(f"  Output: {result}")
    print(f"  Is valid TaskAction: {is_valid}")
    print(f"  Status: {status}")
    
    # Test provider schema
    print("\n[TEST 7] get_json_schema_for_provider")
    for provider in ["vllm", "gemini", "openai"]:
        schema_config = get_json_schema_for_provider(TaskAction, provider)
        print(f"  {provider}: {list(schema_config.keys())}")
else:
    print("  Skipped (Pydantic not installed)")

print("\n" + "=" * 60)
print("Verification Complete")
print("=" * 60)
