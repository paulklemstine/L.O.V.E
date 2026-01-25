
import pytest
from core.token_utils import count_tokens_for_api_models, _get_encoding

def test_token_counting_correctness():
    text = "Hello world"
    # "Hello" " world" -> 2 tokens usually
    count = count_tokens_for_api_models(text)
    assert count > 0
    assert count <= 5 # "Hello", " ", "world" max 3-4

def test_empty_string():
    assert count_tokens_for_api_models("") == 0

def test_large_text():
    text = "word " * 1000
    count = count_tokens_for_api_models(text)
    # 1000 words + spaces -> ~1000-2000 tokens
    assert count > 500

def test_caching_behavior():
    # Call multiple times
    text = "Unique text for caching test"
    count1 = count_tokens_for_api_models(text)
    count2 = count_tokens_for_api_models(text)
    assert count1 == count2

def test_encoding_cache():
    # Check that _get_encoding returns the same object
    enc1 = _get_encoding()
    enc2 = _get_encoding()
    assert enc1 is enc2
    assert enc1 is not None
