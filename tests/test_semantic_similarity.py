import pytest
from core.semantic_similarity import SemanticSimilarityChecker

@pytest.fixture
def checker():
    return SemanticSimilarityChecker()

def test_tokenize(checker):
    text = "The quick Brown fox!"
    # stopwords: the
    # len > 2: fox, brown, quick
    tokens = checker._tokenize(text)
    assert set(tokens) == {"quick", "brown", "fox"}

def test_tokenize_stopwords(checker):
    text = "a an the is are"
    tokens = checker._tokenize(text)
    assert len(tokens) == 0

def test_compute_similarity_identical(checker):
    phrase = "Hello world"
    similarity = checker.compute_similarity(phrase, phrase)
    assert similarity == pytest.approx(1.0)

def test_compute_similarity_different(checker):
    phrase1 = "Apple banana"
    phrase2 = "Space rocket"
    similarity = checker.compute_similarity(phrase1, phrase2)
    assert similarity == 0.0

def test_compute_similarity_partial(checker):
    phrase1 = "hello world"
    phrase2 = "hello space"
    similarity = checker.compute_similarity(phrase1, phrase2)
    assert 0.0 < similarity < 1.0

def test_check_novelty(checker):
    history = ["apple banana", "orange juice"]
    # "apple banana" is already in history, so it should not be novel
    assert checker.check_novelty("apple banana", history) is False

    # "space rocket" is new
    assert checker.check_novelty("space rocket", history) is True

def test_check_novelty_threshold(checker):
    history = ["hello world"]
    # "hello earth" might be similar
    # threshold 0.0 -> everything rejected if > 0
    # threshold 1.0 -> everything accepted unless identical (almost)

    # Let's test explicit threshold
    # If "hello world" and "hello earth" share "hello", they have some similarity.
    # IDF: hello appears in both -> idf=1. earth/world appear in 1 -> idf=~1.4

    assert checker.check_novelty("hello earth", history, threshold=0.1) is False
    assert checker.check_novelty("hello earth", history, threshold=0.99) is True

def test_tokenize_empty(checker):
    assert len(checker._tokenize("")) == 0
