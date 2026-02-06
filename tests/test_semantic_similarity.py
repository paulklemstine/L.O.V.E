
import pytest
from core.semantic_similarity import SemanticSimilarityChecker, check_phrase_novelty

class TestSemanticSimilarityChecker:
    @pytest.fixture
    def checker(self):
        return SemanticSimilarityChecker(similarity_threshold=0.8)

    def test_tokenize(self, checker):
        text = "The quick brown fox jumps over the lazy dog."
        tokens = checker._tokenize(text)
        # Stopwords removed, short words removed
        # Note: 'the' is a stopword. 'over' is not in the default stopword list for this class.
        assert isinstance(tokens, tuple)
        assert 'quick' in tokens
        assert 'fox' in tokens
        assert 'the' not in tokens

    def test_compute_similarity_identical(self, checker):
        phrase = "The quick brown fox"
        assert checker.compute_similarity(phrase, phrase) == 1.0

    def test_compute_similarity_different(self, checker):
        phrase1 = "The quick brown fox"
        phrase2 = "completely different text unrelated"
        score = checker.compute_similarity(phrase1, phrase2)
        assert score < 0.2

    def test_compute_similarity_similar(self, checker):
        phrase1 = "The quick brown fox"
        phrase2 = "A fast brown fox"
        score = checker.compute_similarity(phrase1, phrase2)
        assert score > 0.5

    def test_check_novelty_novel(self, checker):
        history = ["Something completely different"]
        phrase = "The quick brown fox"
        assert checker.check_novelty(phrase, history) is True

    def test_check_novelty_duplicate(self, checker):
        history = ["The quick brown fox"]
        phrase = "The quick brown fox"
        # Exact match should be caught by compute_similarity being 1.0 >= 0.8
        assert checker.check_novelty(phrase, history) is False

    def test_check_novelty_similar(self, checker):
        history = ["The quick brown fox"]
        phrase = "A fast brown fox"
        assert isinstance(checker.check_novelty(phrase, history), bool)

    def test_get_similar_phrases(self, checker):
        history = [
            "The quick brown fox",
            "Something completely different",
            "A fast brown fox"
        ]
        phrase = "The quick brown fox"
        similar = checker.get_similar_phrases(phrase, history, top_k=2)
        assert len(similar) == 2
        assert similar[0][0] == "The quick brown fox"
        assert similar[0][1] == 1.0
        assert similar[1][0] == "A fast brown fox"

    def test_convenience_function(self):
        history = ["The quick brown fox"]
        assert check_phrase_novelty("Something else", history) is True
        assert check_phrase_novelty("The quick brown fox", history) is False

    def test_tokenize_caching_behavior(self, checker):
        # This test ensures that if we change return type to tuple, it still works
        t1 = checker._tokenize("hello world")
        t2 = checker._tokenize("hello world")
        assert t1 == t2
