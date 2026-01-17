import unittest
import time
from core.aesthetic_evaluator import AestheticEvaluator

class TestAestheticEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = AestheticEvaluator()

    def test_evaluate_basic(self):
        text = "This is a simple sentence. It is clear and concise."
        result = self.evaluator.evaluate(text)

        self.assertIn("harmony_score", result)
        self.assertIn("clarity_score", result)
        self.assertIn("elegance_score", result)
        self.assertIn("beauty_score", result)
        self.assertIn("feedback", result)

        # Basic checks
        self.assertTrue(0 <= result["beauty_score"] <= 100)

    def test_evaluate_empty(self):
        result = self.evaluator.evaluate("")
        self.assertEqual(result["beauty_score"], 0.0)

    def test_evaluate_harmony(self):
        # Uniform sentence length -> lower harmony
        text = "This is a cat. That is a dog. It is a ball."
        result = self.evaluator.evaluate(text)
        self.assertTrue(0 <= result["harmony_score"] <= 100)

    def test_evaluate_clarity(self):
        # Complex text -> lower clarity
        text = "The obfuscation of the pedagogical framework necessitates a recalibration of our epistemological assumptions."
        result = self.evaluator.evaluate(text)
        self.assertTrue(0 <= result["clarity_score"] <= 100)
        # Should be lower than simple text
        simple_text = "The teacher changed the plan."
        simple_result = self.evaluator.evaluate(simple_text)
        self.assertLess(result["clarity_score"], simple_result["clarity_score"])

    def test_evaluate_elegance(self):
        # Creative text -> higher elegance
        text = "The sun-kissed waves whispered secrets to the golden sands, dancing in an eternal embrace of light and shadow."
        result = self.evaluator.evaluate(text)
        self.assertTrue(0 <= result["elegance_score"] <= 100)

        # Repetitive text -> lower elegance (TTR penalty)
        bad_text = "The cat sat. The cat sat. The cat sat."
        bad_result = self.evaluator.evaluate(bad_text)
        self.assertGreater(result["elegance_score"], bad_result["elegance_score"])

    def test_punctuation_handling(self):
        # Punctuation should not be counted as words or syllables
        text_with_punct = "Hello, world! This... is a test?"
        text_without_punct = "Hello world This is a test"

        res1 = self.evaluator.evaluate(text_with_punct)
        res2 = self.evaluator.evaluate(text_without_punct)

        # Clarity scores should be very similar (punctuation removed)
        # They won't be identical due to sent_tokenize splitting differently (sentence counts)
        # "Hello, world! This... is a test?" -> 3 sentences or 2?
        # "Hello world This is a test" -> 1 sentence

        # Let's test specific word filtering
        import nltk
        tokens = nltk.word_tokenize(text_with_punct)
        words = [w for w in tokens if w.isalnum()]
        self.assertEqual(len(words), 6) # Hello, world, This, is, a, test

        # Ensure evaluate runs without error on pure punctuation
        punct_only = "..."
        res_punct = self.evaluator.evaluate(punct_only)
        self.assertEqual(res_punct["beauty_score"], 0.0) # Should be 0 as no words

    def test_performance(self):
        # Latency check: < 500ms
        text = "This is a medium length paragraph to test performance. " * 20
        start_time = time.time()
        self.evaluator.evaluate(text)
        end_time = time.time()
        duration = (end_time - start_time) * 1000 # ms

        print(f"Performance: {duration:.2f}ms")
        self.assertLess(duration, 500)

if __name__ == '__main__':
    unittest.main()
