"""
Internal Aesthetic Quality Evaluation Algorithm.

This module provides programmatically evaluated aesthetic quality of text responses
based on three dimensions: Harmony, Clarity, and Creative Elegance.
"""

import math
import statistics
import nltk
import functools
from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import Dict, Any, List

class AestheticEvaluator:
    """
    Evaluates text responses for aesthetic quality.
    """

    def __init__(self):
        # Ensure necessary NLTK data is available
        # In production, these should be pre-installed.
        # We attempt download only as a fallback.
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)

    def evaluate(self, text: str) -> Dict[str, Any]:
        """
        Evaluates the given text and returns a score breakdown.

        Args:
            text: The text to evaluate.

        Returns:
            A dictionary containing individual scores, the composite beauty score,
            and feedback.
        """
        if not text or not text.strip():
            return self._empty_result()

        # Pre-processing
        # Optimization: Use TextBlob to tokenize once and cache tokens
        blob = TextBlob(text)

        # Accessing blob.words triggers tokenization once and caches it in the blob object.
        # blob.tags will reuse these tokens later.
        raw_tokens = blob.words

        # Use simple string sentences to avoid TextBlob Sentence object creation overhead
        sentences = sent_tokenize(text)

        # Filter for actual words (alphanumeric) to fix punctuation skew
        words = [w for w in raw_tokens if w.isalnum()]

        # Calculate Dimensions
        harmony_score = self._calculate_harmony(sentences)
        clarity_score = self._calculate_clarity(sentences, words)
        elegance_score = self._calculate_elegance(blob, words)

        # Composite Score
        # Weights: Harmony 0.3, Clarity 0.3, Creative Elegance 0.4
        beauty_score = (harmony_score * 0.3) + (clarity_score * 0.3) + (elegance_score * 0.4)

        return {
            "harmony_score": round(harmony_score, 2),
            "clarity_score": round(clarity_score, 2),
            "elegance_score": round(elegance_score, 2),
            "beauty_score": round(beauty_score, 2),
            "feedback": self._generate_feedback(harmony_score, clarity_score, elegance_score)
        }

    def _calculate_harmony(self, sentences: List[str]) -> float:
        """
        Evaluates Coherence, Logical Flow, and Semantic Consistency.
        Proxies:
        - Sentence Length Variance: Too low = robotic, Too high = chaotic.
          We want a "Goldilocks" variance.
        """
        # Use filtered word count per sentence for accuracy
        sentence_lengths = []
        total_words = 0
        for s in sentences:
            s_tokens = word_tokenize(s)
            s_words = [t for t in s_tokens if t.isalnum()]
            length = len(s_words)
            sentence_lengths.append(length)
            total_words += length

        if total_words == 0:
            return 0.0

        if len(sentences) <= 1:
            return 70.0 # Neutral score for single sentence

        # Standard Deviation of sentence lengths
        std_dev = statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0
        mean_len = statistics.mean(sentence_lengths)

        # Coefficient of variation (CV) is a better measure than raw std_dev
        cv = std_dev / mean_len if mean_len > 0 else 0

        # Optimal CV is around 0.3 - 0.6 (some variation, but not wild)
        # If CV < 0.2 -> Too uniform (Score drops)
        # If CV > 0.8 -> Too chaotic (Score drops)

        score = 100.0

        if cv < 0.2:
            # Penalize uniformity
            penalty = (0.2 - cv) * 200 # Max penalty ~40
            score -= penalty
        elif cv > 0.8:
            # Penalize chaos
            penalty = (cv - 0.8) * 100 # Max penalty ~20 (chaos is better than uniformity)
            score -= penalty

        # Ensure bounds
        return max(0.0, min(100.0, score))

    def _calculate_clarity(self, sentences: List[str], words: List[str]) -> float:
        """
        Evaluates Readability and Conciseness.
        Uses Flesch Reading Ease approximation.
        """
        if not words or not sentences:
            return 0.0

        num_sentences = len(sentences)
        num_words = len(words)
        num_syllables = sum(self._count_syllables(w) for w in words)

        if num_words == 0:
            return 0.0

        # Flesch Reading Ease Formula
        # 206.835 - 1.015 (total words / total sentences) - 84.6 (total syllables / total words)
        try:
            asl = num_words / num_sentences
            asw = num_syllables / num_words
            flesch_score = 206.835 - (1.015 * asl) - (84.6 * asw)
        except ZeroDivisionError:
            return 0.0

        # Normalize Flesch score (can be < 0 or > 100) to 0-100
        # 60-70 is standard. 100 is very easy. 0 is very hard.
        # For "Clarity", we generally want higher, but maybe not 100 (too simple).
        # Let's just clamp it.
        return max(0.0, min(100.0, flesch_score))

    def _calculate_elegance(self, blob: TextBlob, words: List[str]) -> float:
        """
        Evaluates Creative Elegance: Originality, Stylistic Refinement.
        Proxies:
        - Vocabulary Richness (Type-Token Ratio)
        - Subjectivity (TextBlob) - Higher implies more personal/creative style.
        """
        if not words:
            return 0.0

        # 1. Type-Token Ratio (TTR)
        unique_words = set(w.lower() for w in words)
        ttr = len(unique_words) / len(words)

        # TTR score: Higher is better, but TTR naturally drops as length increases.
        # Simple heuristic: TTR > 0.5 is good for short texts.
        ttr_score = ttr * 100

        # 2. Subjectivity
        subjectivity = blob.sentiment.subjectivity * 100 # 0-100

        # 3. Adjective/Adverb Usage (Stylistic Refinement)
        # We want a healthy mix, not zero, not all.
        tags = blob.tags
        num_adjectives = sum(1 for w, t in tags if t.startswith('JJ'))
        num_adverbs = sum(1 for w, t in tags if t.startswith('RB'))

        modifiers_ratio = (num_adjectives + num_adverbs) / len(words)

        # Optimal modifiers ratio ~ 10-20%?
        modifier_score = 0
        if 0.10 <= modifiers_ratio <= 0.25:
            modifier_score = 100
        elif modifiers_ratio < 0.10:
            modifier_score = modifiers_ratio * 1000 # 0.05 -> 50
        else: # > 0.25
            modifier_score = max(0, 100 - ((modifiers_ratio - 0.25) * 200))

        # Weighted Elegance
        # TTR: 40%, Subjectivity: 30%, Modifiers: 30%
        final_score = (ttr_score * 0.4) + (subjectivity * 0.3) + (modifier_score * 0.3)

        return max(0.0, min(100.0, final_score))

    @staticmethod
    @functools.lru_cache(maxsize=1024)
    def _count_syllables(word: str) -> int:
        """Simple syllable counter."""
        if not word.isalnum():
            return 0

        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        return count

    def _generate_feedback(self, harmony: float, clarity: float, elegance: float) -> str:
        """Generates actionable feedback based on scores."""
        feedback = []

        if harmony < 60:
            feedback.append("Improve flow: sentence lengths are too uniform or too chaotic.")

        if clarity < 60:
            feedback.append("Improve clarity: text is difficult to read. Simplify sentences or vocabulary.")

        if elegance < 60:
            feedback.append("Boost creativity: vocabulary is repetitive or style is dry. Use more varied language.")

        if not feedback:
            feedback.append("Aesthetic quality is excellent.")

        return " ".join(feedback)

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "harmony_score": 0.0,
            "clarity_score": 0.0,
            "elegance_score": 0.0,
            "beauty_score": 0.0,
            "feedback": "No text provided."
        }
