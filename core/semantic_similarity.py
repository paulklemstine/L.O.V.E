"""
Semantic Similarity Module for L.O.V.E.

Provides semantic similarity checking for subliminal phrases and visual descriptions
to prevent repetition while allowing thematically related content.

Uses TF-IDF with cosine similarity for lightweight, dependency-free operation.
"""

import logging
import re
import math
from typing import List, Tuple, Optional, Dict, Set
from collections import Counter

logger = logging.getLogger("SemanticSimilarity")

def log_event(message: str, level: str = "INFO"):
    """Compatibility helper for logging."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    logger.log(lvl, message)

class SemanticSimilarityChecker:
    """
    Lightweight semantic similarity checker using TF-IDF and cosine similarity.
    No external dependencies required.
    """
    
    def __init__(self, similarity_threshold: float = 0.80):
        """
        Initialize the similarity checker.
        
        Args:
            similarity_threshold: Phrases above this similarity score are considered duplicates (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        self._stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'under',
            'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
            'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also'
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and normalize text."""
        # Lowercase and extract words
        text = text.lower()
        words = re.findall(r'\\b[a-z]+\\b', text)
        # Remove stopwords and short words
        return [w for w in words if w not in self._stopwords and len(w) > 2]
    
    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency for a document."""
        counts = Counter(tokens)
        total = len(tokens) if tokens else 1
        return {word: count / total for word, count in counts.items()}
    
    def _compute_idf(self, documents: List[List[str]]) -> Dict[str, float]:
        """Compute inverse document frequency across all documents."""
        n_docs = len(documents)
        if n_docs == 0:
            return {}
        
        # Count documents containing each term
        doc_freq = Counter()
        for doc in documents:
            unique_terms = set(doc)
            doc_freq.update(unique_terms)
        
        # Compute IDF with smoothing
        idf = {}
        for term, freq in doc_freq.items():
            idf[term] = math.log((n_docs + 1) / (freq + 1)) + 1
        
        return idf
    
    def _compute_tfidf(self, tf: Dict[str, float], idf: Dict[str, float]) -> Dict[str, float]:
        """Compute TF-IDF scores."""
        return {term: tf_val * idf.get(term, 1.0) for term, tf_val in tf.items()}
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Compute cosine similarity between two TF-IDF vectors."""
        # Get all terms
        all_terms = set(vec1.keys()) | set(vec2.keys())
        
        if not all_terms:
            return 0.0
        
        # Compute dot product and magnitudes
        dot_product = sum(vec1.get(t, 0) * vec2.get(t, 0) for t in all_terms)
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values())) if vec1 else 0
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values())) if vec2 else 0
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def compute_similarity(self, phrase1: str, phrase2: str) -> float:
        """
        Compute semantic similarity between two phrases.
        
        Returns:
            Similarity score between 0.0 (completely different) and 1.0 (identical)
        """
        tokens1 = self._tokenize(phrase1)
        tokens2 = self._tokenize(phrase2)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Build corpus for IDF
        documents = [tokens1, tokens2]
        idf = self._compute_idf(documents)
        
        # Compute TF-IDF for each
        tfidf1 = self._compute_tfidf(self._compute_tf(tokens1), idf)
        tfidf2 = self._compute_tfidf(self._compute_tf(tokens2), idf)
        
        return self._cosine_similarity(tfidf1, tfidf2)
    
    def check_novelty(self, phrase: str, history: List[str], threshold: Optional[float] = None) -> bool:
        """
        Check if a phrase is novel compared to history.
        
        Args:
            phrase: The new phrase to check
            history: List of previously used phrases
            threshold: Optional override for similarity threshold
            
        Returns:
            True if the phrase is novel (below threshold), False if too similar
        """
        if not phrase or not history:
            return True
        
        threshold = threshold if threshold is not None else self.similarity_threshold
        
        for past_phrase in history:
            similarity = self.compute_similarity(phrase, past_phrase)
            if similarity >= threshold:
                log_event(
                    f"Phrase rejected (similarity={similarity:.2f}): '{phrase[:30]}...' ≈ '{past_phrase[:30]}...'",
                    level="WARNING"
                )
                return False
        
        return True
    
    def get_similar_phrases(self, phrase: str, history: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Find the most similar phrases from history.
        
        Args:
            phrase: The phrase to compare
            history: List of historical phrases
            top_k: Number of top similar phrases to return
            
        Returns:
            List of (phrase, similarity_score) tuples, sorted by similarity
        """
        if not phrase or not history:
            return []
        
        similarities = []
        for past_phrase in history:
            score = self.compute_similarity(phrase, past_phrase)
            similarities.append((past_phrase, score))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def exact_match_filter(self, phrase: str, history: List[str]) -> bool:
        """
        Strict exact match filter (final guardrail).
        
        Returns:
            True if phrase is NOT an exact match (passes filter)
        """
        phrase_normalized = phrase.lower().strip()
        for past in history:
            if past.lower().strip() == phrase_normalized:
                log_event(f"EXACT MATCH BLOCKED: '{phrase}'", level="WARNING")
                return False
        return True


# Global singleton instance
_similarity_checker: Optional[SemanticSimilarityChecker] = None


def get_similarity_checker() -> SemanticSimilarityChecker:
    """Get or create the global similarity checker instance."""
    global _similarity_checker
    if _similarity_checker is None:
        _similarity_checker = SemanticSimilarityChecker(similarity_threshold=0.80)
    return _similarity_checker


def check_phrase_novelty(phrase: str, history: List[str], threshold: float = 0.80) -> bool:
    """
    Convenience function to check phrase novelty.
    
    Args:
        phrase: New phrase to check
        history: List of past phrases
        threshold: Similarity threshold (default 0.80)
    
    Returns:
        True if novel, False if too similar to history
    """
    checker = get_similarity_checker()
    
    # First: exact match filter
    if not checker.exact_match_filter(phrase, history):
        return False
    
    # Second: semantic similarity filter
    return checker.check_novelty(phrase, history, threshold)
