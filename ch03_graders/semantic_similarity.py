"""
Semantic Similarity Graders
===========================

Embedding-based graders for semantic comparison.
Use when exact string matching is too strict and you need
to capture paraphrases or semantically equivalent responses.

Book Reference: Chapter 3, Section "Code-Based Graders"
- Cosine similarity: Vector-based comparison for semantic similarity
- Useful for FAQ consistency checking where paraphrased answers should score highly

Dependencies:
    pip install sentence-transformers numpy scipy
"""

import numpy as np
from typing import List, Optional, Union
from dataclasses import dataclass


@dataclass
class SimilarityResult:
    """Result of a semantic similarity comparison."""
    score: float  # 0.0 to 1.0 (cosine similarity mapped to this range)
    passed: bool  # Whether score meets threshold
    raw_similarity: float  # Raw cosine similarity (-1 to 1)
    explanation: Optional[str] = None


class SemanticSimilarityGrader:
    """
    Grade responses based on semantic similarity using embeddings.

    Uses sentence-transformers for high-quality text embeddings.
    Cosine similarity measures angle between embedding vectors.

    Book Reference: Chapter 3 - "Cosine similarity: Vector-based comparison
    for semantic similarity. Useful for FAQ consistency checking."
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.8,
        device: Optional[str] = None,
    ):
        """
        Initialize SemanticSimilarityGrader.

        Args:
            model_name: Sentence-transformers model name
                - "all-MiniLM-L6-v2": Fast, good quality (384 dims)
                - "all-mpnet-base-v2": Higher quality (768 dims)
                - "paraphrase-MiniLM-L6-v2": Optimized for paraphrase detection
            threshold: Minimum similarity score to pass (0.0 to 1.0)
            device: "cuda", "cpu", or None for auto-detect
        """
        self.threshold = threshold
        self.model_name = model_name
        self._model = None
        self._device = device

    @property
    def model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. "
                    "Install with: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(
                self.model_name,
                device=self._device
            )
        return self._model

    def _cosine_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Cosine similarity = (A . B) / (||A|| * ||B||)
        Range: -1 to 1 (1 = identical, 0 = orthogonal, -1 = opposite)
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array
        """
        return self.model.encode(text, convert_to_numpy=True)

    def grade(
        self,
        expected: str,
        actual: str,
        threshold: Optional[float] = None,
    ) -> SimilarityResult:
        """
        Grade actual response against expected using semantic similarity.

        Args:
            expected: Reference/gold answer
            actual: Model's response
            threshold: Override default threshold for this comparison

        Returns:
            SimilarityResult with similarity score and pass/fail

        Example:
            >>> grader = SemanticSimilarityGrader(threshold=0.8)
            >>> result = grader.grade(
            ...     "The cat sat on the mat",
            ...     "A feline rested on the rug"
            ... )
            >>> print(f"Similarity: {result.score:.3f}, Passed: {result.passed}")
        """
        threshold = threshold or self.threshold

        # Get embeddings
        emb_expected = self.get_embedding(expected)
        emb_actual = self.get_embedding(actual)

        # Compute cosine similarity
        raw_similarity = self._cosine_similarity(emb_expected, emb_actual)

        # Map from [-1, 1] to [0, 1] for easier interpretation
        # Though in practice, text similarity rarely goes below 0
        score = (raw_similarity + 1) / 2

        passed = score >= threshold

        return SimilarityResult(
            score=score,
            passed=passed,
            raw_similarity=raw_similarity,
            explanation=f"Cosine similarity: {raw_similarity:.4f} "
                       f"(threshold: {threshold})"
        )

    def grade_multiple(
        self,
        expected: str,
        actuals: List[str],
        threshold: Optional[float] = None,
    ) -> List[SimilarityResult]:
        """
        Grade multiple responses against a single reference.

        More efficient than calling grade() multiple times
        (reference embedding computed once).

        Args:
            expected: Reference answer
            actuals: List of model responses
            threshold: Similarity threshold

        Returns:
            List of SimilarityResult for each actual response
        """
        threshold = threshold or self.threshold

        # Get embeddings (batch for efficiency)
        emb_expected = self.get_embedding(expected)
        emb_actuals = self.model.encode(actuals, convert_to_numpy=True)

        results = []
        for i, emb_actual in enumerate(emb_actuals):
            raw_sim = self._cosine_similarity(emb_expected, emb_actual)
            score = (raw_sim + 1) / 2

            results.append(SimilarityResult(
                score=score,
                passed=score >= threshold,
                raw_similarity=raw_sim,
                explanation=f"Response {i}: cosine={raw_sim:.4f}"
            ))

        return results


class ROUGEGrader:
    """
    ROUGE-L grader for summarization evaluation.

    ROUGE-L measures longest common subsequence between
    reference and generated text.

    Book Reference: Chapter 3 - "ROUGE-L: Longest common subsequence metric
    for summarization. Measures how much of the reference summary appears
    in the generated summary."
    """

    def __init__(self, threshold: float = 0.5):
        """
        Initialize ROUGE-L grader.

        Args:
            threshold: Minimum ROUGE-L F1 score to pass
        """
        self.threshold = threshold

    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(x), len(y)
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def grade(self, reference: str, generated: str) -> SimilarityResult:
        """
        Compute ROUGE-L score between reference and generated text.

        ROUGE-L uses longest common subsequence (LCS):
        - Precision = LCS / len(generated)
        - Recall = LCS / len(reference)
        - F1 = 2 * P * R / (P + R)

        Example:
            >>> grader = ROUGEGrader(threshold=0.5)
            >>> result = grader.grade(
            ...     "The quick brown fox jumps over the lazy dog",
            ...     "A quick fox jumped over a lazy dog"
            ... )
            >>> print(f"ROUGE-L F1: {result.score:.3f}")
        """
        # Tokenize by whitespace (simple tokenization)
        ref_tokens = reference.lower().split()
        gen_tokens = generated.lower().split()

        if len(ref_tokens) == 0 or len(gen_tokens) == 0:
            return SimilarityResult(
                score=0.0,
                passed=False,
                raw_similarity=0.0,
                explanation="Empty reference or generated text"
            )

        # Compute LCS length
        lcs_len = self._lcs_length(ref_tokens, gen_tokens)

        # Compute precision and recall
        precision = lcs_len / len(gen_tokens)
        recall = lcs_len / len(ref_tokens)

        # Compute F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        passed = f1 >= self.threshold

        return SimilarityResult(
            score=f1,
            passed=passed,
            raw_similarity=f1,
            explanation=f"ROUGE-L: P={precision:.3f}, R={recall:.3f}, "
                       f"F1={f1:.3f} (threshold: {self.threshold})"
        )


class BLEUGrader:
    """
    BLEU score grader for translation/generation evaluation.

    BLEU measures n-gram precision between reference and generated text.
    """

    def __init__(self, threshold: float = 0.3, max_n: int = 4):
        """
        Initialize BLEU grader.

        Args:
            threshold: Minimum BLEU score to pass
            max_n: Maximum n-gram size (typically 4)
        """
        self.threshold = threshold
        self.max_n = max_n

    def _get_ngrams(self, tokens: List[str], n: int) -> dict:
        """Get n-gram counts from tokens."""
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams

    def _clip_count(self, candidate_ngrams: dict, reference_ngrams: dict) -> int:
        """Compute clipped n-gram count."""
        total = 0
        for ngram, count in candidate_ngrams.items():
            ref_count = reference_ngrams.get(ngram, 0)
            total += min(count, ref_count)
        return total

    def grade(self, reference: str, generated: str) -> SimilarityResult:
        """
        Compute BLEU score between reference and generated text.

        Example:
            >>> grader = BLEUGrader(threshold=0.3)
            >>> result = grader.grade(
            ...     "The cat sat on the mat",
            ...     "The cat is on the mat"
            ... )
            >>> print(f"BLEU: {result.score:.3f}")
        """
        ref_tokens = reference.lower().split()
        gen_tokens = generated.lower().split()

        if len(gen_tokens) == 0:
            return SimilarityResult(
                score=0.0, passed=False, raw_similarity=0.0,
                explanation="Empty generated text"
            )

        # Compute n-gram precisions
        precisions = []
        for n in range(1, self.max_n + 1):
            if len(gen_tokens) < n:
                precisions.append(0.0)
                continue

            ref_ngrams = self._get_ngrams(ref_tokens, n)
            gen_ngrams = self._get_ngrams(gen_tokens, n)

            clipped = self._clip_count(gen_ngrams, ref_ngrams)
            total = sum(gen_ngrams.values())

            precision = clipped / total if total > 0 else 0.0
            precisions.append(precision)

        # Geometric mean of precisions
        if 0 in precisions:
            bleu = 0.0
        else:
            log_sum = sum(np.log(p) for p in precisions)
            bleu = np.exp(log_sum / len(precisions))

        # Brevity penalty
        bp = min(1.0, np.exp(1 - len(ref_tokens) / len(gen_tokens)))
        bleu *= bp

        passed = bleu >= self.threshold

        return SimilarityResult(
            score=bleu,
            passed=passed,
            raw_similarity=bleu,
            explanation=f"BLEU-{self.max_n}: {bleu:.4f} (threshold: {self.threshold})"
        )


# Demo usage
if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 3: Semantic Similarity Graders Demo")
    print("=" * 60)

    # Example 1: Embedding-based Similarity
    print("\n1. SemanticSimilarityGrader (Embeddings)")
    print("-" * 40)

    # Note: This requires sentence-transformers to be installed
    try:
        grader = SemanticSimilarityGrader(
            model_name="all-MiniLM-L6-v2",
            threshold=0.7
        )

        test_pairs = [
            ("The cat sat on the mat", "A feline rested on the rug"),
            ("What is the capital of France?", "Name France's capital city"),
            ("The weather is nice today", "I enjoy programming in Python"),
        ]

        for text1, text2 in test_pairs:
            result = grader.grade(text1, text2)
            print(f"'{text1[:30]}...' vs '{text2[:30]}...'")
            print(f"  Similarity: {result.score:.3f}, Passed: {result.passed}")
    except ImportError:
        print("Skipping embedding demo (sentence-transformers not installed)")

    # Example 2: ROUGE-L
    print("\n2. ROUGEGrader (Summarization)")
    print("-" * 40)
    grader = ROUGEGrader(threshold=0.5)

    reference = "The quick brown fox jumps over the lazy dog"
    candidates = [
        "A quick fox jumped over a lazy dog",
        "The fox is quick and brown",
        "Python is a programming language",
    ]

    for candidate in candidates:
        result = grader.grade(reference, candidate)
        print(f"Candidate: '{candidate}'")
        print(f"  {result.explanation}")

    # Example 3: BLEU
    print("\n3. BLEUGrader (Translation)")
    print("-" * 40)
    grader = BLEUGrader(threshold=0.3)

    reference = "The cat sat on the mat"
    candidates = [
        "The cat is on the mat",
        "A cat sat on a mat",
        "The dog ran in the park",
    ]

    for candidate in candidates:
        result = grader.grade(reference, candidate)
        print(f"Candidate: '{candidate}'")
        print(f"  {result.explanation}")

    print("\n" + "=" * 60)
    print("Key Takeaway: Semantic similarity graders are code-based but")
    print("capture meaning, not just exact strings. Use for paraphrases")
    print("and when multiple valid phrasings exist.")
    print("=" * 60)
