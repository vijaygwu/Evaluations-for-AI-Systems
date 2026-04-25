"""
Exact Match Graders
===================

Code-based graders for deterministic evaluation tasks.
These are the foundation layer - fastest, most reliable, extremely scalable.

Book Reference: Chapter 3, Section "Code-Based Graders"

Key Patterns (from OpenAI Evals framework):
- Match: completion.startswith(reference)
- Includes: reference in completion
- FuzzyMatch: bidirectional substring matching
- JsonMatch: semantic JSON comparison
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union
from dataclasses import dataclass


@dataclass
class GradeResult:
    """Result of a grading operation."""
    score: float  # 0.0 to 1.0
    passed: bool  # Binary judgment
    explanation: Optional[str] = None
    metadata: Optional[dict] = None


class BaseGrader(ABC):
    """Abstract base class for all graders."""

    @abstractmethod
    def grade(self, expected: Any, actual: Any) -> GradeResult:
        """
        Grade an actual output against expected output.

        Args:
            expected: The reference/gold answer
            actual: The model's output

        Returns:
            GradeResult with score and pass/fail judgment
        """
        pass


class ExactMatchGrader(BaseGrader):
    """
    Exact string match grader.

    Use for classification tasks where outputs should match
    predefined labels exactly (e.g., sentiment: positive/negative/neutral).

    Book Reference: Chapter 3 - "Direct string comparison after normalization"
    """

    def __init__(
        self,
        normalize: bool = True,
        case_sensitive: bool = False,
        strip_whitespace: bool = True,
    ):
        """
        Initialize ExactMatchGrader.

        Args:
            normalize: Apply normalization before comparison
            case_sensitive: Whether comparison is case-sensitive
            strip_whitespace: Strip leading/trailing whitespace
        """
        self.normalize = normalize
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def _preprocess(self, text: str) -> str:
        """Apply preprocessing to text before comparison."""
        if self.strip_whitespace:
            text = text.strip()
        if not self.case_sensitive:
            text = text.lower()
        return text

    def grade(self, expected: str, actual: str) -> GradeResult:
        """
        Grade by exact string match.

        Example:
            >>> grader = ExactMatchGrader(normalize=True)
            >>> result = grader.grade("POSITIVE", "positive")
            >>> print(result.score)  # 1.0
            >>> print(result.passed)  # True
        """
        if self.normalize:
            expected = self._preprocess(expected)
            actual = self._preprocess(actual)

        passed = expected == actual
        score = 1.0 if passed else 0.0

        return GradeResult(
            score=score,
            passed=passed,
            explanation=f"Expected '{expected}', got '{actual}'",
            metadata={"match_type": "exact"}
        )


class StartsWithGrader(BaseGrader):
    """
    Check if completion starts with any reference answer.

    Book Reference: Chapter 3 - OpenAI Evals "Match" pattern:
    any(completion.startswith(ref) for ref in references)
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def grade(
        self, references: Union[str, List[str]], actual: str
    ) -> GradeResult:
        """
        Grade by prefix matching.

        Args:
            references: Single reference or list of valid prefixes
            actual: Model output to check

        Example:
            >>> grader = StartsWithGrader()
            >>> result = grader.grade(["Yes", "Correct"], "Yes, that is right")
            >>> print(result.passed)  # True
        """
        if isinstance(references, str):
            references = [references]

        if self.normalize:
            references = [r.strip().lower() for r in references]
            actual = actual.strip().lower()

        for ref in references:
            if actual.startswith(ref):
                return GradeResult(
                    score=1.0,
                    passed=True,
                    explanation=f"Output starts with '{ref}'",
                    metadata={"matched_reference": ref}
                )

        return GradeResult(
            score=0.0,
            passed=False,
            explanation=f"Output does not start with any reference",
            metadata={"references": references}
        )


class IncludesGrader(BaseGrader):
    """
    Check if any reference answer appears in the completion.

    Book Reference: Chapter 3 - OpenAI Evals "Includes" pattern:
    any(ref in completion for ref in references)
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def grade(
        self, references: Union[str, List[str]], actual: str
    ) -> GradeResult:
        """
        Grade by substring inclusion.

        Example:
            >>> grader = IncludesGrader()
            >>> result = grader.grade("Paris", "The capital of France is Paris.")
            >>> print(result.passed)  # True
        """
        if isinstance(references, str):
            references = [references]

        if self.normalize:
            references = [r.strip().lower() for r in references]
            actual = actual.strip().lower()

        for ref in references:
            if ref in actual:
                return GradeResult(
                    score=1.0,
                    passed=True,
                    explanation=f"Output includes '{ref}'",
                    metadata={"matched_reference": ref}
                )

        return GradeResult(
            score=0.0,
            passed=False,
            explanation="Output does not include any reference",
            metadata={"references": references}
        )


class FuzzyMatchGrader(BaseGrader):
    """
    Bidirectional substring matching.

    Book Reference: Chapter 3 - OpenAI Evals "FuzzyMatch" pattern:
    any((completion in ref or ref in completion) for ref in references)

    Useful when the model might give a more specific or more general
    answer than the reference.
    """

    def __init__(self, normalize: bool = True):
        self.normalize = normalize

    def grade(
        self, references: Union[str, List[str]], actual: str
    ) -> GradeResult:
        """
        Grade by bidirectional substring match.

        Example:
            >>> grader = FuzzyMatchGrader()
            >>> # Reference is substring of actual
            >>> result = grader.grade("Paris", "Paris, France")
            >>> print(result.passed)  # True
            >>> # Actual is substring of reference
            >>> result = grader.grade("Paris, the capital of France", "Paris")
            >>> print(result.passed)  # True
        """
        if isinstance(references, str):
            references = [references]

        if self.normalize:
            references = [r.strip().lower() for r in references]
            actual = actual.strip().lower()

        for ref in references:
            if actual in ref or ref in actual:
                return GradeResult(
                    score=1.0,
                    passed=True,
                    explanation=f"Fuzzy match with '{ref}'",
                    metadata={"matched_reference": ref}
                )

        return GradeResult(
            score=0.0,
            passed=False,
            explanation="No fuzzy match found",
            metadata={"references": references}
        )


class JsonMatchGrader(BaseGrader):
    """
    Compare JSON objects semantically (ignoring key order and whitespace).

    Book Reference: Chapter 3 - OpenAI Evals "JsonMatch" pattern:
    Useful for structured outputs where semantic equivalence matters
    more than exact string matching.
    """

    def __init__(self, ignore_extra_keys: bool = False):
        """
        Initialize JsonMatchGrader.

        Args:
            ignore_extra_keys: If True, only check that expected keys match
        """
        self.ignore_extra_keys = ignore_extra_keys

    def _parse_json(self, text: str) -> Optional[Any]:
        """Attempt to parse JSON from text."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(json_pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        return None

    def _compare_values(self, expected: Any, actual: Any) -> bool:
        """Recursively compare JSON values."""
        if type(expected) != type(actual):
            return False

        if isinstance(expected, dict):
            if self.ignore_extra_keys:
                # Only check that expected keys exist and match
                return all(
                    k in actual and self._compare_values(v, actual[k])
                    for k, v in expected.items()
                )
            else:
                # Exact key match
                if set(expected.keys()) != set(actual.keys()):
                    return False
                return all(
                    self._compare_values(expected[k], actual[k])
                    for k in expected
                )

        elif isinstance(expected, list):
            if len(expected) != len(actual):
                return False
            return all(
                self._compare_values(e, a) for e, a in zip(expected, actual)
            )

        else:
            return expected == actual

    def grade(self, expected: Union[str, dict], actual: str) -> GradeResult:
        """
        Grade by JSON semantic equivalence.

        Example:
            >>> grader = JsonMatchGrader()
            >>> expected = '{"name": "Alice", "age": 30}'
            >>> actual = '{"age": 30, "name": "Alice"}'  # Different order
            >>> result = grader.grade(expected, actual)
            >>> print(result.passed)  # True
        """
        # Parse expected if string
        if isinstance(expected, str):
            expected_obj = self._parse_json(expected)
            if expected_obj is None:
                return GradeResult(
                    score=0.0,
                    passed=False,
                    explanation="Could not parse expected as JSON",
                )
        else:
            expected_obj = expected

        # Parse actual
        actual_obj = self._parse_json(actual)
        if actual_obj is None:
            return GradeResult(
                score=0.0,
                passed=False,
                explanation="Could not parse actual output as JSON",
            )

        # Compare
        passed = self._compare_values(expected_obj, actual_obj)

        return GradeResult(
            score=1.0 if passed else 0.0,
            passed=passed,
            explanation="JSON structures match" if passed else "JSON structures differ",
            metadata={
                "expected": expected_obj,
                "actual": actual_obj,
            }
        )


class RegexGrader(BaseGrader):
    """
    Grade using regular expression patterns.

    Useful for checking output format or extracting specific information.
    """

    def __init__(self, flags: int = re.IGNORECASE):
        """
        Initialize RegexGrader.

        Args:
            flags: Regular expression flags (default: case-insensitive)
        """
        self.flags = flags

    def grade(self, pattern: str, actual: str) -> GradeResult:
        """
        Grade by regex pattern match.

        Example:
            >>> grader = RegexGrader()
            >>> # Check for email format
            >>> pattern = r'[\\w.]+@[\\w.]+'
            >>> result = grader.grade(pattern, "Contact: alice@example.com")
            >>> print(result.passed)  # True
        """
        match = re.search(pattern, actual, self.flags)

        if match:
            return GradeResult(
                score=1.0,
                passed=True,
                explanation=f"Pattern matched: '{match.group()}'",
                metadata={"match": match.group(), "span": match.span()}
            )
        else:
            return GradeResult(
                score=0.0,
                passed=False,
                explanation=f"Pattern '{pattern}' not found in output",
            )


# Demo usage
if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 3: Exact Match Graders Demo")
    print("=" * 60)

    # Example 1: Exact Match for Classification
    print("\n1. ExactMatchGrader (Classification)")
    print("-" * 40)
    grader = ExactMatchGrader(normalize=True)

    test_cases = [
        ("POSITIVE", "positive"),
        ("negative", "NEGATIVE"),
        ("neutral", "positive"),
    ]

    for expected, actual in test_cases:
        result = grader.grade(expected, actual)
        print(f"Expected: '{expected}' | Actual: '{actual}' | "
              f"Passed: {result.passed}")

    # Example 2: Includes Grader
    print("\n2. IncludesGrader (Answer Extraction)")
    print("-" * 40)
    grader = IncludesGrader()

    test_cases = [
        ("Paris", "The capital of France is Paris."),
        ("42", "The answer is 42."),
        ("Berlin", "The capital of France is Paris."),
    ]

    for reference, response in test_cases:
        result = grader.grade(reference, response)
        print(f"Looking for: '{reference}' | Passed: {result.passed}")

    # Example 3: JSON Match
    print("\n3. JsonMatchGrader (Structured Output)")
    print("-" * 40)
    grader = JsonMatchGrader()

    expected = {"name": "Alice", "age": 30, "city": "NYC"}
    actual_same = '{"age": 30, "city": "NYC", "name": "Alice"}'
    actual_diff = '{"name": "Alice", "age": 31, "city": "NYC"}'

    result = grader.grade(expected, actual_same)
    print(f"Same content, different order: Passed = {result.passed}")

    result = grader.grade(expected, actual_diff)
    print(f"Different age value: Passed = {result.passed}")

    # Example 4: Fuzzy Match
    print("\n4. FuzzyMatchGrader (Flexible Matching)")
    print("-" * 40)
    grader = FuzzyMatchGrader()

    test_cases = [
        ("Paris", "Paris, France"),  # ref in actual
        ("Paris, the capital of France", "Paris"),  # actual in ref
        ("Berlin", "Paris"),  # no match
    ]

    for reference, actual in test_cases:
        result = grader.grade(reference, actual)
        print(f"Ref: '{reference[:20]}...' | Actual: '{actual}' | "
              f"Passed: {result.passed}")

    print("\n" + "=" * 60)
    print("Key Takeaway: Use code-based graders when tasks have")
    print("deterministic correct answers. They're fastest and most reliable.")
    print("=" * 60)
