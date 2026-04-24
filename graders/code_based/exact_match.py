"""
Exact Match and Fuzzy Match Graders
Book 6, Chapter 3: Grader Taxonomy

Basic code-based graders that check for string matches.
Based on OpenAI Evals framework patterns.
"""

from typing import List, Optional
from dataclasses import dataclass


@dataclass
class GradeResult:
    """Result of a grading operation."""
    passed: bool
    score: float
    matched_reference: Optional[str] = None
    details: Optional[dict] = None


class ExactMatchGrader:
    """
    Grade outputs by exact string matching against references.

    From OpenAI Evals - Match pattern:
        any([completion.startswith(ref) for ref in references])

    This implementation supports full match, prefix match, and configurable
    normalization options.

    Example:
        >>> grader = ExactMatchGrader()
        >>> result = grader.grade("42", ["42", "forty-two"])
        >>> result.passed
        True
    """

    def __init__(
        self,
        case_sensitive: bool = True,
        strip_whitespace: bool = True,
        match_mode: str = "full"  # "full", "prefix", "contains"
    ):
        """
        Initialize the grader.

        Args:
            case_sensitive: Whether comparison is case-sensitive
            strip_whitespace: Whether to strip leading/trailing whitespace
            match_mode: "full" for exact match, "prefix" for startswith,
                       "contains" for substring match
        """
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace
        self.match_mode = match_mode

    def grade(self, completion: str, references: List[str]) -> GradeResult:
        """
        Grade a completion against reference answers.

        Args:
            completion: Model output to grade
            references: List of acceptable answers

        Returns:
            GradeResult with pass/fail, score, and matched reference
        """
        completion_normalized = self._normalize(completion)

        for ref in references:
            ref_normalized = self._normalize(ref)

            if self._matches(completion_normalized, ref_normalized):
                return GradeResult(
                    passed=True,
                    score=1.0,
                    matched_reference=ref
                )

        return GradeResult(
            passed=False,
            score=0.0,
            matched_reference=None
        )

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if self.strip_whitespace:
            text = text.strip()
        if not self.case_sensitive:
            text = text.lower()
        return text

    def _matches(self, completion: str, reference: str) -> bool:
        """Check if completion matches reference based on mode."""
        if self.match_mode == "full":
            return completion == reference
        elif self.match_mode == "prefix":
            return completion.startswith(reference)
        elif self.match_mode == "contains":
            return reference in completion
        else:
            raise ValueError(f"Unknown match_mode: {self.match_mode}")


class FuzzyMatchGrader:
    """
    Grade outputs by bidirectional substring matching.

    From OpenAI Evals - FuzzyMatch pattern:
        any([(completion in ref or ref in completion) for ref in references])

    This is more lenient than exact match - either the completion
    contains a reference OR a reference contains the completion.

    Example:
        >>> grader = FuzzyMatchGrader()
        >>> result = grader.grade("The answer is 42", ["42"])
        >>> result.passed  # True - "42" is in completion
        True
    """

    def __init__(self, case_sensitive: bool = False, strip_whitespace: bool = True):
        """
        Initialize the grader.

        Args:
            case_sensitive: Whether comparison is case-sensitive (default False)
            strip_whitespace: Whether to strip whitespace
        """
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace

    def grade(self, completion: str, references: List[str]) -> GradeResult:
        """
        Check if completion contains reference or vice versa.

        Args:
            completion: Model output to grade
            references: List of acceptable answers

        Returns:
            GradeResult with pass/fail and score
        """
        completion_normalized = self._normalize(completion)

        for ref in references:
            ref_normalized = self._normalize(ref)

            # Bidirectional check: either contains the other
            if completion_normalized in ref_normalized or ref_normalized in completion_normalized:
                return GradeResult(
                    passed=True,
                    score=1.0,
                    matched_reference=ref,
                    details={"match_type": "fuzzy"}
                )

        return GradeResult(
            passed=False,
            score=0.0,
            matched_reference=None
        )

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if self.strip_whitespace:
            text = text.strip()
        if not self.case_sensitive:
            text = text.lower()
        return text


class IncludesGrader:
    """
    Grade outputs by checking if any reference appears in completion.

    From OpenAI Evals - Includes pattern:
        any([ref in completion for ref in references])

    One-directional: only checks if reference is in completion,
    not the other way around.

    Example:
        >>> grader = IncludesGrader()
        >>> result = grader.grade("Paris is the capital of France", ["Paris"])
        >>> result.passed
        True
    """

    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

    def grade(self, completion: str, references: List[str]) -> GradeResult:
        """Check if any reference appears in the completion."""
        completion_check = completion if self.case_sensitive else completion.lower()

        for ref in references:
            ref_check = ref if self.case_sensitive else ref.lower()
            if ref_check in completion_check:
                return GradeResult(
                    passed=True,
                    score=1.0,
                    matched_reference=ref
                )

        return GradeResult(passed=False, score=0.0)
