"""
Regex-Based Grader
Book 6, Chapter 3: Grader Taxonomy

Pattern matching grader using regular expressions.
Particularly useful for:
- Extracting structured answers (e.g., "Answer: X")
- Validating format compliance
- Flexible matching with capture groups
"""

import re
from typing import Optional, List, Pattern, Union
from dataclasses import dataclass


@dataclass
class RegexGradeResult:
    """Result of regex grading."""
    passed: bool
    score: float
    matched_text: Optional[str] = None
    groups: Optional[tuple] = None
    group_dict: Optional[dict] = None


class RegexGrader:
    """
    Grade outputs by regex pattern matching.

    Example from MATH evaluation (Chapter 19):
        Pattern: (?i)Answer\\s*:\\s*([^\\n]+)
        Extracts the answer after "Answer:" prefix.

    Example:
        >>> grader = RegexGrader(r"(?i)Answer:\\s*(\\d+)")
        >>> result = grader.grade("After calculation, Answer: 42")
        >>> result.passed
        True
        >>> result.groups
        ('42',)
    """

    def __init__(
        self,
        pattern: Union[str, Pattern],
        flags: int = re.IGNORECASE,
        must_match_fully: bool = False
    ):
        """
        Initialize the regex grader.

        Args:
            pattern: Regex pattern string or compiled pattern
            flags: Regex flags (default: IGNORECASE)
            must_match_fully: If True, pattern must match entire string
        """
        if isinstance(pattern, str):
            self.pattern = re.compile(pattern, flags)
        else:
            self.pattern = pattern
        self.must_match_fully = must_match_fully

    def grade(self, completion: str) -> RegexGradeResult:
        """
        Check if completion matches the regex pattern.

        Args:
            completion: Text to match against

        Returns:
            RegexGradeResult with match details
        """
        if self.must_match_fully:
            match = self.pattern.fullmatch(completion)
        else:
            match = self.pattern.search(completion)

        if match:
            return RegexGradeResult(
                passed=True,
                score=1.0,
                matched_text=match.group(0),
                groups=match.groups() if match.groups() else None,
                group_dict=match.groupdict() if match.groupdict() else None
            )
        else:
            return RegexGradeResult(
                passed=False,
                score=0.0
            )

    def extract(self, completion: str, group: int = 1) -> Optional[str]:
        """
        Extract a specific capture group from the completion.

        Args:
            completion: Text to extract from
            group: Capture group number (1-indexed)

        Returns:
            Extracted text or None if no match
        """
        match = self.pattern.search(completion)
        if match and len(match.groups()) >= group:
            return match.group(group)
        return None


class MultiPatternGrader:
    """
    Grade outputs against multiple regex patterns.

    Useful when there are several acceptable formats for an answer.

    Example:
        >>> grader = MultiPatternGrader([
        ...     r"(?i)the answer is (\\d+)",
        ...     r"(?i)answer:\\s*(\\d+)",
        ...     r"^(\\d+)$"
        ... ])
        >>> grader.grade("The answer is 42").passed
        True
        >>> grader.grade("Answer: 42").passed
        True
        >>> grader.grade("42").passed
        True
    """

    def __init__(self, patterns: List[str], flags: int = re.IGNORECASE):
        """
        Initialize with multiple patterns.

        Args:
            patterns: List of regex pattern strings
            flags: Regex flags applied to all patterns
        """
        self.graders = [RegexGrader(p, flags) for p in patterns]

    def grade(self, completion: str) -> RegexGradeResult:
        """
        Check if completion matches any of the patterns.

        Returns result from first matching pattern.
        """
        for grader in self.graders:
            result = grader.grade(completion)
            if result.passed:
                return result

        return RegexGradeResult(passed=False, score=0.0)


# Pre-built patterns for common use cases
MATH_ANSWER_PATTERN = RegexGrader(r"(?i)Answer\s*:\s*([^\n]+)")
NUMBER_PATTERN = RegexGrader(r"-?\d+\.?\d*")
YES_NO_PATTERN = RegexGrader(r"(?i)^(yes|no)$", must_match_fully=True)
JSON_PATTERN = RegexGrader(r"\{[^{}]*\}")
