"""
Code-Based Graders
Book 6, Chapter 3.1

Code-based graders are the foundation layer of eval systems.
They use deterministic algorithms to check model outputs.

Anthropic ranks code-based grading first: "Fastest and most reliable,
extremely scalable."

Key advantages:
- Speed: Milliseconds per evaluation
- Cost: Near-zero marginal cost
- Determinism: Same input produces the same grade
- Scalability: Can run millions of evaluations without bottlenecks
"""

from .exact_match import ExactMatchGrader, FuzzyMatchGrader
from .regex_grader import RegexGrader
from .json_match import JsonMatchGrader

__all__ = [
    "ExactMatchGrader",
    "FuzzyMatchGrader",
    "RegexGrader",
    "JsonMatchGrader",
]
