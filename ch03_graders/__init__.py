"""
Chapter 3: Grader Taxonomy
==========================

Implements the three fundamental grader types:
1. Code-based graders (exact match, regex, JSON)
2. Model-based graders (LLM-as-judge)
3. Execution-based graders (code correctness)

Book Reference: Chapter 3 covers the grader hierarchy -
prefer code-based when possible, escalate to LLM-as-judge
for subjective quality, reserve human eval for calibration.
"""

__all__ = [
    "ExactMatchGrader",
    "FuzzyMatchGrader",
    "JsonMatchGrader",
    "SemanticSimilarityGrader",
    "LLMJudge",
    "RubricGrader",
]

_EXPORTS = {
    "ExactMatchGrader": ".exact_match",
    "FuzzyMatchGrader": ".exact_match",
    "JsonMatchGrader": ".exact_match",
    "SemanticSimilarityGrader": ".semantic_similarity",
    "LLMJudge": ".llm_as_judge",
    "RubricGrader": ".llm_as_judge",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
