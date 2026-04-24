"""
Model-Based Graders (LLM-as-Judge)
Book 6, Chapter 3.2

Use LLMs to evaluate other LLM outputs against criteria.
Anthropic positions this as: "Fast and flexible, scalable and suitable
for complex judgment. Test to ensure reliability first then scale."

Key considerations:
- Position bias: LLM judges prefer responses based on presentation order
- Verbosity bias: Longer responses often get higher scores
- Self-enhancement bias: Models prefer outputs similar to their own
"""

from .llm_judge import LLMJudge, RubricGrader

__all__ = ["LLMJudge", "RubricGrader"]
