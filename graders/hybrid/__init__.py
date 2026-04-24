"""
Hybrid Graders
Book 6, Chapter 3.4

Cascade and consensus approaches that combine multiple grader types.

Pattern: code → model → human escalation
- Start with fast, cheap code-based grading
- Escalate uncertain cases to LLM judge
- Reserve human evaluation for highest-stakes decisions
"""

from .cascade_grader import CascadeGrader

__all__ = ["CascadeGrader"]
