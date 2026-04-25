"""
Chapter 8: Agent Evaluation Fundamentals
========================================

Agent evaluation extends basic eval concepts to multi-step interactions:
- Trajectory scoring (not just final answer)
- Tool use verification
- State management across turns
- Goal completion vs path quality

Book Reference: Chapter 8 covers why agent evals are hard -
"Agents take actions, use tools, maintain state across turns, and
pursue goals over extended interactions."
"""

from .trajectory_scorer import TrajectoryScorer, TrajectoryResult
from .tool_use_eval import ToolUseEvaluator, ToolCallResult

__all__ = [
    "TrajectoryScorer",
    "TrajectoryResult",
    "ToolUseEvaluator",
    "ToolCallResult",
]
