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

__all__ = [
    "TrajectoryScorer",
    "TrajectoryResult",
    "ToolUseEvaluator",
    "ToolCallResult",
]

_EXPORTS = {
    "TrajectoryScorer": ".trajectory_scorer",
    "TrajectoryResult": ".trajectory_scorer",
    "ToolUseEvaluator": ".tool_use_eval",
    "ToolCallResult": ".tool_use_eval",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
