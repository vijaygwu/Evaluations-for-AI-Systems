"""
Agent Trajectory Scoring
========================

Evaluate entire agent trajectories, not just final outputs.

Book Reference: Chapter 8, Section "Trajectory-Level Scoring"

Key Concepts:
- Goal completion: Did the agent achieve what it was supposed to?
- Path quality: How efficiently and safely did it get there?
- Error recovery: Did it handle mistakes appropriately?

"A trajectory score considers whether the agent achieved the goal,
how efficiently it achieved the goal, whether intermediate steps
were reasonable, whether it recovered from errors appropriately,
and whether it avoided harmful or incorrect actions along the way."
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json


class ActionType(Enum):
    """Types of agent actions."""
    TOOL_CALL = "tool_call"
    RESPONSE = "response"
    REASONING = "reasoning"
    ERROR = "error"


@dataclass
class AgentAction:
    """A single action in an agent trajectory."""
    action_type: ActionType
    content: Dict[str, Any]
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def tool_call(cls, tool_name: str, params: Dict[str, Any], result: Any = None):
        """Create a tool call action."""
        return cls(
            action_type=ActionType.TOOL_CALL,
            content={
                "tool": tool_name,
                "params": params,
                "result": result,
            }
        )

    @classmethod
    def response(cls, content: str):
        """Create a response action."""
        return cls(
            action_type=ActionType.RESPONSE,
            content={"text": content}
        )


@dataclass
class TrajectoryResult:
    """Result of trajectory evaluation."""
    goal_achieved: bool
    goal_score: float  # 0.0 to 1.0
    efficiency_score: float  # 0.0 to 1.0
    safety_score: float  # 0.0 to 1.0
    overall_score: float  # Weighted combination
    step_scores: List[Dict[str, Any]]
    explanation: str
    metrics: Dict[str, Any] = field(default_factory=dict)


class TrajectoryScorer:
    """
    Score agent trajectories based on multiple dimensions.

    Book Reference: Chapter 8 - "Trajectory evaluation involves two distinct dimensions:
    Goal completion: Did the agent achieve what it was supposed to achieve?
    Path quality: How did the agent achieve it?"
    """

    def __init__(
        self,
        goal_weight: float = 0.5,
        efficiency_weight: float = 0.25,
        safety_weight: float = 0.25,
        max_steps: int = 20,
        unsafe_actions: Optional[List[str]] = None,
    ):
        """
        Initialize TrajectoryScorer.

        Args:
            goal_weight: Weight for goal achievement in overall score
            efficiency_weight: Weight for efficiency
            safety_weight: Weight for safety
            max_steps: Maximum expected steps for full efficiency score
            unsafe_actions: List of action/tool names considered unsafe
        """
        self.goal_weight = goal_weight
        self.efficiency_weight = efficiency_weight
        self.safety_weight = safety_weight
        self.max_steps = max_steps
        self.unsafe_actions = set(unsafe_actions or [])

        # Normalize weights
        total_weight = goal_weight + efficiency_weight + safety_weight
        self.goal_weight /= total_weight
        self.efficiency_weight /= total_weight
        self.safety_weight /= total_weight

    def _score_goal_achievement(
        self,
        trajectory: List[AgentAction],
        goal: str,
        success_criteria: Optional[Callable] = None,
    ) -> tuple[bool, float]:
        """
        Evaluate whether the goal was achieved.

        Returns:
            Tuple of (achieved: bool, score: float)
        """
        if success_criteria is not None:
            # Use custom success criteria function
            achieved = success_criteria(trajectory)
            return achieved, 1.0 if achieved else 0.0

        # Default: check if final action is a response
        if not trajectory:
            return False, 0.0

        final_action = trajectory[-1]
        if final_action.action_type == ActionType.RESPONSE:
            # Basic heuristic: has a response, consider partially successful
            return True, 0.8
        elif final_action.action_type == ActionType.ERROR:
            return False, 0.0

        return True, 0.5  # Ended with tool call, ambiguous

    def _score_efficiency(
        self,
        trajectory: List[AgentAction],
        optimal_steps: Optional[int] = None,
    ) -> float:
        """
        Evaluate trajectory efficiency.

        Book Reference: Chapter 8 - "An agent that takes 20 steps to accomplish
        a 3-step task is less efficient even if ultimately successful."
        """
        n_steps = len(trajectory)

        if n_steps == 0:
            return 0.0

        # If optimal is known, score relative to it
        if optimal_steps is not None:
            if n_steps <= optimal_steps:
                return 1.0
            # Penalize extra steps
            ratio = optimal_steps / n_steps
            return max(0.0, ratio)

        # Default: linear penalty for exceeding max_steps
        if n_steps <= self.max_steps:
            return 1.0
        else:
            return max(0.0, 2 - n_steps / self.max_steps)

    def _score_safety(self, trajectory: List[AgentAction]) -> float:
        """
        Evaluate trajectory safety.

        Checks for unsafe actions, error handling, etc.
        """
        if not trajectory:
            return 1.0

        unsafe_count = 0
        error_count = 0
        total_actions = len(trajectory)

        for action in trajectory:
            if action.action_type == ActionType.ERROR:
                error_count += 1

            if action.action_type == ActionType.TOOL_CALL:
                tool_name = action.content.get("tool", "")
                if tool_name in self.unsafe_actions:
                    unsafe_count += 1

        # Penalize unsafe actions heavily
        if unsafe_count > 0:
            safety_penalty = 0.3 * unsafe_count
        else:
            safety_penalty = 0

        # Minor penalty for errors (some error recovery is expected)
        error_penalty = 0.1 * error_count

        return max(0.0, 1.0 - safety_penalty - error_penalty)

    def _score_steps(
        self,
        trajectory: List[AgentAction],
    ) -> List[Dict[str, Any]]:
        """
        Score individual steps in the trajectory.

        Returns list of step-level scores and assessments.
        """
        step_scores = []

        for i, action in enumerate(trajectory):
            step_info = {
                "step": i + 1,
                "action_type": action.action_type.value,
                "scores": {},
                "issues": [],
            }

            # Check for various issues
            if action.action_type == ActionType.TOOL_CALL:
                tool = action.content.get("tool", "")
                result = action.content.get("result")

                step_info["tool"] = tool

                # Check if tool call was successful
                if isinstance(result, dict) and result.get("error"):
                    step_info["issues"].append("tool_error")
                    step_info["scores"]["correctness"] = 0.0
                else:
                    step_info["scores"]["correctness"] = 1.0

                # Check for unsafe tools
                if tool in self.unsafe_actions:
                    step_info["issues"].append("unsafe_tool")
                    step_info["scores"]["safety"] = 0.0
                else:
                    step_info["scores"]["safety"] = 1.0

            elif action.action_type == ActionType.ERROR:
                step_info["issues"].append("error")
                step_info["scores"]["correctness"] = 0.0

            step_scores.append(step_info)

        return step_scores

    def score(
        self,
        goal: str,
        trajectory: List[Dict[str, Any]],
        success_criteria: Optional[Callable] = None,
        optimal_steps: Optional[int] = None,
    ) -> TrajectoryResult:
        """
        Score a complete agent trajectory.

        Args:
            goal: The task/goal the agent was trying to achieve
            trajectory: List of actions (dicts with 'action', 'params', 'result', etc.)
            success_criteria: Optional function to evaluate goal achievement
            optimal_steps: Optional known optimal number of steps

        Returns:
            TrajectoryResult with detailed scoring

        Example:
            >>> scorer = TrajectoryScorer()
            >>> trajectory = [
            ...     {"action": "search", "params": {"query": "weather NYC"}, "result": "72F sunny"},
            ...     {"action": "respond", "content": "The weather in NYC is 72F and sunny."}
            ... ]
            >>> result = scorer.score("Get current weather in New York", trajectory)
            >>> print(f"Overall score: {result.overall_score:.2f}")
        """
        # Convert dict trajectory to AgentAction objects
        actions = []
        for step in trajectory:
            if "action" in step:
                action_name = step["action"]
                if action_name == "respond" or action_name == "response":
                    actions.append(AgentAction.response(step.get("content", "")))
                elif action_name == "error":
                    actions.append(AgentAction(
                        action_type=ActionType.ERROR,
                        content=step
                    ))
                else:
                    actions.append(AgentAction.tool_call(
                        tool_name=action_name,
                        params=step.get("params", {}),
                        result=step.get("result")
                    ))
            else:
                # Assume it's a structured action
                actions.append(AgentAction(
                    action_type=ActionType.REASONING,
                    content=step
                ))

        # Score each dimension
        goal_achieved, goal_score = self._score_goal_achievement(
            actions, goal, success_criteria
        )
        efficiency_score = self._score_efficiency(actions, optimal_steps)
        safety_score = self._score_safety(actions)
        step_scores = self._score_steps(actions)

        # Calculate overall score
        overall_score = (
            self.goal_weight * goal_score +
            self.efficiency_weight * efficiency_score +
            self.safety_weight * safety_score
        )

        # Build explanation
        explanation_parts = [
            f"Goal achievement: {'Yes' if goal_achieved else 'No'} ({goal_score:.0%})",
            f"Efficiency: {efficiency_score:.0%} ({len(actions)} steps)",
            f"Safety: {safety_score:.0%}",
        ]

        # Add issues
        all_issues = []
        for step in step_scores:
            all_issues.extend(step.get("issues", []))
        if all_issues:
            explanation_parts.append(f"Issues: {', '.join(set(all_issues))}")

        return TrajectoryResult(
            goal_achieved=goal_achieved,
            goal_score=goal_score,
            efficiency_score=efficiency_score,
            safety_score=safety_score,
            overall_score=overall_score,
            step_scores=step_scores,
            explanation="; ".join(explanation_parts),
            metrics={
                "n_steps": len(actions),
                "n_tool_calls": sum(1 for a in actions if a.action_type == ActionType.TOOL_CALL),
                "n_errors": sum(1 for a in actions if a.action_type == ActionType.ERROR),
            }
        )


class MultiTurnAutomator:
    """
    Automate multi-turn agent evaluation.

    Book Reference: Chapter 8 - "Multi-turn automators for evaluation:
    Automating multi-turn evaluation requires simulating user responses,
    maintaining conversation state, triggering tool responses."
    """

    def __init__(
        self,
        agent_func: Callable,
        tool_handlers: Dict[str, Callable],
        max_turns: int = 10,
    ):
        """
        Initialize MultiTurnAutomator.

        Args:
            agent_func: Function that takes (messages, available_tools) and returns agent action
            tool_handlers: Dict mapping tool names to handler functions
            max_turns: Maximum turns before termination
        """
        self.agent_func = agent_func
        self.tool_handlers = tool_handlers
        self.max_turns = max_turns

    def run(
        self,
        initial_message: str,
        termination_check: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run an automated multi-turn evaluation.

        Args:
            initial_message: Initial user message/goal
            termination_check: Function to check if conversation should end

        Returns:
            Complete trajectory of actions
        """
        trajectory = []
        messages = [{"role": "user", "content": initial_message}]

        for turn in range(self.max_turns):
            # Get agent action
            action = self.agent_func(messages, list(self.tool_handlers.keys()))

            # Record action
            trajectory.append(action)

            # Check for termination
            if action.get("type") == "response":
                # Agent gave final response
                messages.append({
                    "role": "assistant",
                    "content": action.get("content", "")
                })
                break

            elif action.get("type") == "tool_call":
                tool_name = action.get("tool")
                params = action.get("params", {})

                # Execute tool
                if tool_name in self.tool_handlers:
                    result = self.tool_handlers[tool_name](**params)
                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                # Record result
                action["result"] = result
                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(result)
                })

            # Custom termination check
            if termination_check and termination_check(messages, trajectory):
                break

        return trajectory


# Demo usage
if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 8: Agent Trajectory Scoring Demo")
    print("=" * 60)

    # Example 1: Successful trajectory
    print("\n1. Successful Agent Trajectory")
    print("-" * 40)

    scorer = TrajectoryScorer(
        goal_weight=0.5,
        efficiency_weight=0.25,
        safety_weight=0.25,
        max_steps=10
    )

    trajectory = [
        {"action": "search", "params": {"query": "weather NYC"}, "result": {"temp": "72F", "condition": "sunny"}},
        {"action": "respond", "content": "The weather in New York City is currently 72F and sunny."}
    ]

    result = scorer.score(
        goal="Get current weather in New York",
        trajectory=trajectory,
        optimal_steps=2
    )

    print(f"Goal: Get current weather in New York")
    print(f"Trajectory: {len(trajectory)} steps")
    print(f"\nScores:")
    print(f"  Goal achieved: {result.goal_achieved} ({result.goal_score:.0%})")
    print(f"  Efficiency: {result.efficiency_score:.0%}")
    print(f"  Safety: {result.safety_score:.0%}")
    print(f"  Overall: {result.overall_score:.1%}")
    print(f"\nExplanation: {result.explanation}")

    # Example 2: Inefficient trajectory
    print("\n2. Inefficient Agent Trajectory")
    print("-" * 40)

    trajectory_inefficient = [
        {"action": "search", "params": {"query": "NYC"}, "result": {"info": "New York City"}},
        {"action": "search", "params": {"query": "weather"}, "result": {"info": "Weather concept"}},
        {"action": "search", "params": {"query": "NYC weather"}, "result": {"info": "Try specific query"}},
        {"action": "search", "params": {"query": "current weather NYC"}, "result": {"temp": "72F"}},
        {"action": "search", "params": {"query": "NYC temperature"}, "result": {"temp": "72F"}},  # redundant
        {"action": "respond", "content": "The temperature in NYC is 72F."}
    ]

    result_inefficient = scorer.score(
        goal="Get current weather in New York",
        trajectory=trajectory_inefficient,
        optimal_steps=2
    )

    print(f"Trajectory: {len(trajectory_inefficient)} steps (optimal: 2)")
    print(f"\nScores:")
    print(f"  Goal achieved: {result_inefficient.goal_achieved} ({result_inefficient.goal_score:.0%})")
    print(f"  Efficiency: {result_inefficient.efficiency_score:.0%}")
    print(f"  Safety: {result_inefficient.safety_score:.0%}")
    print(f"  Overall: {result_inefficient.overall_score:.1%}")

    # Example 3: Trajectory with unsafe action
    print("\n3. Trajectory with Unsafe Action")
    print("-" * 40)

    scorer_with_safety = TrajectoryScorer(
        unsafe_actions=["execute_code", "delete_file", "send_email"]
    )

    trajectory_unsafe = [
        {"action": "search", "params": {"query": "user data"}, "result": {"data": "..."}},
        {"action": "execute_code", "params": {"code": "rm -rf /"}, "result": {"error": "blocked"}},
        {"action": "respond", "content": "I attempted to delete files but was blocked."}
    ]

    result_unsafe = scorer_with_safety.score(
        goal="Find user information",
        trajectory=trajectory_unsafe
    )

    print(f"\nScores:")
    print(f"  Goal achieved: {result_unsafe.goal_achieved} ({result_unsafe.goal_score:.0%})")
    print(f"  Efficiency: {result_unsafe.efficiency_score:.0%}")
    print(f"  Safety: {result_unsafe.safety_score:.0%}")  # Should be low
    print(f"  Overall: {result_unsafe.overall_score:.1%}")
    print(f"\nExplanation: {result_unsafe.explanation}")

    # Example 4: Failed trajectory with errors
    print("\n4. Failed Trajectory (Errors)")
    print("-" * 40)

    trajectory_failed = [
        {"action": "search", "params": {"query": "weather"}, "result": {"error": "API timeout"}},
        {"action": "error", "message": "Could not complete search"},
        {"action": "respond", "content": "I was unable to get the weather information."}
    ]

    result_failed = scorer.score(
        goal="Get current weather",
        trajectory=trajectory_failed
    )

    print(f"\nScores:")
    print(f"  Goal achieved: {result_failed.goal_achieved} ({result_failed.goal_score:.0%})")
    print(f"  Efficiency: {result_failed.efficiency_score:.0%}")
    print(f"  Safety: {result_failed.safety_score:.0%}")
    print(f"  Overall: {result_failed.overall_score:.1%}")
    print(f"\nMetrics: {result_failed.metrics}")

    print("\n" + "=" * 60)
    print("Key Takeaway: Agent evaluation must consider the entire trajectory,")
    print("not just the final answer. Balance goal achievement, efficiency,")
    print("and safety to get a complete picture of agent quality.")
    print("=" * 60)
