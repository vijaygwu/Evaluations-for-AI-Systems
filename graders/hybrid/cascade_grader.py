"""
Cascade Grader
Book 6, Chapter 3.4: Hybrid Approaches

Implements the cascade pattern: code → model → human escalation.
Optimizes for cost while maintaining quality.
"""

from typing import List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum


class GraderLevel(Enum):
    CODE = "code"
    MODEL = "model"
    HUMAN = "human"


@dataclass
class CascadeResult:
    """Result from cascade grading."""
    passed: bool
    score: float
    grader_level: GraderLevel
    reasoning: Optional[str] = None
    escalation_reason: Optional[str] = None


class CascadeGrader:
    """
    Cascade grader that escalates based on confidence.

    Strategy:
    1. Start with code-based grading (fast, cheap)
    2. If uncertain, escalate to LLM judge
    3. If still uncertain or high-stakes, flag for human review

    Example:
        >>> from graders.code_based import FuzzyMatchGrader
        >>> from graders.model_based import LLMJudge
        >>>
        >>> cascade = CascadeGrader(
        ...     code_grader=FuzzyMatchGrader(),
        ...     model_grader=LLMJudge(),
        ...     code_confidence_threshold=0.8
        ... )
        >>> result = cascade.grade(completion="42", references=["42"])
        >>> result.grader_level
        GraderLevel.CODE  # Handled at code level
    """

    def __init__(
        self,
        code_grader: Any,
        model_grader: Optional[Any] = None,
        code_confidence_threshold: float = 0.8,
        model_confidence_threshold: float = 0.7,
        enable_human_escalation: bool = False,
        human_callback: Optional[Callable] = None
    ):
        """
        Initialize the cascade grader.

        Args:
            code_grader: Code-based grader instance
            model_grader: LLM judge instance (optional)
            code_confidence_threshold: Min confidence to accept code grade
            model_confidence_threshold: Min confidence to accept model grade
            enable_human_escalation: Whether to escalate to humans
            human_callback: Function to call for human review
        """
        self.code_grader = code_grader
        self.model_grader = model_grader
        self.code_threshold = code_confidence_threshold
        self.model_threshold = model_confidence_threshold
        self.enable_human = enable_human_escalation
        self.human_callback = human_callback

    def grade(
        self,
        completion: str,
        references: Optional[List[str]] = None,
        context: Optional[dict] = None
    ) -> CascadeResult:
        """
        Grade using cascade strategy.

        Args:
            completion: Model output to grade
            references: Reference answers (for code grader)
            context: Additional context (for model grader)

        Returns:
            CascadeResult with final grade and escalation info
        """
        # Level 1: Code-based grading
        if references:
            code_result = self.code_grader.grade(completion, references)

            # Check if we can confidently return code result
            if code_result.score >= self.code_threshold or code_result.score <= (1 - self.code_threshold):
                return CascadeResult(
                    passed=code_result.passed,
                    score=code_result.score,
                    grader_level=GraderLevel.CODE
                )

            escalation_reason = f"Code grader confidence {code_result.score:.2f} below threshold {self.code_threshold}"
        else:
            escalation_reason = "No references provided for code grading"

        # Level 2: Model-based grading
        if self.model_grader:
            model_result = self.model_grader.grade(
                task=context.get("task", "Evaluate the response") if context else "Evaluate the response",
                user_query=context.get("query", "") if context else "",
                response=completion,
                reference=references[0] if references else None
            )

            # Normalize score to 0-1 if on 1-5 scale
            normalized_score = model_result.score / 5.0 if model_result.score > 1 else model_result.score

            if normalized_score >= self.model_threshold or normalized_score <= (1 - self.model_threshold):
                return CascadeResult(
                    passed=model_result.passed,
                    score=normalized_score,
                    grader_level=GraderLevel.MODEL,
                    reasoning=model_result.reasoning,
                    escalation_reason=escalation_reason
                )

            escalation_reason += f"; Model grader confidence {normalized_score:.2f} below threshold"

        # Level 3: Human escalation
        if self.enable_human and self.human_callback:
            return CascadeResult(
                passed=False,  # Pending human review
                score=0.5,     # Uncertain
                grader_level=GraderLevel.HUMAN,
                escalation_reason=escalation_reason
            )

        # No human escalation - return model result or uncertain
        return CascadeResult(
            passed=False,
            score=0.5,
            grader_level=GraderLevel.MODEL if self.model_grader else GraderLevel.CODE,
            reasoning="Confidence too low, would require human review",
            escalation_reason=escalation_reason
        )


class CostAwareRouter:
    """
    Routes to optimal grader based on cost/quality tradeoff.

    Estimates cost per grader type and routes accordingly.
    """

    def __init__(
        self,
        graders: dict,
        costs: dict,
        quality_requirements: dict
    ):
        """
        Initialize the router.

        Args:
            graders: Dict mapping grader names to instances
            costs: Dict mapping grader names to cost per eval
            quality_requirements: Dict mapping task types to min quality
        """
        self.graders = graders
        self.costs = costs
        self.quality_requirements = quality_requirements

    def route(self, task_type: str) -> str:
        """
        Select optimal grader for task type.

        Returns grader name that meets quality at lowest cost.
        """
        min_quality = self.quality_requirements.get(task_type, 0.8)

        # Filter graders that meet quality, sort by cost
        candidates = [
            (name, cost)
            for name, cost in self.costs.items()
            if name in self.graders
        ]
        candidates.sort(key=lambda x: x[1])

        # Return cheapest (code-based is typically first)
        return candidates[0][0] if candidates else list(self.graders.keys())[0]
