"""
LLM-as-Judge Graders
Book 6, Chapter 3.2: Model-Based Graders

Use language models to evaluate outputs against criteria or rubrics.
Supports both reference-based and reference-free judging.
"""

import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    score: float  # 1-5 scale or 0-1 normalized
    passed: bool
    reasoning: str
    criteria_scores: Optional[Dict[str, float]] = None
    raw_response: Optional[str] = None


class LLMJudge:
    """
    LLM-as-Judge grader for complex evaluation tasks.

    Uses an LLM to evaluate responses against a rubric or criteria.
    Supports Anthropic Claude and OpenAI models.

    Example:
        >>> judge = LLMJudge(provider="anthropic", model="claude-3-sonnet-20240229")
        >>> result = judge.grade(
        ...     task="Answer helpfully",
        ...     user_query="What is Python?",
        ...     response="Python is a programming language.",
        ...     rubric=["accuracy", "completeness"]
        ... )
        >>> print(result.score, result.reasoning)
    """

    DEFAULT_PROMPT = '''You are evaluating the quality of an AI assistant's response.

## Task
{task}

## User Query
{user_query}

## Assistant Response
{response}

## Evaluation Criteria
{criteria}

## Instructions
1. Analyze the response against each criterion
2. Provide a brief explanation for your rating
3. Assign a score from 1-5 where:
   - 1 = Poor (fails to address the query)
   - 2 = Below Average (partially addresses, major issues)
   - 3 = Average (addresses query, some issues)
   - 4 = Good (addresses query well, minor issues)
   - 5 = Excellent (fully addresses query, no issues)

## Output Format
Respond with JSON only:
{{
    "reasoning": "Your brief explanation",
    "score": <1-5>,
    "criteria_scores": {{
        "<criterion_1>": <1-5>,
        "<criterion_2>": <1-5>
    }}
}}'''

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-3-sonnet-20240229",
        prompt_template: Optional[str] = None,
        pass_threshold: float = 3.0,
        api_key: Optional[str] = None
    ):
        """
        Initialize the LLM judge.

        Args:
            provider: "anthropic" or "openai"
            model: Model identifier
            prompt_template: Custom prompt template (uses DEFAULT_PROMPT if None)
            pass_threshold: Minimum score to pass (default 3.0 on 1-5 scale)
            api_key: Optional API key (uses environment variable if None)
        """
        self.provider = provider
        self.model = model
        self.prompt_template = prompt_template or self.DEFAULT_PROMPT
        self.pass_threshold = pass_threshold
        self.client = self._init_client(api_key)

    def _init_client(self, api_key: Optional[str]):
        """Initialize the appropriate API client."""
        if self.provider == "anthropic":
            try:
                import anthropic
                return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
        elif self.provider == "openai":
            try:
                import openai
                return openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def grade(
        self,
        task: str,
        user_query: str,
        response: str,
        rubric: Optional[List[str]] = None,
        reference: Optional[str] = None
    ) -> JudgeResult:
        """
        Evaluate a response using the LLM judge.

        Args:
            task: Description of what the assistant should do
            user_query: The user's original query
            response: The assistant's response to evaluate
            rubric: List of criteria to evaluate against
            reference: Optional reference/ideal answer for comparison

        Returns:
            JudgeResult with score, reasoning, and criteria scores
        """
        # Format criteria
        criteria = rubric or ["helpfulness", "accuracy", "clarity"]
        criteria_text = "\n".join(f"- {c}" for c in criteria)

        if reference:
            criteria_text += f"\n\n## Reference Answer (for comparison)\n{reference}"

        # Build prompt
        prompt = self.prompt_template.format(
            task=task,
            user_query=user_query,
            response=response,
            criteria=criteria_text
        )

        # Call LLM
        raw_response = self._call_llm(prompt)

        # Parse response
        return self._parse_response(raw_response)

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        if self.provider == "anthropic":
            message = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024
            )
            return response.choices[0].message.content
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _parse_response(self, raw_response: str) -> JudgeResult:
        """Parse the LLM's JSON response."""
        try:
            # Try to extract JSON from response
            json_start = raw_response.find("{")
            json_end = raw_response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = raw_response[json_start:json_end]
                data = json.loads(json_str)

                score = float(data.get("score", 3))
                return JudgeResult(
                    score=score,
                    passed=score >= self.pass_threshold,
                    reasoning=data.get("reasoning", ""),
                    criteria_scores=data.get("criteria_scores"),
                    raw_response=raw_response
                )
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Fallback: couldn't parse
        return JudgeResult(
            score=0.0,
            passed=False,
            reasoning="Failed to parse judge response",
            raw_response=raw_response
        )


class RubricGrader:
    """
    Rubric-based grader with structured criteria.

    From OpenAI Evals YAML rubric format.

    Example rubric format:
        {
            "fact": {
                "description": "Is the response factually accurate?",
                "choices": ["A: Yes", "B: Partially", "C: No"],
                "scores": {"A": 1.0, "B": 0.5, "C": 0.0}
            }
        }
    """

    def __init__(
        self,
        rubric: Dict[str, Dict[str, Any]],
        provider: str = "anthropic",
        model: str = "claude-3-sonnet-20240229"
    ):
        """
        Initialize with a structured rubric.

        Args:
            rubric: Dict mapping criterion names to their definitions
            provider: LLM provider
            model: Model to use
        """
        self.rubric = rubric
        self.judge = LLMJudge(provider=provider, model=model)

    def grade(self, user_query: str, response: str) -> JudgeResult:
        """Grade using the structured rubric."""
        # Build detailed rubric prompt
        criteria_parts = []
        for name, config in self.rubric.items():
            part = f"### {name}\n{config['description']}\n"
            part += "Choices:\n"
            for choice in config.get("choices", []):
                part += f"  {choice}\n"
            criteria_parts.append(part)

        criteria_text = "\n".join(criteria_parts)

        return self.judge.grade(
            task="Evaluate the response using the rubric",
            user_query=user_query,
            response=response,
            rubric=[criteria_text]
        )
