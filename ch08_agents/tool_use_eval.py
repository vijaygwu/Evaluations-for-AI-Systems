"""
Tool Use Evaluation
===================

Evaluate how agents use tools: selection, parameters, result interpretation.

Book Reference: Chapter 8, Section "Tool Use Evaluation"

Key Aspects:
- Tool Selection: Did the agent choose the right tool?
- Parameter Correctness: Were the parameters correct?
- Result Interpretation: Did the agent correctly interpret the result?
- Error Handling: Did the agent handle errors appropriately?
"""

from typing import List, Dict, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from enum import Enum


class ToolUseError(Enum):
    """Types of tool use errors."""
    WRONG_TOOL = "wrong_tool"
    UNNECESSARY_TOOL = "unnecessary_tool"
    MISSING_TOOL = "missing_tool"
    WRONG_PARAMS = "wrong_params"
    MISSING_PARAMS = "missing_params"
    TYPE_ERROR = "type_error"
    MISINTERPRETATION = "misinterpretation"
    IGNORED_ERROR = "ignored_error"
    IGNORED_RESULT = "ignored_result"


@dataclass
class ToolCall:
    """Represents a single tool call."""
    tool_name: str
    params: Dict[str, Any]
    result: Any = None
    error: Optional[str] = None
    expected_tool: Optional[str] = None
    expected_params: Optional[Dict[str, Any]] = None


@dataclass
class ToolCallResult:
    """Result of evaluating a single tool call."""
    tool_call: ToolCall
    selection_correct: bool
    params_correct: bool
    interpretation_correct: Optional[bool] = None
    error_handled: Optional[bool] = None
    errors: List[ToolUseError] = field(default_factory=list)
    score: float = 0.0
    explanation: str = ""


@dataclass
class ToolUseResult:
    """Result of evaluating all tool use in a trajectory."""
    total_calls: int
    correct_calls: int
    selection_accuracy: float
    param_accuracy: float
    interpretation_accuracy: float
    error_handling_rate: float
    overall_score: float
    call_results: List[ToolCallResult]
    errors_by_type: Dict[str, int]
    explanation: str


class ToolUseEvaluator:
    """
    Evaluate agent tool use across multiple dimensions.

    Book Reference: Chapter 8 - "Evaluating tool use requires examining
    multiple aspects of each tool interaction."

    Dimensions:
    1. Tool Selection: Did the agent choose the right tool for the task?
    2. Parameter Correctness: Did the agent pass correct parameters?
    3. Result Interpretation: Did the agent correctly interpret results?
    4. Error Handling: Did the agent handle errors appropriately?
    """

    def __init__(
        self,
        tool_schema: Optional[Dict[str, Dict[str, Any]]] = None,
        selection_weight: float = 0.3,
        params_weight: float = 0.3,
        interpretation_weight: float = 0.2,
        error_handling_weight: float = 0.2,
    ):
        """
        Initialize ToolUseEvaluator.

        Args:
            tool_schema: Optional schema for available tools (for validation)
            selection_weight: Weight for tool selection in overall score
            params_weight: Weight for parameter correctness
            interpretation_weight: Weight for result interpretation
            error_handling_weight: Weight for error handling
        """
        self.tool_schema = tool_schema or {}

        # Normalize weights
        total = selection_weight + params_weight + interpretation_weight + error_handling_weight
        self.selection_weight = selection_weight / total
        self.params_weight = params_weight / total
        self.interpretation_weight = interpretation_weight / total
        self.error_handling_weight = error_handling_weight / total

    def evaluate_selection(
        self,
        actual_tool: str,
        expected_tool: Optional[str],
        context: Optional[str] = None,
    ) -> tuple[bool, List[ToolUseError]]:
        """
        Evaluate whether the correct tool was selected.

        Book Reference: Chapter 8 - "Tool selection errors include using the wrong tool,
        not using any tool when one was required, using tools unnecessarily."
        """
        errors = []

        if expected_tool is None:
            # No specific tool expected, assume correct
            return True, errors

        if actual_tool == expected_tool:
            return True, errors

        # Wrong tool selected
        errors.append(ToolUseError.WRONG_TOOL)
        return False, errors

    def evaluate_params(
        self,
        actual_params: Dict[str, Any],
        expected_params: Optional[Dict[str, Any]],
        tool_name: Optional[str] = None,
    ) -> tuple[bool, List[ToolUseError]]:
        """
        Evaluate parameter correctness.

        Book Reference: Chapter 8 - "Parameter errors include wrong values,
        wrong types, missing required parameters, malformed parameter structures."
        """
        errors = []

        if expected_params is None:
            # No specific params expected
            # Could validate against schema if available
            if tool_name and tool_name in self.tool_schema:
                schema = self.tool_schema[tool_name]
                required = schema.get("required", [])
                for param in required:
                    if param not in actual_params:
                        errors.append(ToolUseError.MISSING_PARAMS)
                        return False, errors
            return True, errors

        # Check for missing params
        for key in expected_params:
            if key not in actual_params:
                errors.append(ToolUseError.MISSING_PARAMS)
                return False, errors

        # Check for wrong values
        for key, expected_value in expected_params.items():
            actual_value = actual_params.get(key)

            # Type check
            if type(actual_value) != type(expected_value):
                errors.append(ToolUseError.TYPE_ERROR)
                return False, errors

            # Value check (allowing for some flexibility in strings)
            if isinstance(expected_value, str):
                if expected_value.lower() not in str(actual_value).lower():
                    errors.append(ToolUseError.WRONG_PARAMS)
                    return False, errors
            elif actual_value != expected_value:
                errors.append(ToolUseError.WRONG_PARAMS)
                return False, errors

        return True, errors

    def evaluate_interpretation(
        self,
        tool_result: Any,
        agent_interpretation: Optional[str],
        expected_extraction: Optional[Any] = None,
    ) -> tuple[bool, List[ToolUseError]]:
        """
        Evaluate whether the agent correctly interpreted tool results.

        Book Reference: Chapter 8 - "Interpretation errors include ignoring
        relevant results, misreading returned data, failing to handle errors."
        """
        errors = []

        if agent_interpretation is None:
            # Agent didn't provide interpretation
            errors.append(ToolUseError.IGNORED_RESULT)
            return False, errors

        if expected_extraction is not None:
            # Check if expected info is in interpretation
            if isinstance(expected_extraction, str):
                if expected_extraction.lower() not in agent_interpretation.lower():
                    errors.append(ToolUseError.MISINTERPRETATION)
                    return False, errors

        return True, errors

    def evaluate_error_handling(
        self,
        tool_error: Optional[str],
        agent_response: Optional[str],
    ) -> tuple[bool, List[ToolUseError]]:
        """
        Evaluate whether errors were handled appropriately.
        """
        errors = []

        if tool_error is None:
            # No error to handle
            return True, errors

        if agent_response is None:
            errors.append(ToolUseError.IGNORED_ERROR)
            return False, errors

        # Check if agent acknowledged the error
        error_keywords = ["error", "failed", "couldn't", "unable", "problem", "issue"]
        response_lower = agent_response.lower()

        if not any(keyword in response_lower for keyword in error_keywords):
            errors.append(ToolUseError.IGNORED_ERROR)
            return False, errors

        return True, errors

    def evaluate_call(
        self,
        tool_call: ToolCall,
        agent_interpretation: Optional[str] = None,
    ) -> ToolCallResult:
        """
        Evaluate a single tool call.

        Args:
            tool_call: The tool call to evaluate
            agent_interpretation: How the agent interpreted/used the result

        Returns:
            ToolCallResult with detailed evaluation
        """
        all_errors = []

        # Evaluate selection
        selection_correct, selection_errors = self.evaluate_selection(
            tool_call.tool_name,
            tool_call.expected_tool
        )
        all_errors.extend(selection_errors)

        # Evaluate params
        params_correct, params_errors = self.evaluate_params(
            tool_call.params,
            tool_call.expected_params,
            tool_call.tool_name
        )
        all_errors.extend(params_errors)

        # Evaluate interpretation (if result available)
        interpretation_correct = None
        if tool_call.result is not None and agent_interpretation is not None:
            interpretation_correct, interp_errors = self.evaluate_interpretation(
                tool_call.result,
                agent_interpretation
            )
            all_errors.extend(interp_errors)

        # Evaluate error handling (if error occurred)
        error_handled = None
        if tool_call.error is not None:
            error_handled, error_errors = self.evaluate_error_handling(
                tool_call.error,
                agent_interpretation
            )
            all_errors.extend(error_errors)

        # Calculate score
        scores = []
        if selection_correct is not None:
            scores.append(1.0 if selection_correct else 0.0)
        if params_correct is not None:
            scores.append(1.0 if params_correct else 0.0)
        if interpretation_correct is not None:
            scores.append(1.0 if interpretation_correct else 0.0)
        if error_handled is not None:
            scores.append(1.0 if error_handled else 0.0)

        score = sum(scores) / len(scores) if scores else 0.0

        # Build explanation
        explanation_parts = []
        if not selection_correct:
            explanation_parts.append(f"Wrong tool (expected {tool_call.expected_tool})")
        if not params_correct:
            explanation_parts.append("Incorrect parameters")
        if interpretation_correct is False:
            explanation_parts.append("Misinterpreted result")
        if error_handled is False:
            explanation_parts.append("Did not handle error")

        explanation = "; ".join(explanation_parts) if explanation_parts else "Correct"

        return ToolCallResult(
            tool_call=tool_call,
            selection_correct=selection_correct,
            params_correct=params_correct,
            interpretation_correct=interpretation_correct,
            error_handled=error_handled,
            errors=all_errors,
            score=score,
            explanation=explanation
        )

    def evaluate_trajectory(
        self,
        tool_calls: List[Dict[str, Any]],
        agent_interpretations: Optional[List[str]] = None,
        expected_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolUseResult:
        """
        Evaluate all tool use in a trajectory.

        Args:
            tool_calls: List of tool calls made by agent
            agent_interpretations: Optional list of how agent interpreted each result
            expected_calls: Optional list of expected tool calls for comparison

        Returns:
            ToolUseResult with aggregate metrics

        Example:
            >>> evaluator = ToolUseEvaluator()
            >>> calls = [
            ...     {"tool": "search", "params": {"query": "weather NYC"}, "result": {"temp": 72}},
            ...     {"tool": "calculate", "params": {"expr": "72 * 9/5 + 32"}, "result": 161.6}
            ... ]
            >>> result = evaluator.evaluate_trajectory(calls)
        """
        if agent_interpretations is None:
            agent_interpretations = [None] * len(tool_calls)

        if expected_calls is None:
            expected_calls = [{}] * len(tool_calls)

        # Ensure lists are same length
        while len(agent_interpretations) < len(tool_calls):
            agent_interpretations.append(None)
        while len(expected_calls) < len(tool_calls):
            expected_calls.append({})

        # Evaluate each call
        call_results = []
        for i, call_dict in enumerate(tool_calls):
            tool_call = ToolCall(
                tool_name=call_dict.get("tool", call_dict.get("action", "")),
                params=call_dict.get("params", {}),
                result=call_dict.get("result"),
                error=call_dict.get("error"),
                expected_tool=expected_calls[i].get("tool"),
                expected_params=expected_calls[i].get("params"),
            )

            result = self.evaluate_call(tool_call, agent_interpretations[i])
            call_results.append(result)

        # Aggregate metrics
        total_calls = len(call_results)
        if total_calls == 0:
            return ToolUseResult(
                total_calls=0,
                correct_calls=0,
                selection_accuracy=1.0,
                param_accuracy=1.0,
                interpretation_accuracy=1.0,
                error_handling_rate=1.0,
                overall_score=1.0,
                call_results=[],
                errors_by_type={},
                explanation="No tool calls to evaluate"
            )

        correct_calls = sum(1 for r in call_results if r.score >= 0.8)
        selection_correct = sum(1 for r in call_results if r.selection_correct)
        params_correct = sum(1 for r in call_results if r.params_correct)

        interp_results = [r for r in call_results if r.interpretation_correct is not None]
        interp_correct = sum(1 for r in interp_results if r.interpretation_correct)

        error_results = [r for r in call_results if r.error_handled is not None]
        errors_handled = sum(1 for r in error_results if r.error_handled)

        # Count errors by type
        errors_by_type: Dict[str, int] = {}
        for r in call_results:
            for error in r.errors:
                error_name = error.value
                errors_by_type[error_name] = errors_by_type.get(error_name, 0) + 1

        # Calculate metrics
        selection_accuracy = selection_correct / total_calls
        param_accuracy = params_correct / total_calls
        interpretation_accuracy = (
            interp_correct / len(interp_results) if interp_results else 1.0
        )
        error_handling_rate = (
            errors_handled / len(error_results) if error_results else 1.0
        )

        overall_score = (
            self.selection_weight * selection_accuracy +
            self.params_weight * param_accuracy +
            self.interpretation_weight * interpretation_accuracy +
            self.error_handling_weight * error_handling_rate
        )

        # Build explanation
        explanation_parts = [
            f"{correct_calls}/{total_calls} calls correct",
            f"Selection: {selection_accuracy:.0%}",
            f"Params: {param_accuracy:.0%}",
        ]
        if interp_results:
            explanation_parts.append(f"Interpretation: {interpretation_accuracy:.0%}")
        if error_results:
            explanation_parts.append(f"Error handling: {error_handling_rate:.0%}")

        return ToolUseResult(
            total_calls=total_calls,
            correct_calls=correct_calls,
            selection_accuracy=selection_accuracy,
            param_accuracy=param_accuracy,
            interpretation_accuracy=interpretation_accuracy,
            error_handling_rate=error_handling_rate,
            overall_score=overall_score,
            call_results=call_results,
            errors_by_type=errors_by_type,
            explanation="; ".join(explanation_parts)
        )


# Demo usage
if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 8: Tool Use Evaluation Demo")
    print("=" * 60)

    evaluator = ToolUseEvaluator()

    # Example 1: Correct tool use
    print("\n1. Correct Tool Use")
    print("-" * 40)

    tool_calls = [
        {
            "tool": "search",
            "params": {"query": "weather New York"},
            "result": {"temperature": "72F", "condition": "sunny"}
        },
    ]
    expected_calls = [
        {
            "tool": "search",
            "params": {"query": "weather"}  # Partial match OK
        }
    ]
    interpretations = ["The weather in New York is 72F and sunny."]

    result = evaluator.evaluate_trajectory(
        tool_calls, interpretations, expected_calls
    )

    print(f"Total calls: {result.total_calls}")
    print(f"Selection accuracy: {result.selection_accuracy:.0%}")
    print(f"Parameter accuracy: {result.param_accuracy:.0%}")
    print(f"Overall score: {result.overall_score:.1%}")
    print(f"Explanation: {result.explanation}")

    # Example 2: Wrong tool selected
    print("\n2. Wrong Tool Selection")
    print("-" * 40)

    tool_calls = [
        {
            "tool": "calculate",  # Should have used search
            "params": {"expression": "weather NYC"},
            "result": {"error": "Invalid expression"}
        }
    ]
    expected_calls = [
        {"tool": "search", "params": {"query": "weather NYC"}}
    ]

    result = evaluator.evaluate_trajectory(tool_calls, [None], expected_calls)

    print(f"Selection accuracy: {result.selection_accuracy:.0%}")
    print(f"Parameter accuracy: {result.param_accuracy:.0%}")
    print(f"Errors: {result.errors_by_type}")
    for cr in result.call_results:
        print(f"  - {cr.explanation}")

    # Example 3: Missing parameters
    print("\n3. Missing Parameters")
    print("-" * 40)

    tool_calls = [
        {
            "tool": "send_email",
            "params": {"to": "alice@example.com"},  # Missing subject and body
            "result": {"error": "Missing required parameters"}
        }
    ]
    expected_calls = [
        {
            "tool": "send_email",
            "params": {
                "to": "alice@example.com",
                "subject": "Hello",
                "body": "Hi there!"
            }
        }
    ]

    result = evaluator.evaluate_trajectory(tool_calls, [None], expected_calls)

    print(f"Parameter accuracy: {result.param_accuracy:.0%}")
    print(f"Errors: {result.errors_by_type}")
    for cr in result.call_results:
        print(f"  - {cr.explanation}")

    # Example 4: Error not handled
    print("\n4. Error Not Handled")
    print("-" * 40)

    tool_calls = [
        {
            "tool": "api_call",
            "params": {"endpoint": "/data"},
            "result": None,
            "error": "Connection timeout"
        }
    ]
    # Agent just continues without acknowledging the error
    interpretations = ["Here is the data you requested."]

    result = evaluator.evaluate_trajectory(tool_calls, interpretations)

    print(f"Error handling rate: {result.error_handling_rate:.0%}")
    print(f"Errors: {result.errors_by_type}")

    # Example 5: Full trajectory evaluation
    print("\n5. Full Trajectory Evaluation")
    print("-" * 40)

    full_trajectory = [
        {
            "tool": "search",
            "params": {"query": "Python async tutorial"},
            "result": {"links": ["link1", "link2", "link3"]}
        },
        {
            "tool": "fetch",
            "params": {"url": "link1"},
            "result": {"content": "Async/await in Python..."}
        },
        {
            "tool": "summarize",
            "params": {"text": "Async/await in Python..."},
            "result": {"summary": "Python async enables concurrent execution."}
        }
    ]

    interpretations = [
        "Found 3 relevant tutorials.",
        "Retrieved the first tutorial content.",
        "Python async enables concurrent execution using await keyword."
    ]

    result = evaluator.evaluate_trajectory(full_trajectory, interpretations)

    print(f"Total calls: {result.total_calls}")
    print(f"Correct calls: {result.correct_calls}")
    print(f"Overall score: {result.overall_score:.1%}")
    print(f"\nPer-call results:")
    for i, cr in enumerate(result.call_results):
        print(f"  Call {i+1} ({cr.tool_call.tool_name}): {cr.score:.1%} - {cr.explanation}")

    print("\n" + "=" * 60)
    print("Key Takeaway: Tool use evaluation examines selection, parameters,")
    print("interpretation, and error handling. Each dimension can reveal")
    print("different failure modes in agent behavior.")
    print("=" * 60)
