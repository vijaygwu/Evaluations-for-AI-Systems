"""
Execution-Based Graders
=======================

Graders that execute generated code and verify correctness via test cases.
The ultimate test for code generation: does the code actually run correctly?

Book Reference: Chapter 3, Section "Execution-Based Grading"
"For code generation tasks, the ultimate test is whether the code runs correctly.
HumanEval uses functional correctness testing."

Key Metrics:
- pass@k: Probability of at least one correct solution in k samples
- Timeout enforcement to prevent infinite loops

Dependencies:
    pip install RestrictedPython
"""

import sys
import signal
import traceback
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from io import StringIO
from contextlib import contextmanager
import math


@dataclass
class ExecutionResult:
    """Result of code execution."""
    passed: bool
    output: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    timed_out: bool = False


@dataclass
class TestResult:
    """Result of running a test case."""
    test_name: str
    passed: bool
    expected: Any
    actual: Any
    error: Optional[str] = None


@dataclass
class CodeEvalResult:
    """Complete evaluation result for generated code."""
    passed: bool
    score: float  # Fraction of tests passed
    test_results: List[TestResult]
    compilation_error: Optional[str] = None
    pass_at_k: Optional[float] = None


class TimeoutError(Exception):
    """Raised when code execution times out."""
    pass


@contextmanager
def timeout(seconds: float):
    """
    Context manager for execution timeout.

    Uses SIGALRM on Unix systems. Falls back to no timeout on Windows.
    """
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Execution timed out after {seconds} seconds")

    # Only works on Unix
    if hasattr(signal, 'SIGALRM'):
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # No timeout on Windows
        yield


@contextmanager
def capture_output():
    """Capture stdout and stderr during execution."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    try:
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


class CodeExecutionGrader:
    """
    Execute generated code and verify with test cases.

    Book Reference: Chapter 3 - "HumanEval uses functional correctness testing:
    generated code is executed against test cases."

    WARNING: Executing arbitrary code is inherently dangerous. This grader
    uses basic sandboxing but is NOT suitable for untrusted code in production.
    Consider using containerized execution (Docker) for real applications.
    """

    def __init__(
        self,
        timeout_seconds: float = 3.0,
        restricted_execution: bool = True,
    ):
        """
        Initialize CodeExecutionGrader.

        Args:
            timeout_seconds: Maximum execution time per test (default 3.0s as per HumanEval)
            restricted_execution: Use RestrictedPython for safer execution
        """
        self.timeout_seconds = timeout_seconds
        self.restricted_execution = restricted_execution

    def _create_sandbox(self) -> Dict[str, Any]:
        """Create a sandboxed execution environment."""
        # Safe built-ins for code execution
        safe_builtins = {
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'dict': dict,
            'enumerate': enumerate,
            'filter': filter,
            'float': float,
            'int': int,
            'len': len,
            'list': list,
            'map': map,
            'max': max,
            'min': min,
            'print': print,
            'range': range,
            'reversed': reversed,
            'round': round,
            'set': set,
            'sorted': sorted,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'zip': zip,
            'True': True,
            'False': False,
            'None': None,
            'isinstance': isinstance,
            'type': type,
        }

        return {
            '__builtins__': safe_builtins,
            'math': math,
        }

    def execute_code(
        self,
        code: str,
        test_input: Any = None,
        entry_point: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute code and return result.

        Args:
            code: Python code to execute
            test_input: Input to pass to the function
            entry_point: Name of function to call (if None, executes entire code)

        Returns:
            ExecutionResult with output or error
        """
        import time
        start_time = time.time()

        try:
            with timeout(self.timeout_seconds):
                # Create execution environment
                sandbox = self._create_sandbox() if self.restricted_execution else {}

                # Execute code to define functions
                exec(code, sandbox)

                # Call entry point if specified
                if entry_point:
                    if entry_point not in sandbox:
                        return ExecutionResult(
                            passed=False,
                            output=None,
                            error=f"Entry point '{entry_point}' not found in code"
                        )

                    func = sandbox[entry_point]

                    # Handle different input types
                    if test_input is None:
                        result = func()
                    elif isinstance(test_input, tuple):
                        result = func(*test_input)
                    elif isinstance(test_input, dict):
                        result = func(**test_input)
                    else:
                        result = func(test_input)
                else:
                    # No entry point - check if code ran without error
                    result = None

                execution_time = (time.time() - start_time) * 1000

                return ExecutionResult(
                    passed=True,
                    output=result,
                    execution_time_ms=execution_time
                )

        except TimeoutError as e:
            return ExecutionResult(
                passed=False,
                output=None,
                error=str(e),
                timed_out=True
            )

        except Exception as e:
            return ExecutionResult(
                passed=False,
                output=None,
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            )

    def grade(
        self,
        code: str,
        test_cases: List[Dict[str, Any]],
        entry_point: str,
    ) -> CodeEvalResult:
        """
        Grade generated code against test cases.

        Args:
            code: Generated Python code
            test_cases: List of test cases, each with:
                - 'input': Input to pass to function
                - 'expected': Expected output
                - 'name' (optional): Test case name
            entry_point: Name of function to test

        Returns:
            CodeEvalResult with pass/fail and test details

        Example:
            >>> grader = CodeExecutionGrader()
            >>> code = '''
            ... def add(a, b):
            ...     return a + b
            ... '''
            >>> test_cases = [
            ...     {'input': (1, 2), 'expected': 3, 'name': 'basic'},
            ...     {'input': (-1, 1), 'expected': 0, 'name': 'negative'},
            ... ]
            >>> result = grader.grade(code, test_cases, 'add')
            >>> print(f"Score: {result.score:.1%}")  # "Score: 100.0%"
        """
        # First, check if code compiles
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return CodeEvalResult(
                passed=False,
                score=0.0,
                test_results=[],
                compilation_error=f"Syntax error: {e}"
            )

        # Run test cases
        test_results = []
        passed_count = 0

        for i, test_case in enumerate(test_cases):
            test_name = test_case.get('name', f'test_{i}')
            test_input = test_case.get('input')
            expected = test_case.get('expected')

            # Execute
            exec_result = self.execute_code(code, test_input, entry_point)

            if exec_result.error:
                test_results.append(TestResult(
                    test_name=test_name,
                    passed=False,
                    expected=expected,
                    actual=None,
                    error=exec_result.error
                ))
            else:
                # Compare output to expected
                passed = self._compare_outputs(exec_result.output, expected)
                if passed:
                    passed_count += 1

                test_results.append(TestResult(
                    test_name=test_name,
                    passed=passed,
                    expected=expected,
                    actual=exec_result.output,
                ))

        score = passed_count / len(test_cases) if test_cases else 0.0

        return CodeEvalResult(
            passed=(score == 1.0),
            score=score,
            test_results=test_results,
        )

    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        """Compare actual output to expected, with tolerance for floats."""
        if isinstance(expected, float) and isinstance(actual, (int, float)):
            return abs(actual - expected) < 1e-6

        if isinstance(expected, list) and isinstance(actual, list):
            if len(expected) != len(actual):
                return False
            return all(self._compare_outputs(a, e) for a, e in zip(actual, expected))

        return actual == expected


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Calculate pass@k metric.

    Book Reference: Chapter 3 - "pass@k metric measures the probability of
    at least one correct solution in k samples"

    The estimator is:
        pass@k = 1 - C(n-c, k) / C(n, k)

    Where:
    - n = total number of samples
    - c = number of correct samples
    - k = number of samples to select

    Args:
        n: Total samples generated
        c: Number of correct samples
        k: Number of samples selected

    Returns:
        Estimated pass@k probability

    Example:
        >>> # 3 correct out of 10 samples
        >>> pass_at_k(n=10, c=3, k=1)  # ~0.3
        >>> pass_at_k(n=10, c=3, k=5)  # ~0.74
    """
    if n - c < k:
        return 1.0

    # Use log-space for numerical stability
    # pass@k = 1 - C(n-c, k) / C(n, k)
    # = 1 - (n-c)! / (k! * (n-c-k)!) * (k! * (n-k)!) / n!
    # = 1 - (n-c)! * (n-k)! / ((n-c-k)! * n!)

    numerator = 1.0
    denominator = 1.0

    for i in range(k):
        numerator *= (n - c - i)
        denominator *= (n - i)

    return 1.0 - numerator / denominator


class HumanEvalGrader:
    """
    HumanEval-style code generation evaluation.

    Evaluates multiple code samples using pass@k metric.
    """

    def __init__(
        self,
        timeout_seconds: float = 3.0,
        k_values: List[int] = [1, 10, 100],
    ):
        """
        Initialize HumanEvalGrader.

        Args:
            timeout_seconds: Timeout per test execution
            k_values: Values of k for pass@k calculation
        """
        self.execution_grader = CodeExecutionGrader(timeout_seconds=timeout_seconds)
        self.k_values = k_values

    def evaluate_samples(
        self,
        code_samples: List[str],
        test_cases: List[Dict[str, Any]],
        entry_point: str,
    ) -> Dict[str, Any]:
        """
        Evaluate multiple code samples and compute pass@k.

        Args:
            code_samples: List of generated code solutions
            test_cases: Test cases to run
            entry_point: Function name to test

        Returns:
            Dict with pass@k scores and individual results

        Example:
            >>> grader = HumanEvalGrader(k_values=[1, 5, 10])
            >>> samples = [
            ...     "def add(a, b): return a + b",
            ...     "def add(a, b): return a - b",  # Wrong
            ...     "def add(a, b): return a + b",
            ... ]
            >>> test_cases = [{'input': (1, 2), 'expected': 3}]
            >>> result = grader.evaluate_samples(samples, test_cases, 'add')
            >>> print(f"pass@1: {result['pass@1']:.2f}")
        """
        n = len(code_samples)
        correct_count = 0
        results = []

        for i, code in enumerate(code_samples):
            result = self.execution_grader.grade(code, test_cases, entry_point)
            results.append({
                'sample_idx': i,
                'passed': result.passed,
                'score': result.score,
            })
            if result.passed:
                correct_count += 1

        # Compute pass@k for each k
        pass_at_k_scores = {}
        for k in self.k_values:
            if k <= n:
                pass_at_k_scores[f'pass@{k}'] = pass_at_k(n, correct_count, k)

        return {
            'n_samples': n,
            'n_correct': correct_count,
            'correct_rate': correct_count / n if n > 0 else 0.0,
            **pass_at_k_scores,
            'individual_results': results,
        }


# Demo usage
if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 3: Execution-Based Graders Demo")
    print("=" * 60)

    # Example 1: Basic Code Execution
    print("\n1. CodeExecutionGrader (Basic)")
    print("-" * 40)

    grader = CodeExecutionGrader(timeout_seconds=3.0)

    # Correct implementation
    correct_code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""

    test_cases = [
        {'input': 0, 'expected': 1, 'name': 'factorial(0)'},
        {'input': 1, 'expected': 1, 'name': 'factorial(1)'},
        {'input': 5, 'expected': 120, 'name': 'factorial(5)'},
        {'input': 10, 'expected': 3628800, 'name': 'factorial(10)'},
    ]

    result = grader.grade(correct_code, test_cases, 'factorial')
    print(f"Correct implementation: Score = {result.score:.1%}")
    for tr in result.test_results:
        status = "PASS" if tr.passed else "FAIL"
        print(f"  [{status}] {tr.test_name}: expected={tr.expected}, got={tr.actual}")

    # Buggy implementation
    print()
    buggy_code = """
def factorial(n):
    if n == 0:
        return 0  # Bug: should return 1
    return n * factorial(n - 1)
"""

    result = grader.grade(buggy_code, test_cases, 'factorial')
    print(f"Buggy implementation: Score = {result.score:.1%}")
    for tr in result.test_results:
        status = "PASS" if tr.passed else "FAIL"
        print(f"  [{status}] {tr.test_name}: expected={tr.expected}, got={tr.actual}")

    # Example 2: Timeout handling
    print("\n2. Timeout Handling")
    print("-" * 40)

    infinite_loop_code = """
def infinite():
    while True:
        pass
    return 42
"""

    result = grader.grade(
        infinite_loop_code,
        [{'input': None, 'expected': 42}],
        'infinite'
    )
    print(f"Infinite loop: Passed = {result.passed}")
    if result.test_results and result.test_results[0].error:
        print(f"  Error: {result.test_results[0].error[:50]}...")

    # Example 3: pass@k calculation
    print("\n3. pass@k Metric")
    print("-" * 40)

    # Scenario: 3 correct out of 10 samples
    n, c = 10, 3
    print(f"Scenario: {c} correct out of {n} samples")
    for k in [1, 5, 10]:
        score = pass_at_k(n, c, k)
        print(f"  pass@{k} = {score:.3f}")

    # Example 4: HumanEval-style evaluation
    print("\n4. HumanEvalGrader (Multiple Samples)")
    print("-" * 40)

    eval_grader = HumanEvalGrader(k_values=[1, 5])

    code_samples = [
        "def add(a, b): return a + b",      # Correct
        "def add(a, b): return a - b",      # Wrong
        "def add(a, b): return a + b",      # Correct
        "def add(a, b): return a * b",      # Wrong
        "def add(a, b): return a + b",      # Correct
    ]

    add_test_cases = [
        {'input': (1, 2), 'expected': 3},
        {'input': (0, 0), 'expected': 0},
        {'input': (-1, 1), 'expected': 0},
    ]

    result = eval_grader.evaluate_samples(code_samples, add_test_cases, 'add')
    print(f"Samples: {result['n_samples']}, Correct: {result['n_correct']}")
    print(f"Correct rate: {result['correct_rate']:.1%}")
    for k in [1, 5]:
        key = f'pass@{k}'
        if key in result:
            print(f"{key}: {result[key]:.3f}")

    print("\n" + "=" * 60)
    print("Key Takeaway: Execution-based grading is the gold standard")
    print("for code generation - it tests actual behavior, not just")
    print("surface similarity. Use pass@k to measure with multiple samples.")
    print("=" * 60)
