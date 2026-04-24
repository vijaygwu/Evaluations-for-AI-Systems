"""
JSON Match Grader
Book 6, Chapter 3: Grader Taxonomy

Compares JSON objects for semantic equivalence, ignoring:
- Key ordering
- Whitespace formatting
- Optional field presence (configurable)

From OpenAI Evals - JsonMatch pattern.
"""

import json
from typing import List, Optional, Any, Union
from dataclasses import dataclass


@dataclass
class JsonGradeResult:
    """Result of JSON grading."""
    passed: bool
    score: float
    matched_reference: Optional[str] = None
    parse_error: Optional[str] = None
    diff: Optional[dict] = None


class JsonMatchGrader:
    """
    Grade JSON outputs by semantic equivalence.

    Compares parsed JSON objects, ignoring key order and whitespace.

    Example:
        >>> grader = JsonMatchGrader()
        >>> result = grader.grade(
        ...     '{"name": "test", "value": 42}',
        ...     ['{"value": 42, "name": "test"}']
        ... )
        >>> result.passed
        True
    """

    def __init__(
        self,
        ignore_extra_keys: bool = False,
        ignore_missing_keys: bool = False,
        numeric_tolerance: float = 0.0
    ):
        """
        Initialize the JSON grader.

        Args:
            ignore_extra_keys: Don't fail if completion has extra keys
            ignore_missing_keys: Don't fail if completion is missing keys
            numeric_tolerance: Allow numeric values within this tolerance
        """
        self.ignore_extra_keys = ignore_extra_keys
        self.ignore_missing_keys = ignore_missing_keys
        self.numeric_tolerance = numeric_tolerance

    def grade(
        self,
        completion: str,
        references: List[str]
    ) -> JsonGradeResult:
        """
        Grade JSON completion against references.

        Args:
            completion: JSON string to grade
            references: List of acceptable JSON strings

        Returns:
            JsonGradeResult with match details
        """
        # Parse completion
        try:
            completion_obj = json.loads(completion)
        except json.JSONDecodeError as e:
            return JsonGradeResult(
                passed=False,
                score=0.0,
                parse_error=f"Completion parse error: {str(e)}"
            )

        # Check against each reference
        for ref in references:
            try:
                ref_obj = json.loads(ref)
            except json.JSONDecodeError:
                continue  # Skip invalid reference

            if self._objects_match(completion_obj, ref_obj):
                return JsonGradeResult(
                    passed=True,
                    score=1.0,
                    matched_reference=ref
                )

        return JsonGradeResult(
            passed=False,
            score=0.0,
            diff=self._compute_diff(completion_obj, json.loads(references[0]))
            if references else None
        )

    def _objects_match(self, obj1: Any, obj2: Any) -> bool:
        """Recursively compare two objects for equivalence."""
        # Same type check
        if type(obj1) != type(obj2):
            # Special case: int vs float
            if isinstance(obj1, (int, float)) and isinstance(obj2, (int, float)):
                return self._numbers_match(obj1, obj2)
            return False

        # Dict comparison
        if isinstance(obj1, dict):
            keys1 = set(obj1.keys())
            keys2 = set(obj2.keys())

            # Check for extra keys
            if not self.ignore_extra_keys and keys1 - keys2:
                return False

            # Check for missing keys
            if not self.ignore_missing_keys and keys2 - keys1:
                return False

            # Compare shared keys
            shared_keys = keys1 & keys2
            return all(
                self._objects_match(obj1[k], obj2[k])
                for k in shared_keys
            )

        # List comparison (order matters)
        if isinstance(obj1, list):
            if len(obj1) != len(obj2):
                return False
            return all(
                self._objects_match(a, b)
                for a, b in zip(obj1, obj2)
            )

        # Numeric comparison with tolerance
        if isinstance(obj1, (int, float)):
            return self._numbers_match(obj1, obj2)

        # Default: exact equality
        return obj1 == obj2

    def _numbers_match(self, n1: Union[int, float], n2: Union[int, float]) -> bool:
        """Compare numbers with optional tolerance."""
        if self.numeric_tolerance > 0:
            return abs(n1 - n2) <= self.numeric_tolerance
        return n1 == n2

    def _compute_diff(self, obj1: Any, obj2: Any, path: str = "") -> dict:
        """Compute differences between two objects."""
        diff = {}

        if type(obj1) != type(obj2):
            diff[path or "root"] = {"expected": type(obj2).__name__, "got": type(obj1).__name__}
            return diff

        if isinstance(obj1, dict):
            all_keys = set(obj1.keys()) | set(obj2.keys())
            for key in all_keys:
                key_path = f"{path}.{key}" if path else key
                if key not in obj1:
                    diff[key_path] = {"missing_in_completion": True}
                elif key not in obj2:
                    diff[key_path] = {"extra_in_completion": True}
                elif obj1[key] != obj2[key]:
                    if isinstance(obj1[key], dict) and isinstance(obj2[key], dict):
                        diff.update(self._compute_diff(obj1[key], obj2[key], key_path))
                    else:
                        diff[key_path] = {"expected": obj2[key], "got": obj1[key]}

        elif isinstance(obj1, list):
            if len(obj1) != len(obj2):
                diff[path or "root"] = {"expected_length": len(obj2), "got_length": len(obj1)}

        elif obj1 != obj2:
            diff[path or "root"] = {"expected": obj2, "got": obj1}

        return diff


class JsonSchemaGrader:
    """
    Grade JSON outputs against a JSON Schema.

    Validates structure without requiring exact values.
    Useful for checking API response formats.
    """

    def __init__(self, schema: dict):
        """
        Initialize with a JSON Schema.

        Args:
            schema: JSON Schema dict
        """
        try:
            import jsonschema
            self.validator = jsonschema.Draft7Validator(schema)
        except ImportError:
            raise ImportError("jsonschema package required: pip install jsonschema")

    def grade(self, completion: str) -> JsonGradeResult:
        """Validate completion against schema."""
        try:
            obj = json.loads(completion)
        except json.JSONDecodeError as e:
            return JsonGradeResult(
                passed=False,
                score=0.0,
                parse_error=str(e)
            )

        errors = list(self.validator.iter_errors(obj))

        if not errors:
            return JsonGradeResult(passed=True, score=1.0)
        else:
            return JsonGradeResult(
                passed=False,
                score=0.0,
                diff={"schema_errors": [e.message for e in errors[:5]]}
            )
