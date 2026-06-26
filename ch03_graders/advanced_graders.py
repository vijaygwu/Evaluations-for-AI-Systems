"""
Chapter 3: Advanced / composite graders
========================================

Runnable companions to the advanced grader listings in Chapter 3:

- ``MultiAspectGrader``        -- score a response on several dimensions at once.
- ``ConfidenceWeightedGrader`` -- weight each grader's score by its confidence.
- ``contrastive_grade``        -- score by similarity to good vs. bad examples.
- ``StructuredExtractor``      -- regex-extract and validate structured fields.

The per-aspect / per-grader functions and the ``embed_fn`` are supplied by the
caller (any callable with the documented contract); the classes themselves are
complete and runnable.

Book Reference: Chapter 3, "Multi-Aspect / Confidence-Weighted / Contrastive
grading" and "Regex-Based Structured Extraction".
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List, Tuple

import numpy as np


class MultiAspectGrader:
    """Evaluate a response on multiple dimensions simultaneously.

    Each aspect function takes ``(response, context)`` and returns
    ``{"score": float, "passed": bool}``.
    """

    def __init__(self, aspects: Dict[str, Callable[[str, dict], dict]]):
        self.aspects = aspects

    def grade(self, response: str, context: dict) -> dict:
        results: dict = {}
        for aspect_name, grader_fn in self.aspects.items():
            results[aspect_name] = grader_fn(response, context)

        # Compute aggregates before modifying results dict
        aggregate = sum(r["score"] for r in results.values()) / len(results)
        all_passed = all(r.get("passed", True) for r in results.values())

        results["aggregate"] = aggregate
        results["all_passed"] = all_passed
        return results


class ConfidenceWeightedGrader:
    """Weight each grader's score by its self-reported confidence.

    Each grader is a ``(name, function, base_weight)`` tuple; the function
    returns at least ``{"score": float}`` and optionally ``{"confidence": float}``.
    """

    def __init__(self, graders: List[Tuple[str, Callable[[str, dict], dict], float]]):
        self.graders = graders

    def grade(self, response: str, context: dict) -> dict:
        results = []
        for name, grader_fn, base_weight in self.graders:
            result = grader_fn(response, context)
            confidence = result.get("confidence", 1.0)
            results.append({
                "grader": name,
                "score": result["score"],
                "confidence": confidence,
                "weight": base_weight * confidence,
            })

        total_weight = sum(r["weight"] for r in results)
        weighted_score = sum(r["score"] * r["weight"] for r in results) / total_weight

        return {"final_score": weighted_score, "grader_results": results}


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def contrastive_grade(
    response: str,
    good_examples: List[str],
    bad_examples: List[str],
    embed_fn: Callable[[str], np.ndarray],
) -> dict:
    """Grade by similarity to good examples vs. bad examples."""
    response_emb = embed_fn(response)
    good_embs = [embed_fn(ex) for ex in good_examples]
    bad_embs = [embed_fn(ex) for ex in bad_examples]

    avg_good_sim = float(np.mean([cosine_sim(response_emb, g) for g in good_embs]))
    avg_bad_sim = float(np.mean([cosine_sim(response_emb, b) for b in bad_embs]))

    # Score is how much closer to good than bad
    contrast_score = avg_good_sim - avg_bad_sim

    return {
        "contrast_score": contrast_score,
        "good_similarity": avg_good_sim,
        "bad_similarity": avg_bad_sim,
        "classification": "good" if contrast_score > 0 else "bad",
    }


class StructuredExtractor:
    """Extract and validate structured fields from responses."""

    PATTERNS = {
        "answer": r"(?:Answer|ANSWER):\s*(.+?)(?:\n|$)",
        "confidence": r"(?:Confidence|CONFIDENCE):\s*(\d+(?:\.\d+)?)",
        "reasoning": r"(?:Reasoning|REASONING):\s*(.+?)(?:Answer|ANSWER|$)",
        "json_block": r"```json\s*([\s\S]*?)```",
    }

    def extract(self, response: str, fields: List[str]) -> dict:
        results: dict = {}
        for field in fields:
            pattern = self.PATTERNS.get(field)
            if pattern:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                results[field] = match.group(1).strip() if match else None
        return results

    def validate(self, extracted: dict, schema: dict) -> dict:
        """Validate extracted fields against a simple schema."""
        errors = []
        for field, spec in schema.items():
            value = extracted.get(field)
            if spec.get("required") and value is None:
                errors.append(f"Missing required field: {field}")
            if value and "type" in spec:
                if spec["type"] == "number":
                    try:
                        float(value)
                    except ValueError:
                        errors.append(f"Field {field} should be a number")
        return {"valid": len(errors) == 0, "errors": errors}
