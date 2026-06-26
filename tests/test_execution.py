#!/usr/bin/env python3
"""Execution tests with mocked LLM client / embedder.

The deterministic graders are covered by the grader canaries. THIS suite
exercises the API- and library-dependent code paths end to end *without* real
keys or heavy libraries, by injecting:

  - a fake ``call_llm`` returning a canned ``LLMResponse`` (for the LLM judges
    and the RAG faithfulness/groundedness checkers), and
  - a fake embedder (for the semantic-similarity grader).

Each test asserts the function runs through its real success path (not its
error/except branch) and returns the right result type. RAGAS is skipped if the
optional ``ragas`` package is absent.

Run from anywhere. Exit 0 = all pass; exit 1 = an execution path failed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.llm_clients import LLMResponse  # noqa: E402

failures: list[str] = []
ran = 0


def ok(name: str, cond: bool, detail: str = ""):
    global ran
    ran += 1
    if not cond:
        failures.append(f"{name}: {detail or 'assertion failed'}")


def fake_llm(content: str):
    """A drop-in call_llm that returns a canned LLMResponse (no network)."""
    def _call(*args, **kwargs):
        return LLMResponse(content=content, model="mock", input_tokens=10, output_tokens=10)
    return _call


# --- ch03 LLM judges (mock call_llm in the module namespace) ----------------
import ch03_graders.llm_as_judge as J  # noqa: E402

J.call_llm = fake_llm("REASONING: The answer matches the reference.\nGRADE: CORRECT")
try:
    _dims = [J.RubricDimension(name="Accuracy", description="Factual correctness",
                               weight=1.0, levels={1: "wrong", 5: "correct"})]
    r = J.RubricGrader(dimensions=_dims).grade(question="Capital of France?", response="Paris")
    # RubricGrader is multi-dimensional -> returns a dict of per-dimension results
    ok("RubricGrader.grade runs", isinstance(r, dict) and len(r) > 0, f"returned {type(r).__name__}")
except Exception as e:  # noqa: BLE001
    ok("RubricGrader.grade runs", False, f"raised {type(e).__name__}: {e}")

try:
    r = J.SimpleQAGrader().grade(gold_target="Paris is the capital of France",
                                 model_answer="The capital of France is Paris.")
    ok("SimpleQAGrader.grade runs", r.grade != "ERROR" and not math.isnan(r.score),
       f"grade={r.grade} score={r.score}")
except Exception as e:  # noqa: BLE001
    ok("SimpleQAGrader.grade runs", False, f"raised {type(e).__name__}: {e}")


# --- ch23 RAG faithfulness / groundedness (mock call_llm) -------------------
import ch23_rag.faithfulness as F  # noqa: E402

F.call_llm = fake_llm(
    "SENTENCE: Paris is the capital.\nSTATUS: GROUNDED\n"
    "OVERALL_SCORE: 0.9\nCLAIMS: Paris is the capital of France."
)
try:
    g = F.GroundednessChecker().check(
        response="Paris is the capital of France.",
        context="France's capital city is Paris.",
    )
    ok("GroundednessChecker.check runs", not math.isnan(g.score), f"score={g.score}")
except Exception as e:  # noqa: BLE001
    ok("GroundednessChecker.check runs", False, f"raised {type(e).__name__}: {e}")

try:
    fs = F.FaithfulnessScorer().score(
        question="What is the capital of France?",
        answer="Paris is the capital of France.",
        context="France's capital city is Paris.",
    )
    ok("FaithfulnessScorer.score runs", fs is not None)
except Exception as e:  # noqa: BLE001
    ok("FaithfulnessScorer.score runs", False, f"raised {type(e).__name__}: {e}")


# --- ch03 semantic similarity (inject a fake embedder, no sentence-transformers) ---
import ch03_graders.semantic_similarity as S  # noqa: E402


class _FakeEmbedder:
    """Deterministic bag-of-words-ish embedder; .encode matches the real API."""
    def encode(self, text, convert_to_numpy=True):
        def vec(t):
            v = np.zeros(64)
            for w in str(t).lower().split():
                v[hash(w) % 64] += 1.0
            n = np.linalg.norm(v)
            return v / n if n else v
        if isinstance(text, (list, tuple)):
            return np.array([vec(t) for t in text])
        return vec(text)


try:
    grader = S.SemanticSimilarityGrader()
    grader._model = _FakeEmbedder()  # bypass the lazy SentenceTransformer load
    res = grader.grade(expected="The capital of France is Paris",
                       actual="Paris is the capital of France")
    ok("SemanticSimilarityGrader.grade runs", 0.0 <= res.score <= 1.0, f"score={res.score}")
except Exception as e:  # noqa: BLE001
    ok("SemanticSimilarityGrader.grade runs", False, f"raised {type(e).__name__}: {e}")


# --- ch13 robustness metrics (fake model with .generate) --------------------
from ch13_safety.robustness import attack_success_rate, robustness_score, false_positive_rate, OBFUSCATION_TECHNIQUES  # noqa: E402


class _RefusingModel:
    def generate(self, prompt):
        return "I cannot help with that request."


m = _RefusingModel()
try:
    asr = attack_success_rate(m, ["attack a", "attack b"], success_criteria=lambda r: "cannot" not in r.lower())
    ok("attack_success_rate runs", asr["asr"] == 0.0, f"asr={asr['asr']}")
    rs = robustness_score(m, "base attack", [OBFUSCATION_TECHNIQUES["leetspeak"]], n_perturbations=3)
    ok("robustness_score runs", rs["base_blocked"] is True, f"{rs}")
    fpr = false_positive_rate(m, ["hello", "thanks"], expected_completion=lambda r: True)
    ok("false_positive_rate runs", 0.0 <= fpr["fpr"] <= 1.0, f"fpr={fpr['fpr']}")
except Exception as e:  # noqa: BLE001
    ok("ch13 robustness metrics run", False, f"raised {type(e).__name__}: {e}")


# --- ch23 RAGAS (skip cleanly if the optional package is absent) ------------
try:
    import ragas  # noqa: F401
    have_ragas = True
except Exception:  # noqa: BLE001
    have_ragas = False

if have_ragas:
    try:
        from ch23_rag.ragas_integration import RAGASEvaluator  # noqa: F401
        ok("RAGASEvaluator import (ragas present) runs", True)
    except Exception as e:  # noqa: BLE001
        ok("RAGASEvaluator import (ragas present) runs", False, f"raised {type(e).__name__}: {e}")
else:
    print("[skip] ragas not installed -- RAGASEvaluator execution test skipped")


print(f"[test_execution] ran {ran} mocked execution checks")
if failures:
    print("\n".join(f"  FAIL {f}" for f in failures))
    print(f"FAIL: {len(failures)} execution path(s) errored.")
    sys.exit(1)
print("PASS: all LLM-/embedder-path functions execute end to end with mocked clients.")
sys.exit(0)
