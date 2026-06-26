"""
Chapter 23: Groundedness and hallucination detection
=====================================================

Runnable companions to the Chapter 23 listings ``evaluate_groundedness`` and
``detect_rag_hallucinations``. The book listings call helpers they don't define
(``extract_claims``, ``claim_in_source``, ``parse_hallucination_result``);
runnable lightweight versions are provided here so the functions actually run.

For production-grade, error-handling groundedness scoring see
:class:`ch23_rag.faithfulness.GroundednessChecker`; the functions here are the
self-contained, dependency-light versions that mirror the chapter's listings.

Book Reference: Chapter 23, "Groundedness evaluation" and "Hallucination
detection approach".
"""

from __future__ import annotations

import re
from typing import Callable, List

# ---------------------------------------------------------------------------
# Lightweight helpers the book listings reference without defining
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\w+")


def extract_claims(response: str) -> List[str]:
    """Split a response into claim-sized units (one sentence == one claim).

    The book notes "Use NLP or LLM"; this is the dependency-free fallback:
    sentence segmentation. Swap in an NLP/LLM claim extractor for production.
    """
    return [s.strip() for s in _SENTENCE_SPLIT.split(response.strip()) if s.strip()]


def claim_in_source(claim: str, source: str, threshold: float = 0.6) -> bool:
    """Heuristic entailment: is most of the claim's content present in the source?

    Returns True when at least ``threshold`` of the claim's (non-trivial) word
    tokens also appear in the source. A lexical stand-in for the book's
    "appears in or is entailed by sources"; replace with an NLI/LLM check for
    production use.
    """
    claim_tokens = {w.lower() for w in _WORD.findall(claim) if len(w) > 2}
    if not claim_tokens:
        return True
    source_tokens = {w.lower() for w in _WORD.findall(source)}
    overlap = len(claim_tokens & source_tokens) / len(claim_tokens)
    return overlap >= threshold


def parse_hallucination_result(result: str) -> dict:
    """Parse the structured judge output produced by detect_rag_hallucinations."""
    text = result or ""
    if "NO_HALLUCINATIONS_DETECTED" in text:
        return {"hallucination_count": 0, "severity": "NONE", "hallucinations": [], "raw": text}

    count_m = re.search(r"HALLUCINATION_COUNT:\s*(\d+)", text)
    sev_m = re.search(r"SEVERITY:\s*(NONE|LOW|MEDIUM|HIGH)", text, re.IGNORECASE)
    items = re.findall(
        r'HALLUCINATION\s+\d+:\s*"([^"]*)"\s*-\s*Type:\s*(\w+)\s*-\s*Explanation:\s*(.+)',
        text,
    )
    hallucinations = [{"text": t, "type": ty, "explanation": ex.strip()} for t, ty, ex in items]
    count = int(count_m.group(1)) if count_m else len(hallucinations)
    severity = sev_m.group(1).upper() if sev_m else ("HIGH" if count else "NONE")
    return {
        "hallucination_count": count,
        "severity": severity,
        "hallucinations": hallucinations,
        "raw": text,
    }


# ---------------------------------------------------------------------------
# Chapter 23 listings (runnable)
# ---------------------------------------------------------------------------

def evaluate_groundedness(response: str, sources: List[str]) -> dict:
    """Check whether response content can be traced to the sources."""
    claims = extract_claims(response)  # Use NLP or LLM for production

    grounded_claims = []
    ungrounded_claims = []

    for claim in claims:
        if any(claim_in_source(claim, source) for source in sources):
            grounded_claims.append(claim)
        else:
            ungrounded_claims.append(claim)

    return {
        "groundedness_score": len(grounded_claims) / len(claims) if claims else 1.0,
        "grounded_claims": grounded_claims,
        "ungrounded_claims": ungrounded_claims,
    }


def detect_rag_hallucinations(
    response: str,
    retrieved_docs: List[str],
    llm_judge: Callable[[str], str],
) -> dict:
    """Identify potential hallucinations in a RAG response via an LLM judge.

    ``llm_judge`` is any callable taking a prompt and returning the model's text.
    """
    prompt = f"""
    Identify any potential hallucinations in this AI response.

    A hallucination is content that:
    - Is not supported by the provided sources
    - Contradicts the sources
    - Includes fabricated specifics (fake quotes, made-up statistics)

    SOURCES:
    {chr(10).join(retrieved_docs)}

    RESPONSE:
    {response}

    List each potential hallucination:
    HALLUCINATION 1: "[text]" - Type: [UNSUPPORTED/CONTRADICTS/FABRICATED] - Explanation: [why]
    ...

    If no hallucinations found, respond: NO_HALLUCINATIONS_DETECTED

    HALLUCINATION_COUNT: [number]
    SEVERITY: [NONE/LOW/MEDIUM/HIGH]
    """

    result = llm_judge(prompt)
    return parse_hallucination_result(result)
