"""
Chapter 23: RAG Evaluation
==========================

Evaluate Retrieval-Augmented Generation systems:
- Retrieval metrics (precision, recall, MRR, NDCG)
- Generation metrics (faithfulness, relevance)
- End-to-end evaluation

Book Reference: Chapter 23 covers the unique challenges of RAG evaluation -
"you are evaluating a pipeline, not just a model."
"""

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "FaithfulnessScorer",
    "GroundednessChecker",
    "RAGASEvaluator",
    "evaluate_groundedness",
    "detect_rag_hallucinations",
]

_EXPORTS = {
    "precision_at_k": ".retrieval_metrics",
    "recall_at_k": ".retrieval_metrics",
    "mean_reciprocal_rank": ".retrieval_metrics",
    "ndcg_at_k": ".retrieval_metrics",
    "FaithfulnessScorer": ".faithfulness",
    "GroundednessChecker": ".faithfulness",
    "RAGASEvaluator": ".ragas_integration",
    "evaluate_groundedness": ".groundedness",
    "detect_rag_hallucinations": ".groundedness",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
