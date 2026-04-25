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

from .retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)
from .faithfulness import FaithfulnessScorer, GroundednessChecker
from .ragas_integration import RAGASEvaluator

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "FaithfulnessScorer",
    "GroundednessChecker",
    "RAGASEvaluator",
]
