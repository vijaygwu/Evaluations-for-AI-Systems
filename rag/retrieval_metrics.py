"""
RAG Retrieval Metrics
=====================

Evaluate the retrieval component of RAG systems.

Book Reference: Chapter 23, Section "Retrieval Evaluation"

Key Metrics (from Table in Chapter 23):
- Precision@k: Fraction of top-k docs that are relevant (target >0.8)
- Recall@k: Fraction of relevant docs found in top-k (target >0.9)
- MRR: Position of first relevant doc (target >0.7)
- NDCG: Ranking quality with graded relevance (target >0.8)

Dependencies:
    pip install numpy
"""

import math
from typing import List, Set, Dict, Any, Optional, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class RetrievalResult:
    """Result of retrieval metric calculation."""
    metric_name: str
    score: float
    k: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


def precision_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: Optional[int] = None,
) -> float:
    """
    Calculate Precision@k: fraction of retrieved docs that are relevant.

    Book Reference: Chapter 23 - "Of the top k retrieved documents,
    what fraction are relevant? High precision means retrieved documents
    are mostly relevant (low noise)."

    Formula: Precision@k = (# relevant in top k) / k

    Args:
        retrieved: List of retrieved document IDs (in order)
        relevant: Set of relevant document IDs
        k: Number of top documents to consider (default: all retrieved)

    Returns:
        Precision@k score (0.0 to 1.0)

    Example:
        >>> retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        >>> relevant = {"doc1", "doc3", "doc7"}
        >>> precision_at_k(retrieved, relevant, k=5)
        0.4  # 2 relevant out of 5 retrieved
    """
    if k is None:
        k = len(retrieved)

    if k == 0:
        return 0.0

    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for doc in top_k if doc in relevant)

    return relevant_in_top_k / k


def recall_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: Optional[int] = None,
) -> float:
    """
    Calculate Recall@k: fraction of relevant docs found in top k.

    Book Reference: Chapter 23 - "Of all relevant documents, what fraction
    appear in top k? High recall means important documents are not being missed."

    Formula: Recall@k = (# relevant in top k) / (total relevant)

    Args:
        retrieved: List of retrieved document IDs (in order)
        relevant: Set of relevant document IDs
        k: Number of top documents to consider (default: all retrieved)

    Returns:
        Recall@k score (0.0 to 1.0)

    Example:
        >>> retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        >>> relevant = {"doc1", "doc3", "doc7"}
        >>> recall_at_k(retrieved, relevant, k=5)
        0.667  # 2 out of 3 relevant found
    """
    if k is None:
        k = len(retrieved)

    if len(relevant) == 0:
        return 1.0  # No relevant docs, perfect recall trivially

    top_k = retrieved[:k]
    relevant_in_top_k = sum(1 for doc in top_k if doc in relevant)

    return relevant_in_top_k / len(relevant)


def f1_at_k(
    retrieved: List[str],
    relevant: Set[str],
    k: Optional[int] = None,
) -> float:
    """
    Calculate F1@k: harmonic mean of precision and recall at k.

    Args:
        retrieved: List of retrieved document IDs
        relevant: Set of relevant document IDs
        k: Number of top documents to consider

    Returns:
        F1@k score (0.0 to 1.0)
    """
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)


def mean_reciprocal_rank(
    retrieved: List[str],
    relevant: Set[str],
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR): where does the first relevant doc appear?

    Book Reference: Chapter 23 - "MRR = (1/|Q|) * sum(1/rank_i) where rank_i
    is the position of the first relevant document for query i."

    For a single query, MRR = 1/rank of first relevant doc.

    Args:
        retrieved: List of retrieved document IDs (in order)
        relevant: Set of relevant document IDs

    Returns:
        Reciprocal rank (0.0 to 1.0, or 0.0 if no relevant docs found)

    Example:
        >>> retrieved = ["doc2", "doc1", "doc3"]  # doc1 is relevant
        >>> relevant = {"doc1"}
        >>> mean_reciprocal_rank(retrieved, relevant)
        0.5  # First relevant at position 2, so 1/2
    """
    for i, doc in enumerate(retrieved):
        if doc in relevant:
            return 1.0 / (i + 1)

    return 0.0  # No relevant document found


def mrr_batch(
    queries_retrieved: List[List[str]],
    queries_relevant: List[Set[str]],
) -> float:
    """
    Calculate MRR across multiple queries.

    Args:
        queries_retrieved: List of retrieved doc lists (one per query)
        queries_relevant: List of relevant doc sets (one per query)

    Returns:
        Mean Reciprocal Rank across all queries
    """
    if len(queries_retrieved) != len(queries_relevant):
        raise ValueError("Mismatch between queries_retrieved and queries_relevant lengths")

    if len(queries_retrieved) == 0:
        return 0.0

    total_rr = sum(
        mean_reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(queries_retrieved, queries_relevant)
    )

    return total_rr / len(queries_retrieved)


def dcg_at_k(
    retrieved: List[str],
    relevance_scores: Dict[str, float],
    k: Optional[int] = None,
) -> float:
    """
    Calculate Discounted Cumulative Gain at k.

    DCG@k = sum(relevance_i / log2(i+1)) for i in 1..k

    Args:
        retrieved: List of retrieved document IDs (in order)
        relevance_scores: Dict mapping doc IDs to relevance scores (0-3 typical)
        k: Number of top documents to consider

    Returns:
        DCG@k score
    """
    if k is None:
        k = len(retrieved)

    dcg = 0.0
    for i, doc in enumerate(retrieved[:k]):
        relevance = relevance_scores.get(doc, 0.0)
        # Position is i+1 (1-indexed), discount is log2(i+2) to avoid log(1)=0
        dcg += relevance / math.log2(i + 2)

    return dcg


def ndcg_at_k(
    retrieved: List[str],
    relevance_scores: Dict[str, float],
    k: Optional[int] = None,
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at k.

    Book Reference: Chapter 23 - "NDCG measures ranking quality with graded relevance.
    DCG@k = sum(relevance_i / log2(i+1)), NDCG@k = DCG@k / IDCG@k"

    NDCG normalizes DCG by the ideal DCG (perfect ranking).

    Args:
        retrieved: List of retrieved document IDs (in order)
        relevance_scores: Dict mapping doc IDs to relevance scores
        k: Number of top documents to consider

    Returns:
        NDCG@k score (0.0 to 1.0)

    Example:
        >>> retrieved = ["doc1", "doc2", "doc3"]
        >>> relevance = {"doc1": 2, "doc2": 0, "doc3": 3, "doc4": 1}
        >>> ndcg_at_k(retrieved, relevance, k=3)
        # Compares actual ranking to ideal (doc3, doc1, doc4)
    """
    if k is None:
        k = len(retrieved)

    if k == 0:
        return 0.0

    # Calculate DCG for actual ranking
    dcg = dcg_at_k(retrieved, relevance_scores, k)

    # Calculate IDCG (ideal DCG) - sort by relevance descending
    ideal_ranking = sorted(
        relevance_scores.keys(),
        key=lambda x: relevance_scores[x],
        reverse=True
    )
    idcg = dcg_at_k(ideal_ranking, relevance_scores, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def average_precision(
    retrieved: List[str],
    relevant: Set[str],
) -> float:
    """
    Calculate Average Precision (AP).

    AP = sum(P@k * rel(k)) / |relevant|

    Where rel(k) = 1 if doc at position k is relevant, 0 otherwise.

    Args:
        retrieved: List of retrieved document IDs
        relevant: Set of relevant document IDs

    Returns:
        Average Precision score
    """
    if len(relevant) == 0:
        return 1.0

    ap_sum = 0.0
    relevant_count = 0

    for i, doc in enumerate(retrieved):
        if doc in relevant:
            relevant_count += 1
            precision = relevant_count / (i + 1)
            ap_sum += precision

    return ap_sum / len(relevant)


def mean_average_precision(
    queries_retrieved: List[List[str]],
    queries_relevant: List[Set[str]],
) -> float:
    """
    Calculate Mean Average Precision (MAP) across multiple queries.

    Args:
        queries_retrieved: List of retrieved doc lists
        queries_relevant: List of relevant doc sets

    Returns:
        MAP score
    """
    if len(queries_retrieved) == 0:
        return 0.0

    total_ap = sum(
        average_precision(retrieved, relevant)
        for retrieved, relevant in zip(queries_retrieved, queries_relevant)
    )

    return total_ap / len(queries_retrieved)


@dataclass
class RetrievalEvaluation:
    """Complete retrieval evaluation results."""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]
    map_score: float


def evaluate_retrieval(
    retrieved: List[str],
    relevant: Set[str],
    relevance_scores: Optional[Dict[str, float]] = None,
    k_values: List[int] = [1, 3, 5, 10],
) -> RetrievalEvaluation:
    """
    Comprehensive retrieval evaluation.

    Args:
        retrieved: List of retrieved document IDs
        relevant: Set of relevant document IDs
        relevance_scores: Optional graded relevance for NDCG
        k_values: Values of k to compute metrics for

    Returns:
        RetrievalEvaluation with all metrics

    Example:
        >>> retrieved = ["d1", "d2", "d3", "d4", "d5"]
        >>> relevant = {"d1", "d3", "d6"}
        >>> result = evaluate_retrieval(retrieved, relevant)
        >>> print(f"P@5: {result.precision_at_k[5]:.2f}")
    """
    # If no graded relevance, use binary
    if relevance_scores is None:
        relevance_scores = {doc: 1.0 for doc in relevant}

    precision_scores = {}
    recall_scores = {}
    f1_scores = {}
    ndcg_scores = {}

    for k in k_values:
        precision_scores[k] = precision_at_k(retrieved, relevant, k)
        recall_scores[k] = recall_at_k(retrieved, relevant, k)
        f1_scores[k] = f1_at_k(retrieved, relevant, k)
        ndcg_scores[k] = ndcg_at_k(retrieved, relevance_scores, k)

    mrr = mean_reciprocal_rank(retrieved, relevant)
    ap = average_precision(retrieved, relevant)

    return RetrievalEvaluation(
        precision_at_k=precision_scores,
        recall_at_k=recall_scores,
        f1_at_k=f1_scores,
        mrr=mrr,
        ndcg_at_k=ndcg_scores,
        map_score=ap,
    )


# Demo usage
if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 23: RAG Retrieval Metrics Demo")
    print("=" * 60)

    # Example 1: Basic metrics
    print("\n1. Basic Retrieval Metrics")
    print("-" * 40)

    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc7"}  # doc7 not retrieved

    print(f"Retrieved: {retrieved}")
    print(f"Relevant: {relevant}")

    print(f"\nPrecision@5: {precision_at_k(retrieved, relevant, 5):.3f}")
    print(f"Recall@5: {recall_at_k(retrieved, relevant, 5):.3f}")
    print(f"F1@5: {f1_at_k(retrieved, relevant, 5):.3f}")
    print(f"MRR: {mean_reciprocal_rank(retrieved, relevant):.3f}")

    # Example 2: NDCG with graded relevance
    print("\n2. NDCG with Graded Relevance")
    print("-" * 40)

    # Relevance: 0 = not relevant, 1 = marginally, 2 = relevant, 3 = highly relevant
    relevance_scores = {
        "doc1": 3,  # Highly relevant
        "doc2": 0,  # Not relevant
        "doc3": 2,  # Relevant
        "doc4": 1,  # Marginally relevant
        "doc5": 0,  # Not relevant
        "doc6": 3,  # Highly relevant (not retrieved)
    }

    print(f"Relevance scores: {relevance_scores}")
    print(f"Retrieved order: {retrieved}")

    for k in [1, 3, 5]:
        ndcg = ndcg_at_k(retrieved, relevance_scores, k)
        print(f"NDCG@{k}: {ndcg:.3f}")

    # Example 3: MRR across multiple queries
    print("\n3. MRR Across Multiple Queries")
    print("-" * 40)

    queries_retrieved = [
        ["a", "b", "c"],  # First relevant at position 1
        ["x", "y", "z"],  # First relevant at position 3
        ["p", "q", "r"],  # No relevant docs
    ]
    queries_relevant = [
        {"a", "c"},
        {"z"},
        {"w"},
    ]

    for i, (ret, rel) in enumerate(zip(queries_retrieved, queries_relevant)):
        rr = mean_reciprocal_rank(ret, rel)
        print(f"Query {i+1}: RR = {rr:.3f}")

    mrr = mrr_batch(queries_retrieved, queries_relevant)
    print(f"Mean RR: {mrr:.3f}")

    # Example 4: Comprehensive evaluation
    print("\n4. Comprehensive Evaluation")
    print("-" * 40)

    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e",
                 "doc_f", "doc_g", "doc_h", "doc_i", "doc_j"]
    relevant = {"doc_a", "doc_c", "doc_f", "doc_k", "doc_l"}
    relevance_scores = {
        "doc_a": 3, "doc_b": 0, "doc_c": 2, "doc_d": 1, "doc_e": 0,
        "doc_f": 3, "doc_g": 0, "doc_h": 1, "doc_i": 0, "doc_j": 0,
        "doc_k": 2, "doc_l": 3,
    }

    result = evaluate_retrieval(
        retrieved, relevant, relevance_scores,
        k_values=[1, 3, 5, 10]
    )

    print("Precision@k:")
    for k, score in result.precision_at_k.items():
        print(f"  @{k}: {score:.3f}")

    print("\nRecall@k:")
    for k, score in result.recall_at_k.items():
        print(f"  @{k}: {score:.3f}")

    print("\nNDCG@k:")
    for k, score in result.ndcg_at_k.items():
        print(f"  @{k}: {score:.3f}")

    print(f"\nMRR: {result.mrr:.3f}")
    print(f"MAP: {result.map_score:.3f}")

    # Example 5: Interpretation with targets from book
    print("\n5. Comparing to Targets (from Chapter 23)")
    print("-" * 40)
    print("Metric        Score    Target    Status")
    print("-" * 40)

    targets = {
        "Precision@5": (result.precision_at_k[5], 0.8),
        "Recall@5": (result.recall_at_k[5], 0.9),
        "MRR": (result.mrr, 0.7),
        "NDCG@5": (result.ndcg_at_k[5], 0.8),
    }

    for name, (score, target) in targets.items():
        status = "PASS" if score >= target else "FAIL"
        print(f"{name:<14} {score:.3f}    {target:.1f}      {status}")

    print("\n" + "=" * 60)
    print("Key Takeaway: Retrieval metrics measure different aspects:")
    print("- Precision: Are retrieved docs relevant? (reduces noise)")
    print("- Recall: Are relevant docs found? (prevents misses)")
    print("- MRR: Is the first result good? (user experience)")
    print("- NDCG: Is the ranking optimal? (with graded relevance)")
    print("=" * 60)
