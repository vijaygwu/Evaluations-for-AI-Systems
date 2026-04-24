"""
Confidence Interval Calculations
Book 6, Chapter 4: Statistical Rigor in Evaluation

Provides utilities for computing confidence intervals, including:
- Standard proportion CIs
- Bootstrap CIs
- Clustered CIs (accounting for question dependencies)

Key insight from Anthropic research:
"Clustered standard errors on popular evals can be over three times
as large as naive."
"""

import math
from typing import List, Tuple, Optional
import numpy as np


def proportion_ci(
    successes: int,
    total: int,
    confidence: float = 0.95,
    method: str = "wilson"
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for a proportion.

    Args:
        successes: Number of successes (correct answers)
        total: Total number of trials
        confidence: Confidence level (default 0.95)
        method: "normal", "wilson", or "clopper-pearson"

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)

    Example:
        >>> proportion_ci(85, 100)
        (0.85, 0.766, 0.912)  # 85% accuracy with 95% CI
    """
    try:
        from scipy import stats
    except ImportError:
        raise ImportError("scipy required: pip install scipy")

    p = successes / total
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)

    if method == "normal":
        # Normal approximation (simple but can give bounds outside [0,1])
        se = math.sqrt(p * (1 - p) / total)
        lower = max(0, p - z * se)
        upper = min(1, p + z * se)

    elif method == "wilson":
        # Wilson score interval (recommended for most cases)
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
        lower = max(0, center - margin)
        upper = min(1, center + margin)

    elif method == "clopper-pearson":
        # Exact binomial CI (conservative)
        lower = stats.beta.ppf(alpha / 2, successes, total - successes + 1)
        upper = stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes)
        if successes == 0:
            lower = 0
        if successes == total:
            upper = 1

    else:
        raise ValueError(f"Unknown method: {method}")

    return round(p, 4), round(lower, 4), round(upper, 4)


def bootstrap_ci(
    scores: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    statistic: str = "mean"
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval.

    Useful when distribution is unknown or for complex statistics.

    Args:
        scores: List of individual scores
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples
        statistic: "mean", "median", or "proportion"

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)

    Example:
        >>> scores = [1, 1, 1, 0, 1, 0, 1, 1, 0, 1]  # Binary pass/fail
        >>> bootstrap_ci(scores)
        (0.7, 0.4, 0.9)
    """
    scores = np.array(scores)
    n = len(scores)

    # Point estimate
    if statistic == "mean":
        point_est = np.mean(scores)
        stat_func = np.mean
    elif statistic == "median":
        point_est = np.median(scores)
        stat_func = np.median
    elif statistic == "proportion":
        point_est = np.mean(scores)
        stat_func = np.mean
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrap_stats.append(stat_func(sample))

    # Percentile method
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return round(float(point_est), 4), round(float(lower), 4), round(float(upper), 4)


def clustered_ci(
    scores: List[float],
    cluster_ids: List[int],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval accounting for clustering.

    Addresses the problem from Anthropic research: questions in benchmarks
    are often not independent. For example, in MMLU, multiple history
    questions about WWII form a cluster.

    Args:
        scores: List of individual scores (0 or 1 for binary)
        cluster_ids: List of cluster IDs (same length as scores)
        confidence: Confidence level

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound)

    Example:
        >>> # Questions 0-2 are about WWII (cluster 0), 3-5 about WWII (cluster 1)
        >>> scores = [1, 1, 1, 0, 0, 0]  # Model knows WWII, not WWI
        >>> clusters = [0, 0, 0, 1, 1, 1]
        >>> clustered_ci(scores, clusters)
        (0.5, 0.0, 1.0)  # Much wider CI than naive
    """
    try:
        from scipy import stats
    except ImportError:
        raise ImportError("scipy required: pip install scipy")

    scores = np.array(scores)
    cluster_ids = np.array(cluster_ids)

    # Overall mean
    overall_mean = np.mean(scores)

    # Calculate cluster means
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)

    cluster_means = []
    for cid in unique_clusters:
        cluster_scores = scores[cluster_ids == cid]
        cluster_means.append(np.mean(cluster_scores))

    cluster_means = np.array(cluster_means)

    # Clustered standard error
    # This accounts for within-cluster correlation
    cluster_se = np.std(cluster_means, ddof=1) / np.sqrt(n_clusters)

    # t-distribution for small number of clusters
    t_value = stats.t.ppf(1 - (1 - confidence) / 2, df=n_clusters - 1)

    lower = max(0, overall_mean - t_value * cluster_se)
    upper = min(1, overall_mean + t_value * cluster_se)

    return round(float(overall_mean), 4), round(float(lower), 4), round(float(upper), 4)


def naive_vs_clustered_comparison(
    scores: List[float],
    cluster_ids: List[int]
) -> dict:
    """
    Compare naive CI to clustered CI to show the difference.

    Demonstrates Anthropic's finding that clustered standard errors
    can be >3x larger than naive.

    Returns:
        Dict with naive and clustered CIs and the ratio
    """
    naive = bootstrap_ci(scores)
    clustered = clustered_ci(scores, cluster_ids)

    naive_width = naive[2] - naive[1]
    clustered_width = clustered[2] - clustered[1]
    ratio = clustered_width / naive_width if naive_width > 0 else float('inf')

    return {
        "point_estimate": naive[0],
        "naive_ci": (naive[1], naive[2]),
        "naive_width": round(naive_width, 4),
        "clustered_ci": (clustered[1], clustered[2]),
        "clustered_width": round(clustered_width, 4),
        "width_ratio": round(ratio, 2),
        "interpretation": (
            f"Clustered CI is {ratio:.1f}x wider than naive CI. "
            f"{'This is within normal range.' if ratio < 2 else 'Significant clustering effect!'}"
        )
    }


if __name__ == "__main__":
    print("=== Book 6, Chapter 4: Confidence Intervals ===\n")

    # Example: 85% accuracy with 100 samples
    print("Example: 85 correct out of 100")
    p, lower, upper = proportion_ci(85, 100)
    print(f"Point estimate: {p:.1%}")
    print(f"95% CI: [{lower:.1%}, {upper:.1%}]")
    print()

    # Clustering example
    print("Clustering Example (MMLU-style):")
    print("Questions about WWII (cluster 0): all correct")
    print("Questions about WWI (cluster 1): all wrong")
    scores = [1, 1, 1, 0, 0, 0]
    clusters = [0, 0, 0, 1, 1, 1]

    result = naive_vs_clustered_comparison(scores, clusters)
    print(f"Naive CI: {result['naive_ci']}")
    print(f"Clustered CI: {result['clustered_ci']}")
    print(f"Width ratio: {result['width_ratio']}x")
    print(result['interpretation'])
