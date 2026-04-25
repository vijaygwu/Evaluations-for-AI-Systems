"""
Confidence Interval Calculations
================================

Quantify uncertainty in evaluation metrics.

Book Reference: Chapter 4, Section "Confidence Intervals and Uncertainty"

Key Insight: "Point estimates---single numbers like '85% accuracy'---hide
uncertainty. Confidence intervals reveal it."

Formula: CI = mean +/- (1.96 * SEM) for 95% confidence

Dependencies:
    pip install scipy numpy
"""

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class ConfidenceInterval:
    """Result of confidence interval calculation."""
    point_estimate: float
    lower: float
    upper: float
    confidence_level: float
    standard_error: float
    method: str = "normal"

    @property
    def margin_of_error(self) -> float:
        """Half-width of the confidence interval."""
        return (self.upper - self.lower) / 2

    def __str__(self) -> str:
        return (f"{self.point_estimate:.3f} "
                f"[{self.lower:.3f}, {self.upper:.3f}] "
                f"({self.confidence_level:.0%} CI)")


def proportion_ci(
    successes: int,
    total: int,
    confidence: float = 0.95,
    method: str = "wilson",
) -> ConfidenceInterval:
    """
    Calculate confidence interval for a proportion.

    Book Reference: Chapter 4 - "For a proportion p estimated from n samples:
    SEM = sqrt(p(1-p)/n)"

    Args:
        successes: Number of successes (e.g., correct predictions)
        total: Total number of trials
        confidence: Confidence level (default 0.95)
        method: CI method - "normal", "wilson", or "exact"
            - "normal": Standard normal approximation (fast, less accurate for extreme p)
            - "wilson": Wilson score interval (recommended for small n or extreme p)
            - "exact": Clopper-Pearson exact interval (conservative)

    Returns:
        ConfidenceInterval with lower and upper bounds

    Example:
        >>> # 85% accuracy from 400 samples
        >>> ci = proportion_ci(340, 400)
        >>> print(ci)  # "0.850 [0.814, 0.881] (95% CI)"
    """
    if total == 0:
        raise ValueError("Total cannot be zero")

    p = successes / total
    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)

    if method == "normal":
        # Standard normal approximation
        se = math.sqrt(p * (1 - p) / total)
        lower = p - z * se
        upper = p + z * se

    elif method == "wilson":
        # Wilson score interval (better for small n or extreme p)
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator

        lower = center - spread
        upper = center + spread
        se = (upper - lower) / (2 * z)  # Approximate SE from interval width

    elif method == "exact":
        # Clopper-Pearson exact interval
        if successes == 0:
            lower = 0.0
        else:
            lower = stats.beta.ppf(alpha / 2, successes, total - successes + 1)

        if successes == total:
            upper = 1.0
        else:
            upper = stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes)

        se = math.sqrt(p * (1 - p) / total)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'normal', 'wilson', or 'exact'")

    # Clip to [0, 1]
    lower = max(0.0, lower)
    upper = min(1.0, upper)

    return ConfidenceInterval(
        point_estimate=p,
        lower=lower,
        upper=upper,
        confidence_level=confidence,
        standard_error=se,
        method=method,
    )


def mean_ci(
    values: List[float],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """
    Calculate confidence interval for a mean.

    Uses t-distribution for small samples, normal for large.

    Book Reference: Chapter 4 - "Anthropic's statistical methodology uses
    standard error of the mean (SEM) with 95% confidence intervals"

    Args:
        values: List of numeric values
        confidence: Confidence level

    Returns:
        ConfidenceInterval for the mean

    Example:
        >>> scores = [0.85, 0.82, 0.88, 0.84, 0.86, 0.83, 0.87]
        >>> ci = mean_ci(scores)
        >>> print(f"Mean score: {ci}")
    """
    n = len(values)
    if n == 0:
        raise ValueError("Cannot compute CI for empty list")

    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample standard deviation
    se = std / math.sqrt(n)

    alpha = 1 - confidence

    # Use t-distribution for small samples
    if n < 30:
        t_val = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin = t_val * se
        method = "t-distribution"
    else:
        z_val = stats.norm.ppf(1 - alpha / 2)
        margin = z_val * se
        method = "normal"

    return ConfidenceInterval(
        point_estimate=mean,
        lower=mean - margin,
        upper=mean + margin,
        confidence_level=confidence,
        standard_error=se,
        method=method,
    )


def bootstrap_ci(
    values: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    statistic: str = "mean",
    random_state: Optional[int] = None,
) -> ConfidenceInterval:
    """
    Calculate bootstrap confidence interval.

    Non-parametric method that works for any statistic.
    Useful when assumptions of normal/t methods don't hold.

    Book Reference: Chapter 4 mentions bootstrap for non-standard metrics

    Args:
        values: List of values
        confidence: Confidence level
        n_bootstrap: Number of bootstrap samples
        statistic: "mean", "median", or callable
        random_state: Random seed for reproducibility

    Returns:
        ConfidenceInterval using percentile bootstrap

    Example:
        >>> scores = [0.85, 0.82, 0.88, 0.84, 0.86]
        >>> ci = bootstrap_ci(scores, statistic="median")
        >>> print(f"Median: {ci}")
    """
    if random_state is not None:
        np.random.seed(random_state)

    values = np.array(values)
    n = len(values)

    # Determine statistic function
    if statistic == "mean":
        stat_func = np.mean
    elif statistic == "median":
        stat_func = np.median
    elif callable(statistic):
        stat_func = statistic
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Original statistic
    point_estimate = stat_func(values)

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_stats.append(stat_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Percentile method
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    # Standard error from bootstrap
    se = np.std(bootstrap_stats)

    return ConfidenceInterval(
        point_estimate=point_estimate,
        lower=lower,
        upper=upper,
        confidence_level=confidence,
        standard_error=se,
        method=f"bootstrap ({statistic})",
    )


def difference_ci(
    successes_a: int,
    total_a: int,
    successes_b: int,
    total_b: int,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """
    Calculate confidence interval for difference between two proportions.

    Useful for A/B testing model comparisons.

    Args:
        successes_a: Successes in group A
        total_a: Total in group A
        successes_b: Successes in group B
        total_b: Total in group B
        confidence: Confidence level

    Returns:
        ConfidenceInterval for (p_b - p_a)

    Example:
        >>> # Compare two models: A (80%) vs B (85%)
        >>> ci = difference_ci(160, 200, 170, 200)
        >>> print(f"Improvement: {ci}")
    """
    p_a = successes_a / total_a
    p_b = successes_b / total_b
    diff = p_b - p_a

    # Pooled standard error for difference
    se_diff = math.sqrt(
        p_a * (1 - p_a) / total_a +
        p_b * (1 - p_b) / total_b
    )

    alpha = 1 - confidence
    z = stats.norm.ppf(1 - alpha / 2)

    lower = diff - z * se_diff
    upper = diff + z * se_diff

    return ConfidenceInterval(
        point_estimate=diff,
        lower=lower,
        upper=upper,
        confidence_level=confidence,
        standard_error=se_diff,
        method="normal (unpooled)",
    )


def interpret_ci(ci: ConfidenceInterval, threshold: Optional[float] = None) -> str:
    """
    Provide interpretation of a confidence interval.

    Args:
        ci: ConfidenceInterval to interpret
        threshold: Optional threshold to compare against

    Returns:
        Human-readable interpretation
    """
    interpretation = [
        f"Point estimate: {ci.point_estimate:.3f}",
        f"{ci.confidence_level:.0%} CI: [{ci.lower:.3f}, {ci.upper:.3f}]",
        f"Margin of error: +/- {ci.margin_of_error:.3f}",
    ]

    if threshold is not None:
        if ci.lower > threshold:
            interpretation.append(
                f"Statistically above threshold ({threshold}) - "
                f"entire CI is above"
            )
        elif ci.upper < threshold:
            interpretation.append(
                f"Statistically below threshold ({threshold}) - "
                f"entire CI is below"
            )
        else:
            interpretation.append(
                f"Cannot conclude vs threshold ({threshold}) - "
                f"CI includes threshold"
            )

    return "\n".join(interpretation)


# Demo usage
if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 4: Confidence Intervals Demo")
    print("=" * 60)

    # Example 1: Proportion CI (from book)
    print("\n1. Proportion Confidence Interval")
    print("-" * 40)
    print("Scenario: 85% accuracy (340/400) from evaluation")

    ci = proportion_ci(340, 400, method="normal")
    print(f"\nNormal approximation: {ci}")

    ci_wilson = proportion_ci(340, 400, method="wilson")
    print(f"Wilson score: {ci_wilson}")

    print(f"\nInterpretation:")
    print(f"  The true accuracy is between {ci.lower:.1%} and {ci.upper:.1%}")
    print(f"  with {ci.confidence_level:.0%} confidence")

    # Example 2: CI width varies with sample size
    print("\n2. Effect of Sample Size on CI Width")
    print("-" * 40)
    print("Same 85% accuracy, different sample sizes:")

    for n in [100, 400, 1000, 10000]:
        successes = int(0.85 * n)
        ci = proportion_ci(successes, n)
        print(f"  n={n:>5}: {ci.point_estimate:.0%} +/- {ci.margin_of_error:.1%} "
              f"[{ci.lower:.1%}, {ci.upper:.1%}]")

    # Example 3: Difference CI for model comparison
    print("\n3. Comparing Two Models (Difference CI)")
    print("-" * 40)
    print("Model A: 160/200 = 80%")
    print("Model B: 176/200 = 88%")

    ci = difference_ci(160, 200, 176, 200)
    print(f"\nDifference (B - A): {ci}")
    print(f"\nInterpretation:")
    if ci.lower > 0:
        print(f"  Model B is statistically better (CI doesn't include 0)")
    else:
        print(f"  Cannot conclude B is better (CI includes 0)")

    # Example 4: Worked example from book
    print("\n4. Defect Rate Example (from Chapter 4)")
    print("-" * 40)
    print("Threshold: 5% maximum defect rate")

    print("\nWith 200 samples, 3% observed defect rate (6 defects):")
    ci_200 = proportion_ci(6, 200)
    print(f"  CI: {ci_200}")
    print(f"  Upper bound ({ci_200.upper:.1%}) > 5% threshold")
    print(f"  --> Cannot conclude system meets requirements")

    print("\nWith 400 samples, 3% observed defect rate (12 defects):")
    ci_400 = proportion_ci(12, 400)
    print(f"  CI: {ci_400}")
    print(f"  Upper bound ({ci_400.upper:.1%}) < 5% threshold")
    print(f"  --> Can conclude system meets requirements")

    # Example 5: Bootstrap CI
    print("\n5. Bootstrap CI (Non-parametric)")
    print("-" * 40)
    scores = [0.85, 0.82, 0.88, 0.84, 0.86, 0.83, 0.87, 0.89, 0.81, 0.84]
    print(f"Scores: {scores}")

    ci_mean = bootstrap_ci(scores, statistic="mean", random_state=42)
    print(f"\nMean: {ci_mean}")

    ci_median = bootstrap_ci(scores, statistic="median", random_state=42)
    print(f"Median: {ci_median}")

    print("\n" + "=" * 60)
    print("Key Takeaway: Always report confidence intervals alongside")
    print("point estimates. CIs reveal the uncertainty in your metrics")
    print("and help distinguish real effects from noise.")
    print("=" * 60)
