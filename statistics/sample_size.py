"""
Sample Size Calculator
Book 6, Chapter 4: Statistical Rigor in Evaluation

Implements NIST formula and utilities for determining eval sample sizes.

Key principle from Eugene Yan:
"Standard error decreases proportionally to the square root of sample size;
to reduce margin of error by half, quadruple the sample size."
"""

import math
from typing import Optional, Tuple


def nist_sample_size(
    p0: float,
    p1: float,
    alpha: float = 0.05,
    beta: float = 0.10,
    one_sided: bool = True
) -> int:
    """
    Calculate required sample size for detecting proportion shift.

    Based on NIST Engineering Statistics Handbook formula.

    The intuition: Balance two types of errors:
    - False positive (alpha): Declaring improvement when there isn't one
    - False negative (beta): Missing a real improvement

    Args:
        p0: Baseline proportion (e.g., 0.10 for 10% defect rate)
        p1: Target proportion to detect (e.g., 0.20 for 20%)
        alpha: False positive rate (default 0.05 = 5%)
        beta: False negative rate (default 0.10 = 10%, giving 90% power)
        one_sided: Use one-sided test (default True)

    Returns:
        Minimum required sample size (integer)

    Example:
        >>> nist_sample_size(p0=0.10, p1=0.20, alpha=0.05, beta=0.10)
        102

        >>> # Detecting smaller difference requires more samples
        >>> nist_sample_size(p0=0.10, p1=0.15)
        319
    """
    try:
        from scipy import stats
    except ImportError:
        raise ImportError("scipy required: pip install scipy")

    if not (0 < p0 < 1) or not (0 < p1 < 1):
        raise ValueError("Proportions must be between 0 and 1")

    if p0 == p1:
        raise ValueError("p0 and p1 must be different")

    # Z-values for alpha and beta
    if one_sided:
        z_alpha = stats.norm.ppf(1 - alpha)
    else:
        z_alpha = stats.norm.ppf(1 - alpha / 2)

    z_beta = stats.norm.ppf(1 - beta)

    # Effect size
    delta = abs(p1 - p0)

    # NIST formula
    numerator = (z_alpha * math.sqrt(p0 * (1 - p0)) +
                 z_beta * math.sqrt(p1 * (1 - p1)))

    n = (numerator / delta) ** 2

    return math.ceil(n)


def margin_of_error_sample_size(
    margin: float,
    confidence: float = 0.95,
    p: float = 0.5
) -> int:
    """
    Calculate sample size for desired margin of error.

    Args:
        margin: Desired margin of error (e.g., 0.05 for ±5%)
        confidence: Confidence level (default 0.95)
        p: Expected proportion (default 0.5 for worst case)

    Returns:
        Required sample size

    Example:
        >>> margin_of_error_sample_size(0.05)  # ±5% margin
        385
        >>> margin_of_error_sample_size(0.10)  # ±10% margin
        97
        >>> margin_of_error_sample_size(0.03)  # ±3% margin
        1068
    """
    try:
        from scipy import stats
    except ImportError:
        raise ImportError("scipy required: pip install scipy")

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    n = (z ** 2 * p * (1 - p)) / (margin ** 2)
    return math.ceil(n)


def power_analysis(
    n: int,
    p0: float,
    p1: float,
    alpha: float = 0.05,
    one_sided: bool = True
) -> float:
    """
    Calculate the statistical power given sample size.

    Power is the probability of detecting a real effect.
    80% power is standard; 90% is more stringent.

    Args:
        n: Sample size
        p0: Baseline proportion
        p1: Alternative proportion
        alpha: Significance level
        one_sided: One-sided test

    Returns:
        Power (probability of detecting real effect)

    Example:
        >>> power_analysis(n=100, p0=0.10, p1=0.20)
        0.89  # 89% power to detect 10pp difference with n=100
    """
    try:
        from scipy import stats
    except ImportError:
        raise ImportError("scipy required: pip install scipy")

    if one_sided:
        z_alpha = stats.norm.ppf(1 - alpha)
    else:
        z_alpha = stats.norm.ppf(1 - alpha / 2)

    delta = abs(p1 - p0)
    se0 = math.sqrt(p0 * (1 - p0) / n)
    se1 = math.sqrt(p1 * (1 - p1) / n)

    # Z-value under alternative hypothesis
    z_effect = (delta - z_alpha * se0) / se1

    # Power is P(Z > z_effect) under alternative
    power = stats.norm.cdf(z_effect)

    return round(power, 3)


def minimum_detectable_effect(
    n: int,
    p0: float,
    alpha: float = 0.05,
    power: float = 0.80
) -> float:
    """
    Calculate the minimum detectable effect size given sample size.

    Useful for interpreting eval results: "With N samples,
    we can detect differences of X percentage points."

    Args:
        n: Sample size
        p0: Baseline proportion
        alpha: Significance level
        power: Desired power (default 0.80)

    Returns:
        Minimum detectable difference (as proportion)

    Example:
        >>> minimum_detectable_effect(n=400, p0=0.50)
        0.069  # Can detect ~7pp difference with n=400
    """
    try:
        from scipy import stats
    except ImportError:
        raise ImportError("scipy required: pip install scipy")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Approximation using p0 for variance
    se = math.sqrt(p0 * (1 - p0) / n)

    mde = (z_alpha + z_beta) * se

    return round(mde, 3)


def quick_reference_table() -> None:
    """Print the sample size quick reference table from Chapter 4."""
    print("=" * 50)
    print("Sample Size Quick Reference (p=0.5, 95% CI)")
    print("=" * 50)
    print(f"{'Margin of Error':<20} {'Required n':>15}")
    print("-" * 50)

    for margin in [0.10, 0.05, 0.03, 0.01]:
        n = margin_of_error_sample_size(margin)
        print(f"±{margin*100:.0f}%{'':<16} {n:>15,}")

    print("-" * 50)
    print("\nKey insight: To halve the margin of error,")
    print("quadruple the sample size (square root relationship).")
    print()


def comparison_sample_size_table() -> None:
    """Print sample sizes for comparing two proportions."""
    print("=" * 60)
    print("Sample Sizes for Comparing Two Proportions")
    print("(80% power, α=0.05, per group)")
    print("=" * 60)
    print(f"{'Baseline':<12} {'Difference':<20} {'N per group':>15}")
    print("-" * 60)

    scenarios = [
        (0.50, 0.10),
        (0.80, 0.05),
        (0.90, 0.03),
        (0.95, 0.02),
    ]

    for baseline, diff in scenarios:
        n = nist_sample_size(baseline, baseline + diff, beta=0.20)
        print(f"{baseline*100:.0f}%{'':<9} {diff*100:.0f} percentage points{'':<4} {n:>12,}")

    print("-" * 60)
    print("\nNote: Detecting changes near 100% (or 0%) requires")
    print("more samples than detecting changes near 50%.")


if __name__ == "__main__":
    print("=== Book 6, Chapter 4: Sample Size Calculations ===\n")

    # Example from NIST
    print("Example: Detecting shift from 10% to 20% defective")
    n = nist_sample_size(p0=0.10, p1=0.20)
    print(f"Required N = {n}")
    print()

    quick_reference_table()
    print()
    comparison_sample_size_table()
