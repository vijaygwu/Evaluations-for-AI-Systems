"""
Sample Size Calculations
========================

Determine how many test cases you need for statistically meaningful results.

Book Reference: Chapter 4, Section "Sample Size and Power"

Key Insight: "Standard error decreases proportionally to the square root
of sample size; to reduce margin of error by half, quadruple the sample size."

Quick Reference (from Chapter 4):
- +/-10% margin: ~100 samples
- +/-5% margin:  ~385 samples
- +/-3% margin:  ~1,070 samples
- +/-1% margin:  ~9,600 samples

Dependencies:
    pip install scipy numpy
"""

import math
from typing import Optional
from dataclasses import dataclass
from scipy import stats
import numpy as np


@dataclass
class SampleSizeResult:
    """Result of sample size calculation."""
    n: int  # Required sample size
    confidence_level: float
    margin_of_error: float
    power: Optional[float] = None
    notes: str = ""


def sample_size_for_margin_of_error(
    margin_of_error: float,
    confidence_level: float = 0.95,
    proportion: float = 0.5,
) -> SampleSizeResult:
    """
    Calculate sample size needed for a desired margin of error.

    Book Reference: Chapter 4 - "The most common question in evaluation design:
    how many test cases do I need?"

    Uses the formula: n = (z^2 * p * (1-p)) / e^2

    Where:
    - z = z-score for confidence level
    - p = estimated proportion (0.5 is worst case / most conservative)
    - e = margin of error

    Args:
        margin_of_error: Desired margin of error (e.g., 0.05 for +/-5%)
        confidence_level: Confidence level (default 0.95 for 95%)
        proportion: Estimated proportion (0.5 is conservative)

    Returns:
        SampleSizeResult with required sample size

    Example:
        >>> result = sample_size_for_margin_of_error(0.05, 0.95)
        >>> print(f"Need {result.n} samples for +/-5% margin at 95% confidence")
        # Output: Need 385 samples for +/-5% margin at 95% confidence
    """
    # Get z-score for confidence level
    alpha = 1 - confidence_level
    z = stats.norm.ppf(1 - alpha / 2)

    # Calculate sample size
    n = (z ** 2 * proportion * (1 - proportion)) / (margin_of_error ** 2)

    return SampleSizeResult(
        n=math.ceil(n),
        confidence_level=confidence_level,
        margin_of_error=margin_of_error,
        notes=f"Using proportion={proportion}, z={z:.3f}"
    )


def sample_size_for_proportion(
    p0: float,
    p1: float,
    alpha: float = 0.05,
    power: float = 0.80,
    one_sided: bool = True,
) -> SampleSizeResult:
    """
    Calculate sample size to detect a shift in proportion.

    Book Reference: Chapter 4 - "NIST Formula for Detecting Proportion Shifts"

    Formula: N >= ((z_{1-alpha} * sqrt(p0*(1-p0)) + z_{1-beta} * sqrt(p1*(1-p1))) / delta)^2

    Where:
    - alpha = false positive rate (Type I error)
    - beta = false negative rate (1 - power, Type II error)
    - delta = |p1 - p0| = minimum difference to detect

    Args:
        p0: Baseline proportion (e.g., current accuracy of 0.80)
        p1: Target proportion (e.g., improved accuracy of 0.85)
        alpha: Significance level (default 0.05)
        power: Statistical power (default 0.80, meaning 80% chance to detect real effect)
        one_sided: Use one-sided test (default True for "improvement" scenarios)

    Returns:
        SampleSizeResult with required sample size per group

    Example:
        >>> # Detect improvement from 80% to 85% accuracy
        >>> result = sample_size_for_proportion(0.80, 0.85, power=0.80)
        >>> print(f"Need {result.n} samples per group")
    """
    beta = 1 - power
    delta = abs(p1 - p0)

    if delta == 0:
        raise ValueError("p0 and p1 cannot be equal")

    # Z-scores
    if one_sided:
        z_alpha = stats.norm.ppf(1 - alpha)
    else:
        z_alpha = stats.norm.ppf(1 - alpha / 2)

    z_beta = stats.norm.ppf(power)

    # Calculate sample size using NIST formula
    numerator = z_alpha * math.sqrt(p0 * (1 - p0)) + z_beta * math.sqrt(p1 * (1 - p1))
    n = (numerator / delta) ** 2

    # Add continuity correction
    n_corrected = n + 1 / (2 * delta)

    return SampleSizeResult(
        n=math.ceil(n_corrected),
        confidence_level=1 - alpha,
        margin_of_error=delta,
        power=power,
        notes=f"Detecting shift from {p0:.1%} to {p1:.1%}, "
              f"{'one-sided' if one_sided else 'two-sided'} test"
    )


def calculate_sample_size(
    baseline: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Convenience function: calculate sample size to detect minimum detectable effect.

    Args:
        baseline: Current proportion/accuracy (e.g., 0.80)
        mde: Minimum detectable effect as absolute difference (e.g., 0.05 for 5 percentage points)
        alpha: Significance level
        power: Statistical power

    Returns:
        Required sample size per group

    Example:
        >>> n = calculate_sample_size(baseline=0.80, mde=0.05, power=0.80)
        >>> print(f"Need {n} samples per group to detect 5pp improvement")
    """
    target = baseline + mde
    # Ensure target is within valid range
    target = min(max(target, 0.01), 0.99)

    result = sample_size_for_proportion(baseline, target, alpha, power)
    return result.n


def power_analysis(
    n: int,
    p0: float,
    p1: float,
    alpha: float = 0.05,
    one_sided: bool = True,
) -> float:
    """
    Calculate power given a fixed sample size.

    Answers: "With N samples, what's my chance of detecting this effect?"

    Args:
        n: Sample size per group
        p0: Baseline proportion
        p1: Alternative proportion
        alpha: Significance level
        one_sided: One-sided test

    Returns:
        Statistical power (probability of detecting the effect if it exists)

    Example:
        >>> power = power_analysis(n=400, p0=0.80, p1=0.85)
        >>> print(f"Power: {power:.1%}")  # "Power: 80.2%"
    """
    delta = abs(p1 - p0)

    if one_sided:
        z_alpha = stats.norm.ppf(1 - alpha)
    else:
        z_alpha = stats.norm.ppf(1 - alpha / 2)

    # Standard error under null
    se_null = math.sqrt(p0 * (1 - p0) / n)

    # Standard error under alternative
    se_alt = math.sqrt(p1 * (1 - p1) / n)

    # Effect size in standard error units
    z_beta = (delta - z_alpha * se_null) / se_alt

    # Power is P(Z > z_beta) under the alternative
    power = stats.norm.cdf(z_beta)

    return power


def detectable_effect(
    n: int,
    baseline: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """
    Calculate minimum detectable effect for a given sample size.

    Answers: "With N samples, what's the smallest effect I can reliably detect?"

    Args:
        n: Sample size per group
        baseline: Baseline proportion
        alpha: Significance level
        power: Desired power

    Returns:
        Minimum detectable effect (absolute difference)

    Example:
        >>> mde = detectable_effect(n=400, baseline=0.80)
        >>> print(f"Can detect {mde:.1%} improvement with 400 samples")
    """
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)

    # Approximate using baseline variance (iterative solution would be more accurate)
    se = math.sqrt(baseline * (1 - baseline) / n)

    mde = (z_alpha + z_beta) * se

    return mde


# Quick reference tables from Chapter 4
def print_sample_size_reference():
    """Print quick reference table for common sample size scenarios."""
    print("=" * 60)
    print("Sample Size Quick Reference (95% confidence)")
    print("=" * 60)

    print("\nTable 1: Sample size for margin of error")
    print("-" * 40)
    print(f"{'Margin of Error':<20} {'Required n':<15}")
    for moe in [0.10, 0.05, 0.03, 0.01]:
        result = sample_size_for_margin_of_error(moe)
        print(f"+/-{moe:.0%}               {result.n:<15}")

    print("\nTable 2: Sample size per group for detecting differences")
    print("-" * 50)
    print(f"{'Baseline':<12} {'Effect':<12} {'Samples/Group':<15}")

    scenarios = [
        (0.50, 0.10),  # 50% -> 60%
        (0.80, 0.05),  # 80% -> 85%
        (0.90, 0.03),  # 90% -> 93%
        (0.95, 0.02),  # 95% -> 97%
    ]

    for baseline, effect in scenarios:
        n = calculate_sample_size(baseline, effect)
        print(f"{baseline:.0%} -> {baseline+effect:.0%}    {effect:.0%}          ~{n}")


# Demo usage
if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 4: Sample Size Calculations Demo")
    print("=" * 60)

    # Example 1: Margin of Error
    print("\n1. Sample Size for Margin of Error")
    print("-" * 40)

    for moe in [0.10, 0.05, 0.03, 0.01]:
        result = sample_size_for_margin_of_error(moe)
        print(f"+/-{moe:.0%} margin of error: n = {result.n}")

    # Example 2: Detecting Proportion Shift (from book)
    print("\n2. Detecting Proportion Shift (NIST Example)")
    print("-" * 40)
    print("Scenario: Detect shift from 10% to 20% defective")

    result = sample_size_for_proportion(0.10, 0.20, alpha=0.05, power=0.90)
    print(f"Required n = {result.n}")
    print(f"  {result.notes}")

    # Example 3: Real evaluation scenario
    print("\n3. Practical Example: Model Comparison")
    print("-" * 40)
    print("Scenario: Current model has 80% accuracy.")
    print("Want to detect if new model is 5 percentage points better.")

    n = calculate_sample_size(baseline=0.80, mde=0.05, power=0.80)
    print(f"\nRequired samples per model: {n}")

    # Check power at various sample sizes
    print("\nPower at different sample sizes:")
    for n_test in [100, 200, 400, 800]:
        pwr = power_analysis(n=n_test, p0=0.80, p1=0.85)
        print(f"  n={n_test}: power = {pwr:.1%}")

    # Example 4: What can we detect?
    print("\n4. Minimum Detectable Effect")
    print("-" * 40)
    print("With 500 samples per group at 80% baseline:")

    mde = detectable_effect(n=500, baseline=0.80)
    print(f"  Minimum detectable effect: {mde:.1%} ({mde*100:.1f} percentage points)")

    # Print reference tables
    print("\n")
    print_sample_size_reference()

    print("\n" + "=" * 60)
    print("Key Takeaway: Always calculate required sample size BEFORE")
    print("running your evaluation. Underpowered tests waste resources")
    print("and produce unreliable results.")
    print("=" * 60)
