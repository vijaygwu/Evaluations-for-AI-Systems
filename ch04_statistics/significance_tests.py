"""
Statistical Significance Tests
==============================

Hypothesis testing for model comparisons.

Book Reference: Chapter 4 discusses comparing model versions and
determining if improvements are real or due to random variation.

Key Question: "A new prompt that improves accuracy from 85% to 87% -
should you ship it?"

Dependencies:
    pip install scipy numpy
"""

import math
from typing import Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import numpy as np


@dataclass
class TestResult:
    """Result of a statistical significance test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool  # At the specified alpha level
    alpha: float
    effect_size: Optional[float] = None
    interpretation: str = ""


def compare_proportions(
    successes_a: int,
    total_a: int,
    successes_b: int,
    total_b: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Compare two proportions using a z-test.

    Tests whether the difference between two independent proportions
    is statistically significant.

    Book Reference: Chapter 4 - Comparing model versions to determine
    if improvement is real

    Args:
        successes_a: Number of successes in group A
        total_a: Total trials in group A
        successes_b: Number of successes in group B
        total_b: Total trials in group B
        alpha: Significance level (default 0.05)
        alternative: "two-sided", "greater" (B > A), or "less" (B < A)

    Returns:
        TestResult with z-statistic and p-value

    Example:
        >>> # Model A: 80% accuracy, Model B: 85% accuracy
        >>> result = compare_proportions(160, 200, 170, 200)
        >>> print(f"p-value: {result.p_value:.4f}")
        >>> print(f"Significant: {result.significant}")
    """
    p_a = successes_a / total_a
    p_b = successes_b / total_b

    # Pooled proportion under null hypothesis
    p_pooled = (successes_a + successes_b) / (total_a + total_b)

    # Standard error under null
    se = math.sqrt(p_pooled * (1 - p_pooled) * (1/total_a + 1/total_b))

    if se == 0:
        return TestResult(
            test_name="Z-test for proportions",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            interpretation="Cannot compute: both groups have same proportion"
        )

    # Z-statistic
    z = (p_b - p_a) / se

    # P-value based on alternative hypothesis
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == "greater":
        p_value = 1 - stats.norm.cdf(z)
    elif alternative == "less":
        p_value = stats.norm.cdf(z)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Effect size (Cohen's h for proportions)
    effect_size = 2 * (math.asin(math.sqrt(p_b)) - math.asin(math.sqrt(p_a)))

    significant = p_value < alpha

    # Interpretation
    diff = p_b - p_a
    if significant:
        if diff > 0:
            interp = f"Model B is significantly better ({p_b:.1%} vs {p_a:.1%}, p={p_value:.4f})"
        else:
            interp = f"Model A is significantly better ({p_a:.1%} vs {p_b:.1%}, p={p_value:.4f})"
    else:
        interp = f"No significant difference ({p_a:.1%} vs {p_b:.1%}, p={p_value:.4f})"

    return TestResult(
        test_name="Z-test for proportions",
        statistic=z,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=effect_size,
        interpretation=interp,
    )


def paired_t_test(
    scores_a: list,
    scores_b: list,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """
    Paired t-test for comparing model scores on the same test cases.

    Use when you have scores from two models on IDENTICAL test cases.
    This accounts for item-level variance and is more powerful than
    comparing proportions when applicable.

    Args:
        scores_a: List of scores from model A (same length as scores_b)
        scores_b: List of scores from model B (paired with scores_a)
        alpha: Significance level
        alternative: "two-sided", "greater", or "less"

    Returns:
        TestResult with t-statistic and p-value

    Example:
        >>> # Same 10 questions, scored by two models
        >>> model_a_scores = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]  # 6/10
        >>> model_b_scores = [1, 1, 1, 1, 0, 1, 1, 1, 1, 0]  # 8/10
        >>> result = paired_t_test(model_a_scores, model_b_scores)
        >>> print(f"p-value: {result.p_value:.4f}")
    """
    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    if len(scores_a) != len(scores_b):
        raise ValueError("Score lists must have same length for paired test")

    # Compute differences
    differences = scores_b - scores_a
    n = len(differences)

    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    se_diff = std_diff / math.sqrt(n)

    if se_diff == 0:
        # All differences are the same
        if mean_diff == 0:
            return TestResult(
                test_name="Paired t-test",
                statistic=0.0,
                p_value=1.0,
                significant=False,
                alpha=alpha,
                interpretation="No difference between models"
            )
        else:
            return TestResult(
                test_name="Paired t-test",
                statistic=float('inf'),
                p_value=0.0,
                significant=True,
                alpha=alpha,
                effect_size=mean_diff,
                interpretation="Perfect difference between models"
            )

    # T-statistic
    t = mean_diff / se_diff
    df = n - 1

    # P-value
    if alternative == "two-sided":
        p_value = 2 * (1 - stats.t.cdf(abs(t), df))
    elif alternative == "greater":
        p_value = 1 - stats.t.cdf(t, df)
    elif alternative == "less":
        p_value = stats.t.cdf(t, df)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    # Effect size (Cohen's d for paired samples)
    effect_size = mean_diff / std_diff if std_diff > 0 else 0

    significant = p_value < alpha

    if significant:
        if mean_diff > 0:
            interp = f"Model B significantly better (mean diff = {mean_diff:.3f}, p={p_value:.4f})"
        else:
            interp = f"Model A significantly better (mean diff = {-mean_diff:.3f}, p={p_value:.4f})"
    else:
        interp = f"No significant difference (mean diff = {mean_diff:.3f}, p={p_value:.4f})"

    return TestResult(
        test_name="Paired t-test",
        statistic=t,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=effect_size,
        interpretation=interp,
    )


def mcnemar_test(
    both_correct: int,
    only_a_correct: int,
    only_b_correct: int,
    both_wrong: int,
    alpha: float = 0.05,
) -> TestResult:
    """
    McNemar's test for paired nominal data.

    Specifically designed for comparing classifiers on the same test set.
    Uses only the "discordant" pairs (where the classifiers disagree).

    Args:
        both_correct: Cases where both A and B are correct
        only_a_correct: Cases where only A is correct
        only_b_correct: Cases where only B is correct
        both_wrong: Cases where both are wrong
        alpha: Significance level

    Returns:
        TestResult with chi-squared statistic and p-value

    Example:
        >>> # Confusion matrix for paired predictions:
        >>> #              Model B Correct    Model B Wrong
        >>> # Model A Correct      70              10
        >>> # Model A Wrong        20              0
        >>> result = mcnemar_test(70, 10, 20, 0)
        >>> print(f"p-value: {result.p_value:.4f}")
    """
    # McNemar uses only discordant pairs
    b = only_a_correct  # A correct, B wrong
    c = only_b_correct  # A wrong, B correct

    n_discordant = b + c

    if n_discordant == 0:
        return TestResult(
            test_name="McNemar's test",
            statistic=0.0,
            p_value=1.0,
            significant=False,
            alpha=alpha,
            interpretation="No discordant pairs - models make identical predictions"
        )

    # McNemar's chi-squared with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)

    # P-value from chi-squared distribution (df=1)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    # Effect size: odds ratio
    if b > 0 and c > 0:
        effect_size = c / b  # Odds ratio (how much more often B is right when they disagree)
    else:
        effect_size = float('inf') if c > b else 0.0

    significant = p_value < alpha

    total = both_correct + only_a_correct + only_b_correct + both_wrong
    acc_a = (both_correct + only_a_correct) / total
    acc_b = (both_correct + only_b_correct) / total

    if significant:
        if only_b_correct > only_a_correct:
            interp = f"Model B significantly better ({acc_b:.1%} vs {acc_a:.1%}, p={p_value:.4f})"
        else:
            interp = f"Model A significantly better ({acc_a:.1%} vs {acc_b:.1%}, p={p_value:.4f})"
    else:
        interp = f"No significant difference ({acc_a:.1%} vs {acc_b:.1%}, p={p_value:.4f})"

    return TestResult(
        test_name="McNemar's test",
        statistic=chi2,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=effect_size,
        interpretation=interp,
    )


def permutation_test(
    scores_a: list,
    scores_b: list,
    n_permutations: int = 10000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> TestResult:
    """
    Permutation test for comparing two groups.

    Non-parametric test that makes no distributional assumptions.
    Answers: "If there were no difference between models, how likely
    is it to see a difference this large by chance?"

    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B (can be different length)
        n_permutations: Number of permutations
        alpha: Significance level
        random_state: Random seed

    Returns:
        TestResult with permutation-based p-value

    Example:
        >>> np.random.seed(42)
        >>> model_a = [0.82, 0.85, 0.79, 0.83, 0.81]
        >>> model_b = [0.88, 0.87, 0.86, 0.89, 0.85]
        >>> result = permutation_test(model_a, model_b)
        >>> print(f"p-value: {result.p_value:.4f}")
    """
    if random_state is not None:
        np.random.seed(random_state)

    scores_a = np.array(scores_a)
    scores_b = np.array(scores_b)

    # Observed difference
    observed_diff = np.mean(scores_b) - np.mean(scores_a)

    # Pool all scores
    pooled = np.concatenate([scores_a, scores_b])
    n_a = len(scores_a)

    # Permutation distribution
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(pooled)
        perm_a = pooled[:n_a]
        perm_b = pooled[n_a:]
        perm_diffs.append(np.mean(perm_b) - np.mean(perm_a))

    perm_diffs = np.array(perm_diffs)

    # Two-sided p-value: proportion of permutations at least as extreme
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))

    significant = p_value < alpha

    if significant:
        if observed_diff > 0:
            interp = f"Model B significantly better (diff = {observed_diff:.3f}, p={p_value:.4f})"
        else:
            interp = f"Model A significantly better (diff = {-observed_diff:.3f}, p={p_value:.4f})"
    else:
        interp = f"No significant difference (diff = {observed_diff:.3f}, p={p_value:.4f})"

    return TestResult(
        test_name="Permutation test",
        statistic=observed_diff,
        p_value=p_value,
        significant=significant,
        alpha=alpha,
        effect_size=observed_diff,
        interpretation=interp,
    )


def multiple_comparison_correction(
    p_values: list,
    alpha: float = 0.05,
    method: str = "bonferroni",
) -> Tuple[list, float]:
    """
    Correct for multiple comparisons.

    When running many tests, the chance of false positives increases.
    Use this to adjust alpha or p-values accordingly.

    Args:
        p_values: List of p-values from multiple tests
        alpha: Original significance level
        method: "bonferroni" or "holm"

    Returns:
        Tuple of (adjusted p-values, adjusted alpha)

    Example:
        >>> p_values = [0.01, 0.03, 0.04, 0.06]
        >>> adjusted, new_alpha = multiple_comparison_correction(p_values)
        >>> print(f"Original alpha: 0.05, Adjusted: {new_alpha:.4f}")
    """
    n = len(p_values)

    if method == "bonferroni":
        # Simple: divide alpha by number of tests
        adjusted_alpha = alpha / n
        adjusted_p = [min(p * n, 1.0) for p in p_values]

    elif method == "holm":
        # Holm-Bonferroni: step-down procedure
        sorted_indices = np.argsort(p_values)
        adjusted_p = [0.0] * n

        for i, idx in enumerate(sorted_indices):
            adjusted_p[idx] = min(p_values[idx] * (n - i), 1.0)

        # Enforce monotonicity
        max_so_far = 0
        for idx in sorted_indices:
            max_so_far = max(max_so_far, adjusted_p[idx])
            adjusted_p[idx] = max_so_far

        adjusted_alpha = alpha  # Holm adjusts p-values, not alpha

    else:
        raise ValueError(f"Unknown method: {method}")

    return adjusted_p, adjusted_alpha


# Demo usage
if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 4: Significance Tests Demo")
    print("=" * 60)

    # Example 1: Compare proportions
    print("\n1. Comparing Two Model Accuracies (Z-test)")
    print("-" * 40)
    print("Model A: 160/200 correct (80%)")
    print("Model B: 170/200 correct (85%)")

    result = compare_proportions(160, 200, 170, 200)
    print(f"\n{result.interpretation}")
    print(f"Z-statistic: {result.statistic:.3f}")
    print(f"P-value: {result.p_value:.4f}")
    print(f"Effect size (Cohen's h): {result.effect_size:.3f}")

    # Example 2: Paired t-test
    print("\n2. Paired Comparison on Same Test Cases")
    print("-" * 40)

    np.random.seed(42)
    n_questions = 100
    # Simulate: Model A gets 80% correct, Model B gets 85% correct
    # but on the same questions
    model_a_scores = np.random.binomial(1, 0.80, n_questions)
    model_b_scores = np.random.binomial(1, 0.85, n_questions)

    print(f"Model A accuracy: {model_a_scores.mean():.1%}")
    print(f"Model B accuracy: {model_b_scores.mean():.1%}")

    result = paired_t_test(model_a_scores.tolist(), model_b_scores.tolist())
    print(f"\n{result.interpretation}")
    print(f"T-statistic: {result.statistic:.3f}")
    print(f"P-value: {result.p_value:.4f}")

    # Example 3: McNemar's test
    print("\n3. McNemar's Test (Classifier Comparison)")
    print("-" * 40)
    print("Contingency table:")
    print("                  Model B Correct   Model B Wrong")
    print("Model A Correct        70              10")
    print("Model A Wrong          20               0")

    result = mcnemar_test(70, 10, 20, 0)
    print(f"\n{result.interpretation}")
    print(f"Chi-squared: {result.statistic:.3f}")
    print(f"P-value: {result.p_value:.4f}")

    # Example 4: When difference is NOT significant
    print("\n4. Non-Significant Difference")
    print("-" * 40)
    print("Model A: 82/100 correct (82%)")
    print("Model B: 85/100 correct (85%)")

    result = compare_proportions(82, 100, 85, 100)
    print(f"\n{result.interpretation}")
    print("Note: Small sample size means we can't detect a 3pp difference")

    # Example 5: Multiple comparisons
    print("\n5. Multiple Comparison Correction")
    print("-" * 40)
    print("Testing 4 model variants, original p-values:")

    p_values = [0.01, 0.03, 0.04, 0.06]
    for i, p in enumerate(p_values):
        print(f"  Model {i+1}: p = {p}")

    adjusted, new_alpha = multiple_comparison_correction(p_values, method="bonferroni")
    print(f"\nBonferroni correction (alpha = 0.05 / 4 = {new_alpha:.4f}):")
    for i, (orig, adj) in enumerate(zip(p_values, adjusted)):
        sig = "significant" if adj < 0.05 else "not significant"
        print(f"  Model {i+1}: adjusted p = {adj:.4f} ({sig})")

    print("\n" + "=" * 60)
    print("Key Takeaway: Don't ship changes based on point estimates alone.")
    print("Use significance tests to distinguish real improvements from noise.")
    print("For paired data, use paired tests (more powerful than unpaired).")
    print("=" * 60)
