"""
Chapter 4: Statistical Rigor in Evaluation
==========================================

Statistical foundations for AI evaluation:
- Sample size calculations
- Confidence intervals
- Hypothesis testing
- Power analysis

Book Reference: Chapter 4 covers the statistical foundations -
"Evaluation without statistical rigor produces noise, not signal."
"""

from .sample_size import calculate_sample_size, sample_size_for_proportion
from .confidence_intervals import (
    proportion_ci,
    mean_ci,
    bootstrap_ci,
)
from .significance_tests import (
    compare_proportions,
    paired_t_test,
    mcnemar_test,
)

__all__ = [
    "calculate_sample_size",
    "sample_size_for_proportion",
    "proportion_ci",
    "mean_ci",
    "bootstrap_ci",
    "compare_proportions",
    "paired_t_test",
    "mcnemar_test",
]
