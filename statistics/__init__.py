"""
Statistics Module
Book 6, Chapter 4: Statistical Rigor in Evaluation

This module provides statistical utilities for evaluation design:
- Sample size calculations
- Confidence intervals
- Clustering adjustments
- Power analysis
"""

from .sample_size import (
    nist_sample_size,
    margin_of_error_sample_size,
    quick_reference_table,
)
from .confidence_intervals import (
    proportion_ci,
    bootstrap_ci,
    clustered_ci,
)

__all__ = [
    "nist_sample_size",
    "margin_of_error_sample_size",
    "quick_reference_table",
    "proportion_ci",
    "bootstrap_ci",
    "clustered_ci",
]
