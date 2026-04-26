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

_EXPORTS = {
    "calculate_sample_size": ".sample_size",
    "sample_size_for_proportion": ".sample_size",
    "proportion_ci": ".confidence_intervals",
    "mean_ci": ".confidence_intervals",
    "bootstrap_ci": ".confidence_intervals",
    "compare_proportions": ".significance_tests",
    "paired_t_test": ".significance_tests",
    "mcnemar_test": ".significance_tests",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
