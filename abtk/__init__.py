"""abtk: a small A/B testing toolkit for marketing and product experiments."""

from .analyze import TestResult, bayesian_proportions, cuped, means_test, proportions_test, srm_check
from .design import mde_for_sample, power_proportions, sample_size_means, sample_size_proportions, test_duration_days
from .sequential import false_positive_rate_with_peeking, obrien_fleming_bounds

__all__ = [
    "TestResult",
    "bayesian_proportions",
    "cuped",
    "means_test",
    "proportions_test",
    "srm_check",
    "mde_for_sample",
    "power_proportions",
    "sample_size_means",
    "sample_size_proportions",
    "test_duration_days",
    "false_positive_rate_with_peeking",
    "obrien_fleming_bounds",
]
