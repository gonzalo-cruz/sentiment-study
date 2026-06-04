"""Statistical validation: Mann-Whitney U, confidence intervals."""

import numpy as np
from scipy import stats


def mann_whitney_test(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
) -> tuple[float, float]:
    """Run two-sided Mann-Whitney U test on two independent samples.

    Args:
        sample_a: First sample of numeric values.
        sample_b: Second sample of numeric values.

    Returns:
        Tuple of (U statistic, p-value).
    """
    result = stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)
