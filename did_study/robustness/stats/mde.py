"""
Minimum detectable effect (MDE) calculations.

This module provides a helper function to compute the minimum
detectable effect for a difference-in-differences estimator given a
standard error and number of clusters.  The calculation uses a
t-distribution with degrees of freedom equal to the number of
clusters minus one when SciPy is available, otherwise falls back to
the standard normal approximation.
"""

from __future__ import annotations

from typing import Optional
import numpy as np


def analytic_mde_from_se(
    se: float,
    n_clusters: int,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> float:
    """Approximate the minimum detectable effect (MDE).

    Given a standard error and number of clusters, compute the effect
    size that would be detected with probability ``power`` at
    significance level ``alpha``.  When SciPy is available a
    t-distribution is used with degrees of freedom ``n_clusters-1``;
    otherwise a normal approximation is used.

    Parameters
    ----------
    se : float
        Standard error of the estimator.
    n_clusters : int
        Number of clusters (units) in the sample.
    alpha : float, default 0.05
        Significance level for a two-sided test.
    power : float, default 0.80
        Desired power of the test.
    two_sided : bool, default True
        If ``True`` use a two-sided critical value, otherwise
        one-sided.

    Returns
    -------
    float
        The minimum detectable effect expressed in the same units as
        ``se``.
    """
    try:
        from scipy.stats import t  # type: ignore
        df = max(int(n_clusters) - 1, 1)
        if two_sided:
            crit = t.ppf(1 - alpha / 2, df) + t.ppf(power, df)
        else:
            crit = t.ppf(1 - alpha, df) + t.ppf(power, df)
    except Exception:
        from statistics import NormalDist  # type: ignore
        z = NormalDist().inv_cdf
        if two_sided:
            crit = z(1 - alpha / 2) + z(power)
        else:
            crit = z(1 - alpha) + z(power)
    return float(abs(se) * crit)