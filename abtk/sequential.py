"""Looking at results before the end, safely.

Checking a test every few days and stopping at the first p < 0.05 inflates the
false positive rate far above 5%. Group-sequential boundaries fix this: a few
planned looks, each with a stricter threshold, so the overall error stays at
alpha. O'Brien-Fleming boundaries are very strict early and close to the usual
1.96 at the end, so the final analysis loses almost no power.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _max_ratio(n_looks: int, sims: int, seed: int) -> np.ndarray:
    """For simulated null tests, max over looks of |Z_k| * sqrt(k / K)."""
    rng = np.random.default_rng(seed)
    increments = rng.standard_normal((sims, n_looks))
    k = np.arange(1, n_looks + 1)
    z = np.cumsum(increments, axis=1) / np.sqrt(k)
    return np.max(np.abs(z) * np.sqrt(k / n_looks), axis=1)


def obrien_fleming_bounds(n_looks: int, alpha: float = 0.05, sims: int = 400_000, seed: int = 0) -> np.ndarray:
    """Two-sided O'Brien-Fleming z-boundaries for equally spaced looks.

    The boundary at look k is c * sqrt(K / k); c is found by simulation so that
    the chance of crossing any boundary under no effect equals ``alpha``.
    """
    if n_looks < 1:
        raise ValueError("n_looks must be at least 1")
    if n_looks == 1:
        return np.array([stats.norm.ppf(1 - alpha / 2)])
    c = float(np.quantile(_max_ratio(n_looks, sims, seed), 1 - alpha))
    k = np.arange(1, n_looks + 1)
    return c * np.sqrt(n_looks / k)


def false_positive_rate_with_peeking(
    n_looks: int, alpha: float = 0.05, sims: int = 100_000, bounds: np.ndarray | None = None, seed: int = 1
) -> float:
    """Share of no-effect tests that would be declared significant at some look.

    Without ``bounds`` this uses the naive rule (stop at the first |z| > 1.96).
    """
    rng = np.random.default_rng(seed)
    z = np.cumsum(rng.standard_normal((sims, n_looks)), axis=1) / np.sqrt(np.arange(1, n_looks + 1))
    thresholds = np.full(n_looks, stats.norm.ppf(1 - alpha / 2)) if bounds is None else np.asarray(bounds)
    return float(np.mean(np.any(np.abs(z) > thresholds, axis=1)))
