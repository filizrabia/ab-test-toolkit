"""Before the test: how many users, and how long?"""

from __future__ import annotations

import math

from scipy import stats


def sample_size_proportions(
    baseline: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    relative: bool = True,
    ratio: float = 1.0,
) -> tuple[int, int]:
    """Users needed per group to detect a change in a conversion rate.

    ``mde`` is the minimum detectable effect, relative to ``baseline`` by
    default (0.1 = a 10% lift). ``ratio`` is treatment size / control size.
    Uses the two-sided z-test with unpooled variance under the alternative.
    """
    if not 0 < baseline < 1:
        raise ValueError("baseline must be between 0 and 1")
    p1 = baseline
    p2 = baseline * (1 + mde) if relative else baseline + mde
    if not 0 < p2 < 1:
        raise ValueError("baseline + effect must stay between 0 and 1")
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    p_bar = (p1 + ratio * p2) / (1 + ratio)
    num = (
        z_a * math.sqrt(p_bar * (1 - p_bar) * (1 + 1 / ratio))
        + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)
    ) ** 2
    n_control = math.ceil(num / (p2 - p1) ** 2)
    return n_control, math.ceil(n_control * ratio)


def sample_size_means(sd: float, mde: float, alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0) -> tuple[int, int]:
    """Users needed per group to detect an absolute difference ``mde`` in a mean (e.g. revenue per user)."""
    if sd <= 0 or mde == 0:
        raise ValueError("sd must be positive and mde non-zero")
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    n_control = math.ceil((1 + 1 / ratio) * (sd * (z_a + z_b) / mde) ** 2)
    return n_control, math.ceil(n_control * ratio)


def power_proportions(baseline: float, mde: float, n_per_group: int, alpha: float = 0.05, relative: bool = True) -> float:
    """Power of a two-sided test with ``n_per_group`` users in each arm."""
    p1 = baseline
    p2 = baseline * (1 + mde) if relative else baseline + mde
    se = math.sqrt(p1 * (1 - p1) / n_per_group + p2 * (1 - p2) / n_per_group)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z = abs(p2 - p1) / se
    return float(stats.norm.cdf(z - z_a) + stats.norm.cdf(-z - z_a))


def test_duration_days(n_total: int, daily_eligible_users: int, min_days: int = 7, round_to_weeks: bool = True) -> int:
    """Days to reach ``n_total`` users, at least one full week, optionally rounded up to whole weeks."""
    days = max(min_days, math.ceil(n_total / daily_eligible_users))
    return int(7 * math.ceil(days / 7)) if round_to_weeks else days


def mde_for_sample(baseline: float, n_per_group: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest relative lift detectable with a given sample size (inverse of ``sample_size_proportions``)."""
    lo, hi = 1e-4, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if baseline * (1 + mid) >= 1 or power_proportions(baseline, mid, n_per_group, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return float(hi)


__all__ = [
    "sample_size_proportions",
    "sample_size_means",
    "power_proportions",
    "test_duration_days",
    "mde_for_sample",
]
