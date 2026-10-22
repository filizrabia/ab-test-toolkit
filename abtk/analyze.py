"""After the test: is the difference real, and how big is it?"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    control: float
    treatment: float
    abs_diff: float
    rel_lift: float
    ci_abs: tuple[float, float]
    ci_rel: tuple[float, float]
    p_value: float
    alpha: float

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def __str__(self) -> str:
        lo, hi = self.ci_rel
        verdict = "significant" if self.significant else "not significant"
        return (
            f"control {self.control:.4g} vs treatment {self.treatment:.4g}: "
            f"lift {self.rel_lift:+.2%} ({1 - self.alpha:.0%} CI {lo:+.2%} to {hi:+.2%}), "
            f"p = {self.p_value:.4f}, {verdict}"
        )


def _relative_ci(m_c, m_t, var_c, var_t, z):
    """Delta-method CI for treatment / control - 1, on the log scale (stays asymmetric and positive-safe)."""
    log_ratio = np.log(m_t / m_c)
    se = np.sqrt(var_t / m_t**2 + var_c / m_c**2)
    return float(np.exp(log_ratio - z * se) - 1), float(np.exp(log_ratio + z * se) - 1)


def proportions_test(conv_c: int, n_c: int, conv_t: int, n_t: int, alpha: float = 0.05) -> TestResult:
    """Two-sided z-test for a difference in conversion rates."""
    p_c, p_t = conv_c / n_c, conv_t / n_t
    p_pool = (conv_c + conv_t) / (n_c + n_t)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_c + 1 / n_t))
    p_value = float(2 * stats.norm.sf(abs(p_t - p_c) / se_pool))
    var_c, var_t = p_c * (1 - p_c) / n_c, p_t * (1 - p_t) / n_t
    z = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(var_c + var_t)
    return TestResult(
        control=p_c,
        treatment=p_t,
        abs_diff=p_t - p_c,
        rel_lift=p_t / p_c - 1,
        ci_abs=(p_t - p_c - z * se, p_t - p_c + z * se),
        ci_rel=_relative_ci(p_c, p_t, var_c, var_t, z),
        p_value=p_value,
        alpha=alpha,
    )


def means_test(x_c: np.ndarray, x_t: np.ndarray, alpha: float = 0.05) -> TestResult:
    """Welch's t-test for a difference in means (e.g. revenue per user)."""
    x_c, x_t = np.asarray(x_c, float), np.asarray(x_t, float)
    m_c, m_t = x_c.mean(), x_t.mean()
    var_c, var_t = x_c.var(ddof=1) / len(x_c), x_t.var(ddof=1) / len(x_t)
    se = np.sqrt(var_c + var_t)
    dof = (var_c + var_t) ** 2 / (var_c**2 / (len(x_c) - 1) + var_t**2 / (len(x_t) - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, dof)
    p_value = float(2 * stats.t.sf(abs(m_t - m_c) / se, dof))
    return TestResult(
        control=float(m_c),
        treatment=float(m_t),
        abs_diff=float(m_t - m_c),
        rel_lift=float(m_t / m_c - 1),
        ci_abs=(float(m_t - m_c - t_crit * se), float(m_t - m_c + t_crit * se)),
        ci_rel=_relative_ci(m_c, m_t, var_c, var_t, t_crit),
        p_value=p_value,
        alpha=alpha,
    )


def cuped(y: np.ndarray, x_pre: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED: remove the part of ``y`` explained by a pre-experiment covariate.

    Pass the pooled data of both groups; the adjusted metric keeps the same mean
    but has lower variance, so the same test needs fewer users. Returns the
    adjusted values and the variance reduction achieved (0.3 = 30% less variance).
    """
    y, x_pre = np.asarray(y, float), np.asarray(x_pre, float)
    theta = np.cov(y, x_pre)[0, 1] / np.var(x_pre, ddof=1)
    adjusted = y - theta * (x_pre - x_pre.mean())
    return adjusted, float(1 - adjusted.var() / y.var())


def bayesian_proportions(
    conv_c: int, n_c: int, conv_t: int, n_t: int, prior: tuple[float, float] = (1, 1), draws: int = 200_000, seed: int = 0
) -> dict[str, float]:
    """Beta-binomial comparison: probability treatment is better, and expected loss of shipping each arm."""
    rng = np.random.default_rng(seed)
    a, b = prior
    s_c = rng.beta(a + conv_c, b + n_c - conv_c, draws)
    s_t = rng.beta(a + conv_t, b + n_t - conv_t, draws)
    return {
        "prob_treatment_better": float((s_t > s_c).mean()),
        "expected_lift": float((s_t / s_c - 1).mean()),
        "expected_loss_treatment": float(np.maximum(s_c - s_t, 0).mean()),
        "expected_loss_control": float(np.maximum(s_t - s_c, 0).mean()),
    }


def srm_check(n_c: int, n_t: int, expected_ratio: float = 1.0, threshold: float = 0.001) -> dict[str, float | bool]:
    """Sample ratio mismatch: did traffic split the way it was supposed to?

    A tiny p-value means the assignment or the tracking is broken, and the test
    result should not be trusted no matter how good it looks.
    """
    total = n_c + n_t
    expected_c = total / (1 + expected_ratio)
    chi2, p = stats.chisquare([n_c, n_t], [expected_c, total - expected_c])
    return {"chi2": float(chi2), "p_value": float(p), "mismatch": bool(p < threshold)}
