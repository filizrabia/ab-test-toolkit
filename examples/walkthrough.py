"""A full experiment, from planning to decision, on simulated data.

    python examples/walkthrough.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import abtk

IMAGES = Path(__file__).resolve().parents[1] / "images"
IMAGES.mkdir(exist_ok=True)
plt.rcParams.update(
    {"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10}
)

# 1. Plan --------------------------------------------------------------------
baseline, mde, daily_users = 0.032, 0.08, 12_000
n, _ = abtk.sample_size_proportions(baseline, mde)
days = abtk.test_duration_days(2 * n, daily_users)
print(f"1. Plan: {n:,} users per group for an 8% lift on a {baseline:.1%} conversion rate -> run {days} days")

lifts = np.linspace(0.03, 0.25, 60)
sizes = [abtk.sample_size_proportions(baseline, m)[0] for m in lifts]
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.plot(lifts * 100, np.array(sizes) / 1000, color="#3B5BDB", lw=2)
ax.scatter([mde * 100], [n / 1000], color="#E8590C", zorder=3)
ax.annotate(f"{n / 1000:.0f}k per group\nfor an 8% lift", (mde * 100, n / 1000), xytext=(12, 150), textcoords="data", fontsize=8,
            arrowprops=dict(arrowstyle="-", color="#868E96"))
ax.set_yscale("log")
ax.set_xlabel("Minimum detectable lift (%)")
ax.set_ylabel("Users per group (thousands)")
ax.set_title(f"Smaller effects need many more users (baseline {baseline:.1%})", loc="left")
fig.tight_layout()
fig.savefig(IMAGES / "sample_size.png")
plt.close(fig)

# 2. Run (simulated) ---------------------------------------------------------
rng = np.random.default_rng(42)
n_c, n_t = n, n + 180
true_lift = 0.09
pre_revenue = rng.gamma(1.5, 30, n_c + n_t)  # revenue per user in the 4 weeks before the test
p_user = np.clip(baseline * (pre_revenue / pre_revenue.mean()) ** 0.5, 0, 1)
treat = np.r_[np.zeros(n_c, bool), np.ones(n_t, bool)]
converted = rng.random(n_c + n_t) < p_user * np.where(treat, 1 + true_lift, 1)
revenue = converted * rng.gamma(2.0, 35, n_c + n_t) + 0.15 * pre_revenue

# 3. Check -------------------------------------------------------------------
srm = abtk.srm_check(n_c, n_t)
print(f"2. SRM check: p = {srm['p_value']:.3f} -> {'MISMATCH, stop' if srm['mismatch'] else 'traffic split looks fine'}")

# 4. Analyze -----------------------------------------------------------------
conv = abtk.proportions_test(converted[~treat].sum(), n_c, converted[treat].sum(), n_t)
print(f"3. Conversion: {conv}")

rpu = abtk.means_test(revenue[~treat], revenue[treat])
adj, reduction = abtk.cuped(revenue, pre_revenue)
rpu_cuped = abtk.means_test(adj[~treat], adj[treat])
print(f"4. Revenue/user, raw:   {rpu}")
print(f"   Revenue/user, CUPED: {rpu_cuped}  (variance -{reduction:.0%})")

bayes = abtk.bayesian_proportions(converted[~treat].sum(), n_c, converted[treat].sum(), n_t)
print(f"5. Bayesian: P(treatment better) = {bayes['prob_treatment_better']:.1%}, "
      f"expected loss if shipped = {bayes['expected_loss_treatment']:.5f}")

fig, ax = plt.subplots(figsize=(6.5, 2.6))
for i, (_label, r) in enumerate([("Revenue/user (raw)", rpu), ("Revenue/user (CUPED)", rpu_cuped), ("Conversion rate", conv)]):
    ax.errorbar(r.rel_lift * 100, i, xerr=[[(r.rel_lift - r.ci_rel[0]) * 100], [(r.ci_rel[1] - r.rel_lift) * 100]],
                fmt="o", color="#3B5BDB", capsize=4)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(range(3), ["Revenue/user (raw)", "Revenue/user (CUPED)", "Conversion rate"])
ax.set_xlabel("Lift vs control (%), 95% CI")
ax.set_title("CUPED narrows the interval on the same data", loc="left")
fig.tight_layout()
fig.savefig(IMAGES / "results.png")
plt.close(fig)

# 5. Peeking ----------------------------------------------------------------
looks = [1, 2, 3, 5, 10, 20]
naive = [abtk.false_positive_rate_with_peeking(k) for k in looks]
safe = abtk.false_positive_rate_with_peeking(5, bounds=abtk.obrien_fleming_bounds(5))
print(f"6. Peeking: naive false positive rate with 5 looks = {naive[3]:.1%}, with O'Brien-Fleming bounds = {safe:.1%}")
fig, ax = plt.subplots(figsize=(6.5, 3))
ax.plot(looks, np.array(naive) * 100, color="#E8590C", lw=2, marker="o", label="Stop at first p < 0.05")
ax.scatter([5], [safe * 100], color="#3B5BDB", s=50, zorder=3, label="5 planned looks, O'Brien-Fleming")
ax.axhline(5, color="#868E96", ls=":", label="Intended 5%")
ax.set_xlabel("Number of times results are checked")
ax.set_ylabel("False positive rate (%)")
ax.set_title("Peeking inflates false positives; planned looks don't", loc="left")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(IMAGES / "peeking.png")
plt.close(fig)
