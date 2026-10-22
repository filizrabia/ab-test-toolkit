import unittest

import numpy as np

import abtk


class DesignTests(unittest.TestCase):
    def test_sample_size_matches_reference(self):
        # 10% baseline, 10% relative lift, alpha 0.05, power 0.8: 14,751 per group
        # (same answer as the widely used Evan Miller calculator).
        self.assertEqual(abtk.sample_size_proportions(0.10, 0.10), (14751, 14751))

    def test_power_round_trip(self):
        n, _ = abtk.sample_size_proportions(0.05, 0.15)
        self.assertAlmostEqual(abtk.power_proportions(0.05, 0.15, n), 0.8, places=2)
        self.assertAlmostEqual(abtk.mde_for_sample(0.05, n), 0.15, places=2)

    def test_unequal_split_needs_more_users(self):
        equal = sum(abtk.sample_size_proportions(0.1, 0.1))
        skewed = sum(abtk.sample_size_proportions(0.1, 0.1, ratio=0.25))
        self.assertGreater(skewed, equal)

    def test_means_sample_size(self):
        # n = 2 * (sd * (1.96 + 0.84) / mde)^2 = 2 * (10 * 2.8 / 1)^2 ~ 1570
        n, _ = abtk.sample_size_means(sd=10, mde=1)
        self.assertTrue(1560 < n < 1580)

    def test_duration_rounds_to_full_weeks(self):
        self.assertEqual(abtk.test_duration_days(30_000, 2_000), 21)
        self.assertEqual(abtk.test_duration_days(1_000, 2_000), 7)


class AnalyzeTests(unittest.TestCase):
    def test_proportions_test(self):
        r = abtk.proportions_test(1000, 10000, 1100, 10000)
        self.assertAlmostEqual(r.p_value, 0.02108, places=4)
        self.assertAlmostEqual(r.rel_lift, 0.10)
        self.assertTrue(r.significant)
        self.assertLess(r.ci_rel[0], 0.10)
        self.assertGreater(r.ci_rel[1], 0.10)

    def test_false_positive_rate_is_close_to_alpha(self):
        rng = np.random.default_rng(3)
        hits = 0
        for _ in range(2000):
            c, t = rng.binomial(5000, 0.1), rng.binomial(5000, 0.1)
            hits += abtk.proportions_test(c, 5000, t, 5000).significant
        self.assertTrue(0.035 < hits / 2000 < 0.065)

    def test_means_test_detects_real_difference(self):
        rng = np.random.default_rng(1)
        r = abtk.means_test(rng.normal(100, 20, 3000), rng.normal(103, 20, 3000))
        self.assertTrue(r.significant)
        self.assertTrue(r.ci_abs[0] < 3 < r.ci_abs[1])

    def test_cuped_reduces_variance_and_keeps_mean(self):
        rng = np.random.default_rng(2)
        pre = rng.gamma(2, 50, 10_000)
        y = 0.8 * pre + rng.normal(0, 30, 10_000)
        adj, reduction = abtk.cuped(y, pre)
        self.assertAlmostEqual(adj.mean(), y.mean(), places=6)
        self.assertGreater(reduction, 0.5)

    def test_bayesian(self):
        out = abtk.bayesian_proportions(1000, 10000, 1100, 10000)
        self.assertGreater(out["prob_treatment_better"], 0.98)
        self.assertLess(out["expected_loss_treatment"], out["expected_loss_control"])

    def test_srm(self):
        self.assertFalse(abtk.srm_check(50_100, 49_900)["mismatch"])
        self.assertTrue(abtk.srm_check(51_500, 48_500)["mismatch"])


class SequentialTests(unittest.TestCase):
    def test_obrien_fleming_matches_published_values(self):
        # Published two-sided boundaries for 5 equally spaced looks at alpha = 0.05.
        expected = [4.562, 3.226, 2.634, 2.281, 2.040]
        np.testing.assert_allclose(abtk.obrien_fleming_bounds(5), expected, atol=0.02)

    def test_peeking_inflates_errors_and_bounds_fix_it(self):
        naive = abtk.false_positive_rate_with_peeking(5, sims=50_000)
        safe = abtk.false_positive_rate_with_peeking(5, sims=50_000, bounds=abtk.obrien_fleming_bounds(5))
        self.assertGreater(naive, 0.12)
        self.assertAlmostEqual(safe, 0.05, delta=0.006)


if __name__ == "__main__":
    unittest.main()
