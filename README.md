# abtk: a small A/B testing toolkit

The handful of functions I reach for in every marketing or product experiment, in one dependency-light package (numpy + scipy): plan the test, sanity-check it, analyze it, and decide.

```python
import abtk

abtk.sample_size_proportions(baseline=0.032, mde=0.08)   # (77062, 77062) users per group
abtk.srm_check(n_c=77_062, n_t=77_242)                   # did traffic split as planned?
abtk.proportions_test(2288, 77_062, 2536, 77_242)         # lift, CI, p-value
abtk.bayesian_proportions(2288, 77_062, 2536, 77_242)     # P(B > A), expected loss
```

## What's in it

**Plan**
- `sample_size_proportions`, `sample_size_means`: users needed per group, with unequal splits supported
- `power_proportions`, `mde_for_sample`: what a given sample can and can't detect
- `test_duration_days`: runtime from daily traffic, rounded to full weeks so weekday effects even out

**Check**
- `srm_check`: sample ratio mismatch. If this fails, the result is not trustworthy, however good it looks

**Analyze**
- `proportions_test`: two-sided z-test for conversion rates
- `means_test`: Welch's t-test for revenue per user and other continuous metrics
- `cuped`: variance reduction using a pre-experiment covariate
- `bayesian_proportions`: beta-binomial probability that treatment wins, plus expected loss of each decision

Every result reports the absolute difference, the relative lift and a confidence interval for both. The relative interval uses the delta method on the log scale, so it stays asymmetric the way ratios are.

## Walkthrough

`examples/walkthrough.py` runs a full simulated experiment: a checkout change on a 3.2% conversion rate, planned for an 8% lift.

![Sample size vs detectable lift](images/sample_size.png)

Halving the detectable effect roughly quadruples the sample, which is why the minimum effect worth detecting should be agreed before the test starts.

![Results](images/results.png)

On the same data, CUPED (using each user's revenue in the four weeks before the test) cuts the variance of revenue per user by about 16% and tightens the interval accordingly.

## Install and test

```bash
pip install -e ".[examples]"
python examples/walkthrough.py
python -m unittest discover -s tests -t .
```

The tests check the sample size against a standard reference value, that the false positive rate stays at alpha under no effect, and that CUPED keeps the mean while reducing variance.

## License

MIT
