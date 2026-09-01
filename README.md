# ⚖️ clincausal

Propensity score matching, inverse probability weighting (IPTW), and doubly
robust (AIPW) estimation of treatment effects from observational clinical
data — as a small, tested, pip-installable Python library, plus an interactive
Streamlit app to explore results without writing any code.

A prediction model answers "what will happen to this patient?" A causal
inference method answers a different question: "what would happen to this
patient under a different treatment?" Prediction models don't answer that on
their own — `clincausal` is for the second question.


## What's included

- **Propensity scores**: fit a propensity model, check and enforce common
  support, clip extreme scores
- **Matching**: greedy nearest-neighbor matching on the logit propensity score
  (with caliper, k:1 matching, with/without replacement), ATT estimation with
  bootstrap confidence intervals
- **Weighting (IPTW)**: ATE and ATT weights, stabilization, weight trimming,
  Hájek-normalized weighted effect estimation with bootstrap CIs
- **Doubly robust estimation (AIPW)**: combines a propensity model and an
  outcome model — consistent if *either* is correctly specified, not
  necessarily both. Includes a `compare_estimators` function showing the
  naive, regression-only, IPTW-only, and AIPW estimates side by side.
- **Balance diagnostics**: standardized mean differences before/after
  adjustment, love plot data
- **Streamlit demo app**: the classic Hernán & Robins NHEFS dataset (does
  quitting smoking cause weight gain?) built in, or upload your own data

## Install the library

```bash
pip install -e ".[plots]"
```

(Not yet published to PyPI:
URL: `pip install git+https://github.com/<your-username>/clincausal.git`)

## Quickstart

```python
import clincausal as cc

ps = cc.propensity.fit_propensity_scores(X, treatment)["propensity_scores"]

# Matching
match = cc.matching.match_on_propensity(ps, treatment, caliper=0.2)
cc.matching.att_from_matching(y, match)
# {'att': 2.1, 'n_pairs': 340, ...}

# Weighting
weights = cc.weighting.compute_iptw_weights(ps, treatment, estimand="ATE")
cc.weighting.weighted_ate(y, treatment, weights)

# Doubly robust — the recommended default when you're not certain either
# the propensity model or the outcome model is exactly right
cc.estimation.aipw_ate(X, y, treatment)
cc.estimation.compare_estimators(X, y, treatment)
# {'naive_difference': 2.5, 'regression_only': 3.4, 'iptw_only': 3.3, 'aipw': 3.3}

# Balance diagnostics
cc.balance.balance_table(X, treatment, weights_after=weights)
```

## Run the interactive demo app locally

```bash
pip install -r requirements.txt
streamlit run demo_app/app.py
```

The demo uses the NHEFS dataset (via the `causaldata` package, bundled — no
network access needed): does quitting smoking cause weight gain? The naive
comparison gives ~2.5 kg; matching, weighting, and AIPW all converge around
~3.3 kg, consistent with the published estimate in Hernán & Robins,
*Causal Inference: What If*.


## Run the tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The test suite includes a direct demonstration of the double robustness
property: two scenarios, each with one model (propensity or outcome)
deliberately misspecified, showing that the single-model estimator (IPTW-only
or regression-only) is biased while AIPW stays close to the true effect in
both.

## Project structure

```
src/clincausal/
  propensity.py    # propensity score fitting, common support, clipping
  balance.py       # standardized mean differences, balance tables
  matching.py      # greedy nearest-neighbor matching, ATT estimation
  weighting.py     # IPTW weights, weighted ATE/ATT estimation
  estimation.py    # AIPW (doubly robust) estimation, estimator comparison
  plotting.py       # optional matplotlib plots (love plot, overlap histogram)
tests/              # pytest unit tests, including the double-robustness demo
demo_app/
  app.py            # Streamlit app (interactive Plotly plots)
  make_demo_data.py # prepares the NHEFS demo dataset
  sample_data/       # the generated demo CSV
```

## A note on interpretation

- **Propensity methods only adjust for confounders you actually measured and
  included.** No method here can correct for an unmeasured confounder — that
  requires either believing there isn't an important one, or a different
  design (instrumental variables, sensitivity analysis, etc.).
- **Check common support before trusting an estimate.** A large fraction of
  patients outside the common support region means you're extrapolating, not
  interpolating, for part of the sample.
- **Always look at the balance table, not just the point estimate.** A
  method that doesn't visibly improve covariate balance hasn't done its job,
  regardless of what number comes out the other end.
- **AIPW is a hedge, not a guarantee.** It protects you if *one* of your two
  models is right — if both are badly wrong, no method here can save you.
- This tool is meant for exploration and teaching, not as a replacement for
  careful causal study design and a validated analysis pipeline when
  preparing results for publication.

## License

MIT — see [LICENSE](LICENSE).
