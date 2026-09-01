import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from clincausal import estimation as est
from clincausal import propensity as prop


def _confounded_data(n=2000, seed=0, true_ate=2.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    propensity_true = 1 / (1 + np.exp(-(0.8 * x)))
    treatment = rng.binomial(1, propensity_true)

    noise = rng.normal(0, 1, n)
    y0 = 1.0 + 1.5 * x + noise
    y1 = y0 + true_ate
    y = np.where(treatment == 1, y1, y0)

    X = x.reshape(-1, 1)
    return X, y, treatment, true_ate


def _double_robustness_data(n=20000, seed=0, true_ate=3.0):
    """
    Two covariates. Treatment assignment depends on BOTH x1 and x2 (so a
    propensity model omitting x2 is genuinely misspecified). The outcome
    depends on x1 linearly but on x2 THROUGH A CUBIC TERM (so an outcome
    model using x2 linearly, without the cubic term, is genuinely
    misspecified).

    The cubic (not squared) term matters for making this a *visible* demo:
    squaring an even function of a variable whose mean gets shifted by
    selection roughly conserves E[x2^2] between arms (variance shrinks as the
    mean shifts, largely canceling out) — so a squared nonlinearity barely
    biases anything in practice, which makes for an unconvincing test. A cubic
    term doesn't have that cancellation, and confounding shows up clearly.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)

    logit_e = 0.4 * x1 + 1.0 * x2
    e_true = 1 / (1 + np.exp(-logit_e))
    treatment = rng.binomial(1, e_true)

    noise = rng.normal(0, 0.5, n)
    y0 = 2.0 + 1.0 * x1 + 1.2 * (x2 ** 3) + noise
    y1 = y0 + true_ate
    y = np.where(treatment == 1, y1, y0)

    return x1, x2, y, treatment, true_ate


# ---------------------------------------------------------------------------
# Basic AIPW correctness (both models correctly specified)
# ---------------------------------------------------------------------------

def test_aipw_recovers_known_ate_when_both_models_correct():
    X, y, treatment, true_ate = _confounded_data(n=4000, seed=1)
    ate = est.aipw_ate(X, y, treatment)
    assert ate == pytest.approx(true_ate, abs=0.3)


def test_bootstrap_aipw_ci_contains_point_estimate():
    X, y, treatment, _ = _confounded_data(n=800, seed=2)
    result = est.bootstrap_aipw_ci(X, y, treatment, n_boot=100, seed=0)
    assert result["ci_low"] <= result["ate"] <= result["ci_high"]


def test_compare_estimators_returns_all_four_keys():
    X, y, treatment, _ = _confounded_data(n=1000, seed=3)
    result = est.compare_estimators(X, y, treatment)
    assert set(["naive_difference", "regression_only", "iptw_only", "aipw"]) == set(result.keys())


# ---------------------------------------------------------------------------
# The double robustness property: AIPW survives ONE misspecified model
# ---------------------------------------------------------------------------

def test_aipw_survives_misspecified_outcome_model_if_propensity_correct():
    x1, x2, y, treatment, true_ate = _double_robustness_data(seed=4)

    X_ps_correct = np.column_stack([x1, x2])           # propensity: correctly includes both drivers
    X_outcome_misspec = np.column_stack([x1, x2])       # outcome: MISSING the x2^3 term -> misspecified

    ps = prop.fit_propensity_scores(X_ps_correct, treatment)["propensity_scores"]
    ps = prop.clip_propensity_scores(ps)
    outcome_fit = est.fit_outcome_models(X_outcome_misspec, y, treatment)
    psi = est.aipw_scores(y, treatment, ps, outcome_fit["mu1"], outcome_fit["mu0"])
    aipw_estimate = np.mean(psi)

    regression_only = np.mean(outcome_fit["mu1"] - outcome_fit["mu0"])

    # the misspecified outcome model should show a real, visible bias here...
    assert abs(regression_only - true_ate) > 0.15
    # ...while AIPW, rescued by the correctly specified propensity model, stays close
    assert abs(aipw_estimate - true_ate) < 0.15
    assert abs(aipw_estimate - true_ate) < abs(regression_only - true_ate)


def test_aipw_survives_misspecified_propensity_model_if_outcome_correct():
    x1, x2, y, treatment, true_ate = _double_robustness_data(seed=5)

    X_ps_misspec = x1.reshape(-1, 1)                              # propensity: MISSING x2 -> misspecified
    X_outcome_correct = np.column_stack([x1, x2, x2 ** 3])         # outcome: correctly includes x2^3

    ps = prop.fit_propensity_scores(X_ps_misspec, treatment)["propensity_scores"]
    ps = prop.clip_propensity_scores(ps)
    outcome_fit = est.fit_outcome_models(X_outcome_correct, y, treatment)
    psi = est.aipw_scores(y, treatment, ps, outcome_fit["mu1"], outcome_fit["mu0"])
    aipw_estimate = np.mean(psi)

    from clincausal import weighting as wt
    weights = wt.compute_iptw_weights(ps, treatment, estimand="ATE")
    iptw_only = wt.weighted_ate(y, treatment, weights)

    # the misspecified propensity model should show a LARGE bias here (omitting
    # a strong confounder entirely is a much bigger problem than a wrong functional
    # form) — IPTW-only badly overshoots the true effect...
    assert abs(iptw_only - true_ate) > 1.0
    # ...while AIPW, rescued by the correctly specified outcome model, stays close
    assert abs(aipw_estimate - true_ate) < 0.15
    assert abs(aipw_estimate - true_ate) < abs(iptw_only - true_ate)
