import numpy as np
import pytest

from clincausal import weighting as wt
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
    return X, y, treatment, true_ate, propensity_true


def test_ate_weights_sum_reasonably():
    e = np.array([0.5, 0.5, 0.5, 0.5])
    treatment = np.array([1, 1, 0, 0])
    weights = wt.compute_iptw_weights(e, treatment, estimand="ATE", stabilized=False)
    # unstabilized ATE weights at e=0.5 are all 2.0
    assert np.allclose(weights, 2.0)


def test_att_weights_are_one_for_treated():
    e = np.array([0.3, 0.7, 0.3, 0.7])
    treatment = np.array([1, 1, 0, 0])
    weights = wt.compute_iptw_weights(e, treatment, estimand="ATT")
    assert np.allclose(weights[treatment == 1], 1.0)


def test_invalid_estimand_raises():
    with pytest.raises(ValueError):
        wt.compute_iptw_weights(np.array([0.5]), np.array([1]), estimand="bogus")


def test_trim_weights_caps_at_percentile():
    weights = np.array([1, 2, 3, 4, 100])
    trimmed = wt.trim_weights(weights, percentile=80)
    assert trimmed.max() <= np.percentile(weights, 80) + 1e-9
    assert trimmed.max() < weights.max()


def test_weighted_ate_recovers_known_effect_with_oracle_weights():
    X, y, treatment, true_ate, propensity_true = _confounded_data(n=5000, seed=1)
    weights = wt.compute_iptw_weights(propensity_true, treatment, estimand="ATE", stabilized=False)
    ate = wt.weighted_ate(y, treatment, weights)
    assert ate == pytest.approx(true_ate, abs=0.3)


def test_bootstrap_weighted_ate_ci_contains_point_estimate():
    X, y, treatment, _, _ = _confounded_data(n=800, seed=2)
    result = wt.bootstrap_weighted_ate_ci(X, y, treatment, n_boot=100, seed=0)
    assert result["ci_low"] <= result["ate"] <= result["ci_high"]


def test_naive_difference_is_biased_but_iptw_corrects_it():
    X, y, treatment, true_ate, propensity_true = _confounded_data(n=5000, seed=3, true_ate=2.0)
    naive = y[treatment == 1].mean() - y[treatment == 0].mean()

    weights = wt.compute_iptw_weights(propensity_true, treatment, estimand="ATE", stabilized=False)
    adjusted = wt.weighted_ate(y, treatment, weights)

    # naive estimate should be noticeably further from the true ATE than the IPTW-adjusted one
    assert abs(adjusted - true_ate) < abs(naive - true_ate)
