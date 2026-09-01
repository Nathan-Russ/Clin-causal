import numpy as np
import pytest

from clincausal import propensity as prop


def _confounded_data(n=2000, seed=0):
    """
    Standard synthetic setup: a single confounder x affects both treatment
    assignment (via a logistic propensity model) and the outcome, with a known
    constant treatment effect `true_ate` added on top.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    true_ate = 2.0

    propensity_true = 1 / (1 + np.exp(-(0.8 * x)))
    treatment = rng.binomial(1, propensity_true)

    noise = rng.normal(0, 1, n)
    y0 = 1.0 + 1.5 * x + noise
    y1 = y0 + true_ate
    y = np.where(treatment == 1, y1, y0)

    X = x.reshape(-1, 1)
    return X, y, treatment, true_ate


def test_fit_propensity_scores_returns_values_in_unit_interval():
    X, y, treatment, _ = _confounded_data()
    result = prop.fit_propensity_scores(X, treatment)
    assert np.all(result["propensity_scores"] > 0)
    assert np.all(result["propensity_scores"] < 1)


def test_fit_propensity_scores_recovers_reasonable_separation():
    X, y, treatment, _ = _confounded_data(n=3000)
    result = prop.fit_propensity_scores(X, treatment)
    ps = result["propensity_scores"]
    # treated patients should have systematically higher fitted propensity
    assert ps[treatment == 1].mean() > ps[treatment == 0].mean()


def test_clip_propensity_scores_bounds_away_from_zero_and_one():
    ps = np.array([0.0, 0.0001, 0.5, 0.9999, 1.0])
    clipped = prop.clip_propensity_scores(ps, eps=1e-3)
    assert np.all(clipped >= 1e-3)
    assert np.all(clipped <= 1 - 1e-3)


def test_check_common_support_identifies_overlap_region():
    treatment = np.array([1, 1, 1, 0, 0, 0])
    ps = np.array([0.6, 0.7, 0.9, 0.1, 0.3, 0.5])
    result = prop.check_common_support(ps, treatment)
    # overlap = [max(min_treated=0.6, min_control=0.1), min(max_treated=0.9, max_control=0.5)]
    assert result["lower"] == pytest.approx(0.6)
    assert result["upper"] == pytest.approx(0.5)
    # since lower > upper here, everything is technically outside a valid overlap window
    assert result["n_outside_support"] == 6


def test_check_common_support_with_real_overlap():
    treatment = np.array([1, 1, 1, 0, 0, 0])
    ps = np.array([0.3, 0.5, 0.7, 0.2, 0.4, 0.6])
    result = prop.check_common_support(ps, treatment)
    assert result["lower"] < result["upper"]
    assert result["n_outside_support"] < 6


def test_trim_to_common_support_drops_expected_rows():
    X, y, treatment, _ = _confounded_data(n=500)
    ps = prop.fit_propensity_scores(X, treatment)["propensity_scores"]
    trimmed = prop.trim_to_common_support(X, treatment, ps, y=y)
    assert len(trimmed["X"]) == len(treatment) - trimmed["n_dropped"]
    assert len(trimmed["y"]) == len(trimmed["X"])


def test_validate_rejects_non_binary_treatment():
    with pytest.raises(ValueError):
        prop.fit_propensity_scores(np.zeros((5, 1)), [0, 1, 2, 0, 1])
