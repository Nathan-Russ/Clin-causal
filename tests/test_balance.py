import numpy as np
import pytest

from clincausal import balance as bal


def test_smd_is_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1000)
    treatment = rng.binomial(1, 0.5, 1000)  # random assignment, independent of x
    X = x.reshape(-1, 1)
    smd = bal.standardized_mean_difference(X, treatment)
    assert abs(smd[0]) < 0.15  # should be small (not exactly zero due to sampling noise)


def test_smd_is_large_for_strongly_confounded_covariate():
    rng = np.random.default_rng(1)
    n = 1000
    treatment = rng.binomial(1, 0.5, n)
    # covariate is shifted substantially between groups
    x = rng.normal(0, 1, n) + treatment * 2.0
    X = x.reshape(-1, 1)
    smd = bal.standardized_mean_difference(X, treatment)
    assert abs(smd[0]) > 1.0


def test_weighted_smd_reduces_imbalance():
    rng = np.random.default_rng(2)
    n = 2000
    x = rng.normal(0, 1, n)
    propensity_true = 1 / (1 + np.exp(-(0.9 * x)))
    treatment = rng.binomial(1, propensity_true)
    X = x.reshape(-1, 1)

    smd_before = bal.standardized_mean_difference(X, treatment)

    # perfect (oracle) IPTW weights using the TRUE propensity
    weights = treatment / propensity_true + (1 - treatment) / (1 - propensity_true)
    smd_after = bal.standardized_mean_difference(X, treatment, weights=weights)

    assert abs(smd_after[0]) < abs(smd_before[0])


def test_balance_table_sorted_by_absolute_smd_before():
    X = np.column_stack([
        np.array([0, 0, 0, 5, 5, 5], dtype=float),   # large imbalance
        np.array([1, 2, 1, 1, 2, 1], dtype=float),   # small imbalance
    ])
    treatment = np.array([0, 0, 0, 1, 1, 1])
    table = bal.balance_table(X, treatment, feature_names=["big_gap", "small_gap"])
    assert table.iloc[0]["feature"] == "big_gap"


def test_max_absolute_smd_matches_manual_max():
    rng = np.random.default_rng(3)
    n = 500
    X = rng.normal(0, 1, (n, 3))
    treatment = rng.binomial(1, 0.5, n)
    smds = bal.standardized_mean_difference(X, treatment)
    assert bal.max_absolute_smd(X, treatment) == pytest.approx(np.max(np.abs(smds)))
