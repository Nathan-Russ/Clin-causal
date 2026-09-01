import numpy as np
import pytest

from clincausal import matching as match
from clincausal import propensity as prop
from clincausal import balance as bal


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


def test_match_on_propensity_respects_caliper():
    X, y, treatment, _ = _confounded_data(n=1000, seed=1)
    ps = prop.fit_propensity_scores(X, treatment)["propensity_scores"]
    caliper = 0.1
    result = match.match_on_propensity(ps, treatment, caliper=caliper, seed=0)

    logit_ps = match._logit(ps)
    distances = np.abs(logit_ps[result["treated_idx"]] - logit_ps[result["control_idx"]])
    assert np.all(distances <= caliper + 1e-9)


def test_match_on_propensity_without_replacement_uses_each_control_once():
    X, y, treatment, _ = _confounded_data(n=500, seed=2)
    ps = prop.fit_propensity_scores(X, treatment)["propensity_scores"]
    result = match.match_on_propensity(ps, treatment, replace=False, seed=0)
    control_idx = result["control_idx"]
    assert len(control_idx) == len(np.unique(control_idx))


def test_match_on_propensity_with_replacement_can_reuse_controls():
    # Deliberately create far more treated than control patients, so reuse is required
    rng = np.random.default_rng(3)
    n_treated, n_control = 200, 20
    ps_treated = rng.uniform(0.4, 0.6, n_treated)
    ps_control = rng.uniform(0.4, 0.6, n_control)
    ps = np.concatenate([ps_treated, ps_control])
    treatment = np.concatenate([np.ones(n_treated), np.zeros(n_control)]).astype(int)

    result = match.match_on_propensity(ps, treatment, replace=True, caliper=np.inf, seed=0)
    assert len(result["control_idx"]) > n_control  # more matches than unique controls -> reuse happened


def test_att_from_matching_recovers_known_ate_reasonably():
    X, y, treatment, true_ate = _confounded_data(n=3000, seed=4, true_ate=2.0)
    ps = prop.fit_propensity_scores(X, treatment)["propensity_scores"]
    result = match.match_on_propensity(ps, treatment, seed=0)
    att = match.att_from_matching(y, result)
    assert att["att"] == pytest.approx(true_ate, abs=0.5)


def test_matching_improves_covariate_balance():
    X, y, treatment, _ = _confounded_data(n=2000, seed=5)
    ps = prop.fit_propensity_scores(X, treatment)["propensity_scores"]
    smd_before = bal.max_absolute_smd(X, treatment)

    result = match.match_on_propensity(ps, treatment, seed=0)
    matched_rows = np.concatenate([result["treated_idx"], result["control_idx"]])
    matched_treatment = np.concatenate([np.ones(len(result["treated_idx"])), np.zeros(len(result["control_idx"]))])
    smd_after = bal.max_absolute_smd(X[matched_rows], matched_treatment)

    assert smd_after < smd_before


def test_att_from_matching_raises_on_no_matches():
    with pytest.raises(ValueError):
        match.att_from_matching(np.array([1.0, 2.0]), {"treated_idx": np.array([]), "control_idx": np.array([])})


def test_bootstrap_att_ci_contains_point_estimate():
    X, y, treatment, _ = _confounded_data(n=800, seed=6)
    result = match.bootstrap_att_ci(X, y, treatment, n_boot=100, seed=0)
    assert result["ci_low"] <= result["att"] <= result["ci_high"]
