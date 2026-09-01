"""
Propensity score estimation and common-support diagnostics for clincausal.

The propensity score is the probability of receiving treatment given observed
covariates, e(x) = P(T=1 | X=x). It's the workhorse behind every method in this
package: matching pairs similar patients, weighting reweights the sample to look
like a randomized trial, and AIPW combines it with an outcome model for a
doubly-robust estimate.

None of this replaces thinking carefully about which confounders belong in X —
propensity methods only adjust for confounders you actually measured and included.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def _validate(X, treatment):
    X = np.asarray(X, dtype=float)
    treatment = np.asarray(treatment)
    if not np.all((treatment == 0) | (treatment == 1)):
        raise ValueError("treatment must be binary (0/1)")
    if X.shape[0] != treatment.shape[0]:
        raise ValueError(f"X and treatment must have the same number of rows, got {X.shape[0]} and {treatment.shape[0]}")
    return X, treatment.astype(int)


def fit_propensity_scores(X, treatment, model=None) -> dict:
    """
    Fit a propensity score model and return fitted scores.

    model: an unfitted sklearn-compatible classifier with .fit / .predict_proba.
    Defaults to logistic regression, the standard choice unless you have a
    specific reason to use something more flexible (with more flexibility comes
    more risk of extreme, unstable propensity scores near 0 or 1).
    """
    X, treatment = _validate(X, treatment)
    if model is None:
        model = LogisticRegression(max_iter=1000)
    model.fit(X, treatment)
    propensity_scores = model.predict_proba(X)[:, 1]
    return {"model": model, "propensity_scores": propensity_scores}


def clip_propensity_scores(propensity_scores, eps: float = 1e-3) -> np.ndarray:
    """
    Clip propensity scores away from 0/1. Scores near the boundary correspond to
    patients who were (almost) never or always treated given their covariates —
    weighting or matching on them is numerically unstable and often a sign of
    poor common support, not just a rounding issue.
    """
    return np.clip(propensity_scores, eps, 1 - eps)


def check_common_support(propensity_scores, treatment) -> dict:
    """
    Common support = the region of propensity scores where both treated and
    control patients exist, so a comparison is even possible. Patients outside
    it have no comparable counterfactual in the data and should generally be
    excluded rather than extrapolated over.
    """
    propensity_scores = np.asarray(propensity_scores, dtype=float)
    treatment = np.asarray(treatment).astype(int)

    treated_ps = propensity_scores[treatment == 1]
    control_ps = propensity_scores[treatment == 0]

    lower = max(treated_ps.min(), control_ps.min())
    upper = min(treated_ps.max(), control_ps.max())

    in_support = (propensity_scores >= lower) & (propensity_scores <= upper)
    n_outside = int((~in_support).sum())

    return {
        "lower": float(lower),
        "upper": float(upper),
        "in_support": in_support,
        "n_outside_support": n_outside,
        "frac_outside_support": float(n_outside / len(propensity_scores)),
    }


def trim_to_common_support(X, treatment, propensity_scores, y=None):
    """
    Restrict the sample to the common support region. Returns the trimmed
    (X, treatment, propensity_scores[, y]) and how many rows were dropped.
    """
    support = check_common_support(propensity_scores, treatment)
    mask = support["in_support"]

    X = np.asarray(X)[mask]
    treatment = np.asarray(treatment)[mask]
    propensity_scores = np.asarray(propensity_scores)[mask]

    result = {"X": X, "treatment": treatment, "propensity_scores": propensity_scores,
              "n_dropped": support["n_outside_support"]}
    if y is not None:
        result["y"] = np.asarray(y)[mask]
    return result
