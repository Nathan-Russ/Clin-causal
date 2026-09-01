"""
Augmented inverse probability weighting (AIPW) for clincausal — a doubly robust
estimator of the average treatment effect.

AIPW combines a propensity model and an outcome model:

    psi_i = (mu1(x_i) - mu0(x_i))
            + T_i * (y_i - mu1(x_i)) / e(x_i)
            - (1-T_i) * (y_i - mu0(x_i)) / (1 - e(x_i))
    ATE_hat = mean(psi_i)

The "doubly robust" property: this estimator is consistent if EITHER the
propensity model e(x) OR the outcome models mu0/mu1 are correctly specified —
not necessarily both. A pure regression estimator only works if the outcome
model is right; a pure IPTW estimator only works if the propensity model is
right. AIPW gives you two independent chances to get it right, which is why
it's generally preferred when you're not certain either model is well specified
(which, in practice, is always).
"""

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from . import propensity as _prop


def _fit_outcome_model(X, y, model_factory):
    model = model_factory()
    model.fit(X, y)
    if hasattr(model, "predict_proba"):
        return lambda Xnew: model.predict_proba(Xnew)[:, 1]
    return model.predict


def fit_outcome_models(X, y, treatment, model_factory=None) -> dict:
    """
    Fit separate outcome models on the treated and control arms.

    model_factory: a zero-argument callable returning a fresh, unfitted
    sklearn-compatible model, e.g. `lambda: LinearRegression()` for a continuous
    outcome (the default) or `lambda: LogisticRegression()` for a binary one.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment).astype(int)

    if model_factory is None:
        model_factory = lambda: LinearRegression()

    predict_mu1 = _fit_outcome_model(X[treatment == 1], y[treatment == 1], model_factory)
    predict_mu0 = _fit_outcome_model(X[treatment == 0], y[treatment == 0], model_factory)

    mu1 = predict_mu1(X)
    mu0 = predict_mu0(X)
    return {"mu1": mu1, "mu0": mu0}


def aipw_scores(y, treatment, propensity_scores, mu1, mu0) -> np.ndarray:
    """The per-patient AIPW pseudo-outcome (psi_i above) — its mean is the ATE estimate."""
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment).astype(int)
    e = np.asarray(propensity_scores, dtype=float)
    mu1 = np.asarray(mu1, dtype=float)
    mu0 = np.asarray(mu0, dtype=float)

    return (mu1 - mu0) + treatment * (y - mu1) / e - (1 - treatment) * (y - mu0) / (1 - e)


def aipw_ate(X, y, treatment, ps_model=None, outcome_model_factory=None, trim_eps: float = 1e-3) -> float:
    """Point estimate of the ATE via AIPW."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment).astype(int)

    ps = _prop.fit_propensity_scores(X, treatment, model=ps_model)["propensity_scores"]
    ps = _prop.clip_propensity_scores(ps, eps=trim_eps)
    outcome_fit = fit_outcome_models(X, y, treatment, model_factory=outcome_model_factory)

    psi = aipw_scores(y, treatment, ps, outcome_fit["mu1"], outcome_fit["mu0"])
    return float(np.mean(psi))


def bootstrap_aipw_ci(X, y, treatment, ps_model=None, outcome_model_factory=None, trim_eps: float = 1e-3,
                       n_boot: int = 500, ci: float = 0.95, seed: int = None) -> dict:
    """
    Bootstrap CI for the AIPW ATE, refitting both the propensity and outcome
    models on every resample.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment).astype(int)
    n = len(y)

    point_ate = aipw_ate(X, y, treatment, ps_model=ps_model, outcome_model_factory=outcome_model_factory, trim_eps=trim_eps)

    rng = np.random.default_rng(seed)
    boot_ates = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(treatment[idx])) < 2:
            continue
        try:
            boot_ates.append(aipw_ate(X[idx], y[idx], treatment[idx], ps_model=ps_model,
                                       outcome_model_factory=outcome_model_factory, trim_eps=trim_eps))
        except (ValueError, ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
            continue

    boot_ates = np.array(boot_ates)
    alpha = (1 - ci) / 2
    return {
        "ate": point_ate,
        "ci_low": float(np.nanquantile(boot_ates, alpha)),
        "ci_high": float(np.nanquantile(boot_ates, 1 - alpha)),
        "n_boot_used": len(boot_ates),
    }


def compare_estimators(X, y, treatment, ps_model=None, outcome_model_factory=None) -> dict:
    """
    Convenience function computing the naive (unadjusted), regression-only,
    IPTW-only, and AIPW estimates side by side — useful for seeing how much
    confounding adjustment actually changes the answer, and as a sanity check
    that the different approaches roughly agree (large disagreement is itself
    informative: it suggests the estimate is sensitive to modeling choices).
    """
    from . import weighting as _weight

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment).astype(int)

    naive = float(y[treatment == 1].mean() - y[treatment == 0].mean())

    outcome_fit = fit_outcome_models(X, y, treatment, model_factory=outcome_model_factory)
    regression_only = float(np.mean(outcome_fit["mu1"] - outcome_fit["mu0"]))

    ps = _prop.fit_propensity_scores(X, treatment, model=ps_model)["propensity_scores"]
    ps = _prop.clip_propensity_scores(ps)
    weights = _weight.compute_iptw_weights(ps, treatment, estimand="ATE")
    iptw_only = _weight.weighted_ate(y, treatment, weights)

    doubly_robust = aipw_ate(X, y, treatment, ps_model=ps_model, outcome_model_factory=outcome_model_factory)

    return {
        "naive_difference": naive,
        "regression_only": regression_only,
        "iptw_only": iptw_only,
        "aipw": doubly_robust,
    }
