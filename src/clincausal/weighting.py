"""
Inverse probability of treatment weighting (IPTW) for clincausal.

Instead of discarding unmatched patients, IPTW reweights every patient so the
weighted sample looks like a randomized trial: patients who got the treatment
they were unlikely to get (given their covariates) count more, since they're
"standing in" for the many similar patients who didn't get it.
"""

import numpy as np


def compute_iptw_weights(propensity_scores, treatment, estimand: str = "ATE", stabilized: bool = True) -> np.ndarray:
    """
    IPTW weights for estimating the average treatment effect (ATE) or the
    average treatment effect on the treated (ATT).

    ATE weights: w = T/e + (1-T)/(1-e) — reweights everyone to represent the
    full population under both treatment conditions.
    ATT weights: w = T + (1-T) * e/(1-e) — treated patients keep weight 1;
    controls are reweighted to resemble the treated population's covariate
    distribution.

    stabilized: multiply by the marginal probability of each treatment level,
    which doesn't change the point estimate but reduces variance from very
    large weights — recommended by default (Robins, Hernán & Brumback, 2000).
    """
    e = np.asarray(propensity_scores, dtype=float)
    treatment = np.asarray(treatment).astype(int)

    if estimand == "ATE":
        weights = treatment / e + (1 - treatment) / (1 - e)
        if stabilized:
            marginal_treated = treatment.mean()
            marginal_control = 1 - marginal_treated
            weights = treatment * marginal_treated / e + (1 - treatment) * marginal_control / (1 - e)
    elif estimand == "ATT":
        weights = treatment + (1 - treatment) * e / (1 - e)
    else:
        raise ValueError("estimand must be 'ATE' or 'ATT'")

    return weights


def trim_weights(weights, percentile: float = 99.0) -> np.ndarray:
    """
    Cap weights at the given percentile (winsorizing, not dropping) to limit
    the influence of a small number of extreme-propensity patients on the
    estimate. There's a real bias/variance trade-off here — trimming reduces
    variance at the cost of some bias if those extreme patients matter.
    """
    weights = np.asarray(weights, dtype=float)
    cap = np.percentile(weights, percentile)
    return np.minimum(weights, cap)


def weighted_ate(y, treatment, weights) -> float:
    """
    Hájek-normalized weighted difference in means — the standard IPTW point
    estimate. Normalizing by the sum of weights in each arm (rather than by n)
    is what makes this well-behaved with stabilized weights.
    """
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment).astype(int)
    weights = np.asarray(weights, dtype=float)

    w_treated, w_control = weights[treatment == 1], weights[treatment == 0]
    y_treated, y_control = y[treatment == 1], y[treatment == 0]

    mean_treated = np.average(y_treated, weights=w_treated)
    mean_control = np.average(y_control, weights=w_control)
    return float(mean_treated - mean_control)


def bootstrap_weighted_ate_ci(X, y, treatment, ps_model=None, estimand: str = "ATE", stabilized: bool = True,
                               trim_percentile: float = None, n_boot: int = 500, ci: float = 0.95,
                               seed: int = None) -> dict:
    """
    Bootstrap CI for the IPTW ATE/ATT, refitting the propensity model on every
    resample so the CI reflects uncertainty in the propensity model too.
    """
    from . import propensity as _prop

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment).astype(int)
    n = len(y)

    def _point_estimate(Xi, yi, ti):
        ps = _prop.fit_propensity_scores(Xi, ti, model=ps_model)["propensity_scores"]
        ps = _prop.clip_propensity_scores(ps)
        w = compute_iptw_weights(ps, ti, estimand=estimand, stabilized=stabilized)
        if trim_percentile is not None:
            w = trim_weights(w, percentile=trim_percentile)
        return weighted_ate(yi, ti, w)

    point_ate = _point_estimate(X, y, treatment)

    rng = np.random.default_rng(seed)
    boot_ates = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(treatment[idx])) < 2:
            continue
        try:
            boot_ates.append(_point_estimate(X[idx], y[idx], treatment[idx]))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            continue

    boot_ates = np.array(boot_ates)
    alpha = (1 - ci) / 2
    return {
        "ate": point_ate,
        "ci_low": float(np.nanquantile(boot_ates, alpha)),
        "ci_high": float(np.nanquantile(boot_ates, 1 - alpha)),
        "n_boot_used": len(boot_ates),
    }
