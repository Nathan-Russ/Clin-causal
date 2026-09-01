"""
Covariate balance diagnostics for clincausal.

The whole point of matching or weighting is to make the treated and control
groups look alike on measured covariates — as similar as they'd be if treatment
had been randomly assigned. Balance diagnostics are how you check whether that
actually happened, rather than just assuming it did because you ran a method
with a respectable name.
"""

import numpy as np
import pandas as pd


def standardized_mean_difference(X, treatment, weights=None) -> np.ndarray:
    """
    Standardized mean difference (SMD) per covariate: the difference in means
    between treated and control, divided by the pooled standard deviation.

    A common rule of thumb: |SMD| < 0.1 indicates good balance on that covariate;
    |SMD| > 0.25 indicates a covariate that still meaningfully differs between
    groups after adjustment. There's nothing magic about these cutoffs — they're
    a starting point for judgment, not a pass/fail test.

    weights: optional per-patient weights (e.g. IPTW weights) for computing
    weighted means/variances — use this to check post-weighting balance.
    """
    X = np.asarray(X, dtype=float)
    treatment = np.asarray(treatment).astype(int)
    n_features = X.shape[1]

    if weights is None:
        weights = np.ones(len(treatment))
    weights = np.asarray(weights, dtype=float)

    smds = np.empty(n_features)
    for j in range(n_features):
        x = X[:, j]
        w_t, w_c = weights[treatment == 1], weights[treatment == 0]
        x_t, x_c = x[treatment == 1], x[treatment == 0]

        mean_t = np.average(x_t, weights=w_t)
        mean_c = np.average(x_c, weights=w_c)
        var_t = np.average((x_t - mean_t) ** 2, weights=w_t)
        var_c = np.average((x_c - mean_c) ** 2, weights=w_c)
        pooled_sd = np.sqrt((var_t + var_c) / 2)

        smds[j] = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0.0

    return smds


def balance_table(X, treatment, feature_names=None, weights_before=None, weights_after=None) -> pd.DataFrame:
    """
    SMD before and after adjustment, side by side, for every covariate — the
    numbers behind a love plot. Pass `weights_after` for weighting-based
    adjustment, or pass an already-matched/subset X as `X` with weights_after=None
    if you're checking balance after matching instead.
    """
    X = np.asarray(X, dtype=float)
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    smd_before = standardized_mean_difference(X, treatment, weights=weights_before)
    result = pd.DataFrame({"feature": feature_names, "smd_before": smd_before})

    if weights_after is not None:
        smd_after = standardized_mean_difference(X, treatment, weights=weights_after)
        result["smd_after"] = smd_after

    return result.reindex(result["smd_before"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def max_absolute_smd(X, treatment, weights=None) -> float:
    """The worst-balanced covariate's |SMD| — a quick one-number balance summary."""
    return float(np.max(np.abs(standardized_mean_difference(X, treatment, weights=weights))))
