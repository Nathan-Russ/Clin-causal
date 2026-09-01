"""
clincausal — propensity score matching, inverse probability weighting, and
doubly robust (AIPW) estimation of treatment effects from observational data.

Quickstart:

    import clincausal as cc

    ps = cc.propensity.fit_propensity_scores(X, treatment)["propensity_scores"]

    # Matching
    match = cc.matching.match_on_propensity(ps, treatment)
    cc.matching.att_from_matching(y, match)

    # Weighting
    weights = cc.weighting.compute_iptw_weights(ps, treatment, estimand="ATE")
    cc.weighting.weighted_ate(y, treatment, weights)

    # Doubly robust
    cc.estimation.aipw_ate(X, y, treatment)
    cc.estimation.compare_estimators(X, y, treatment)

    # Balance diagnostics
    cc.balance.balance_table(X, treatment, weights_after=weights)
"""

from . import propensity
from . import balance
from . import matching
from . import weighting
from . import estimation

try:
    from . import plotting  # optional: requires matplotlib
except ImportError:
    plotting = None

__version__ = "0.1.0"

__all__ = ["propensity", "balance", "matching", "weighting", "estimation", "plotting", "__version__"]
