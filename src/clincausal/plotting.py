"""Matplotlib plotting helpers for clincausal (static plots for notebooks/reports)."""

import numpy as np

from . import balance as _balance


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting. Install with: pip install clincausal[plots]") from e


def plot_propensity_overlap(propensity_scores, treatment, ax=None, bins: int = 30):
    """Histogram of propensity scores by treatment group — the visual check for common support."""
    plt = _require_matplotlib()
    propensity_scores = np.asarray(propensity_scores, dtype=float)
    treatment = np.asarray(treatment).astype(int)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    ax.hist(propensity_scores[treatment == 1], bins=bins, alpha=0.6, label="Treated", density=True)
    ax.hist(propensity_scores[treatment == 0], bins=bins, alpha=0.6, label="Control", density=True)
    ax.set_xlabel("Propensity score")
    ax.set_ylabel("Density")
    ax.set_title("Propensity Score Overlap")
    ax.legend()
    return ax


def plot_love(X, treatment, feature_names=None, weights_before=None, weights_after=None, ax=None):
    """Love plot: standardized mean difference per covariate, before vs after adjustment."""
    plt = _require_matplotlib()
    table = _balance.balance_table(X, treatment, feature_names=feature_names,
                                    weights_before=weights_before, weights_after=weights_after)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(table))))

    y_pos = np.arange(len(table))
    ax.scatter(table["smd_before"], y_pos, label="Before", color="#D55E00")
    if "smd_after" in table.columns:
        ax.scatter(table["smd_after"], y_pos, label="After", color="#0072B2")
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.8)
    ax.axvline(0.1, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(-0.1, color="gray", linestyle="--", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(table["feature"])
    ax.set_xlabel("Standardized mean difference")
    ax.set_title("Covariate Balance (Love Plot)")
    ax.legend()
    return ax
