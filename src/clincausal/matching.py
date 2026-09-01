"""
Propensity score matching for clincausal.

Matches each treated patient to their nearest control(s) on the logit of the
propensity score — matching on the logit rather than the raw score is standard
practice (Austin, 2011), since propensity scores are compressed near 0 and 1
where a fixed caliper behaves very differently than in the middle of the range.

This is a greedy nearest-neighbor matcher: fast and simple to reason about, at
the cost of not being globally optimal (an "optimal matching" algorithm can do
marginally better on average match quality). For most applied purposes greedy
matching with a sensible caliper performs similarly well in practice.
"""

import numpy as np
import pandas as pd

from . import balance as _balance


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def match_on_propensity(propensity_scores, treatment, caliper: float = None,
                         ratio: int = 1, replace: bool = False, seed: int = None) -> dict:
    """
    Greedy nearest-neighbor matching on the logit propensity score.

    caliper: maximum allowed distance in logit(propensity) units between a
    matched pair. If None, defaults to 0.2 * SD(logit(propensity)) — Austin's
    commonly cited rule of thumb. Pass caliper=np.inf for no caliper at all.
    ratio: number of controls to match to each treated patient (k:1 matching).
    replace: whether a control can be reused across multiple treated patients.

    Returns a dict with `treated_idx` and `control_idx` (parallel arrays — every
    matched pair is one entry in each), and `n_unmatched` treated patients that
    couldn't find a control within the caliper.
    """
    propensity_scores = np.asarray(propensity_scores, dtype=float)
    treatment = np.asarray(treatment).astype(int)
    logit_ps = _logit(propensity_scores)

    if caliper is None:
        caliper = 0.2 * np.std(logit_ps)

    treated_positions = np.where(treatment == 1)[0]
    control_positions = np.where(treatment == 0)[0]

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(treated_positions))  # randomize match order to avoid systematic bias

    control_logit = logit_ps[control_positions]
    sort_order = np.argsort(control_logit)
    sorted_control_positions = control_positions[sort_order]
    sorted_control_logit = control_logit[sort_order]
    available = np.ones(len(sorted_control_positions), dtype=bool)

    matched_treated, matched_control = [], []
    n_unmatched = 0

    for t_pos_idx in order:
        t_pos = treated_positions[t_pos_idx]
        t_logit = logit_ps[t_pos]

        matches_found = 0
        for _ in range(ratio):
            avail_idx = np.where(available)[0] if not replace else np.arange(len(sorted_control_logit))
            if len(avail_idx) == 0:
                break
            candidate_logit = sorted_control_logit[avail_idx]
            nearest_local = avail_idx[np.argmin(np.abs(candidate_logit - t_logit))]
            distance = abs(sorted_control_logit[nearest_local] - t_logit)

            if distance > caliper:
                break

            matched_treated.append(t_pos)
            matched_control.append(sorted_control_positions[nearest_local])
            matches_found += 1
            if not replace:
                available[nearest_local] = False

        if matches_found == 0:
            n_unmatched += 1

    return {
        "treated_idx": np.array(matched_treated, dtype=int),
        "control_idx": np.array(matched_control, dtype=int),
        "n_unmatched": n_unmatched,
        "n_treated_total": len(treated_positions),
        "caliper_used": float(caliper),
    }


def att_from_matching(y, match_result: dict) -> dict:
    """
    Average treatment effect on the treated (ATT) from a matched sample —
    the mean outcome difference within matched pairs. With k:1 matching, each
    treated patient's control matches are averaged first so each treated
    patient contributes equally regardless of how many controls they matched to.
    """
    y = np.asarray(y, dtype=float)
    treated_idx = match_result["treated_idx"]
    control_idx = match_result["control_idx"]

    if len(treated_idx) == 0:
        raise ValueError("No matched pairs to estimate from.")

    df = pd.DataFrame({"treated_idx": treated_idx, "y_control": y[control_idx]})
    mean_control_per_treated = df.groupby("treated_idx")["y_control"].mean()
    y_treated_unique = pd.Series(y[mean_control_per_treated.index.values], index=mean_control_per_treated.index)

    pair_diffs = y_treated_unique - mean_control_per_treated
    att = float(pair_diffs.mean())
    return {"att": att, "n_pairs": len(pair_diffs), "pair_differences": pair_diffs.values}


def bootstrap_att_ci(X, y, treatment, ps_model=None, caliper=None, ratio=1, replace=False,
                      n_boot: int = 500, ci: float = 0.95, seed: int = None) -> dict:
    """
    Bootstrap CI for the matched ATT, refitting the propensity model and
    rematching on every resample — this is the honest approach, since it
    accounts for uncertainty in the propensity model itself, not just in
    which patients happened to be sampled.
    """
    from . import propensity as _prop

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    treatment = np.asarray(treatment).astype(int)
    n = len(y)

    point_ps = _prop.fit_propensity_scores(X, treatment, model=ps_model)["propensity_scores"]
    point_match = match_on_propensity(point_ps, treatment, caliper=caliper, ratio=ratio, replace=replace, seed=seed)
    point_att = att_from_matching(y, point_match)["att"]

    rng = np.random.default_rng(seed)
    boot_atts = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(treatment[idx])) < 2:
            continue
        try:
            boot_ps = _prop.fit_propensity_scores(X[idx], treatment[idx], model=ps_model)["propensity_scores"]
            boot_match = match_on_propensity(boot_ps, treatment[idx], caliper=caliper, ratio=ratio, replace=replace, seed=b)
            if len(boot_match["treated_idx"]) < 2:
                continue
            boot_atts.append(att_from_matching(y[idx], boot_match)["att"])
        except (ValueError, ZeroDivisionError):
            continue

    boot_atts = np.array(boot_atts)
    alpha = (1 - ci) / 2
    return {
        "att": point_att,
        "ci_low": float(np.nanquantile(boot_atts, alpha)),
        "ci_high": float(np.nanquantile(boot_atts, 1 - alpha)),
        "n_boot_used": len(boot_atts),
    }
