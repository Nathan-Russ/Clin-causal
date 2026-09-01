"""
clincausal demo app — propensity score matching, IPTW, and doubly robust (AIPW)
treatment effect estimation.

Run locally:  streamlit run demo_app/app.py
Deploy free:  push to GitHub, then deploy on share.streamlit.io
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import clincausal as cc  # noqa: E402

st.set_page_config(page_title="clincausal — Treatment Effect Estimation", page_icon="⚖️", layout="wide")

DEMO_PATH = Path(__file__).resolve().parent / "sample_data" / "nhefs_demo.csv"
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

st.title("⚖️ clincausal — Treatment Effect Estimation")
st.caption(
    "Propensity score matching, inverse probability weighting, and doubly robust (AIPW) "
    "estimation for treatment effects from observational data."
)

# ---------------------------------------------------------------------------
# Sidebar — data loading
# ---------------------------------------------------------------------------
st.sidebar.title("⚖️ clincausal")
source = st.sidebar.radio("Data source", ["Try the demo dataset (NHEFS)", "Upload my own"], index=0)

df = None
if source.startswith("Try the demo"):
    df = pd.read_csv(DEMO_PATH)
    st.sidebar.success(f"Loaded NHEFS: {len(df):,} patients")
    st.sidebar.caption(
        "The classic Hernán & Robins teaching dataset: does quitting smoking "
        "(`treatment`) cause weight gain (`outcome`, kg change 1971-1982)? "
        "Confounded by age, sex, smoking history, exercise, and more."
    )
else:
    up = st.sidebar.file_uploader(
        "CSV with a `treatment` column (0/1), an outcome column, and covariates",
        type=["csv"],
    )
    if up is not None:
        df = pd.read_csv(up)
        st.sidebar.success(f"Loaded: {len(df):,} rows")

if df is None:
    st.info("👈 Load the demo dataset from the sidebar, or upload your own.")
    st.markdown("""
    #### Expected CSV format
    - A **treatment** column (0/1)
    - An **outcome** column (numeric)
    - Numeric **covariates** — confounders that plausibly affect both treatment
      assignment and the outcome
    """)
    st.stop()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
treatment_candidates = [c for c in numeric_cols if set(df[c].dropna().unique()).issubset({0, 1})]

st.sidebar.header("Columns")
default_treatment = "treatment" if "treatment" in treatment_candidates else treatment_candidates[0]
treatment_col = st.sidebar.selectbox("Treatment column (0/1)", treatment_candidates,
                                      index=treatment_candidates.index(default_treatment))
outcome_candidates = [c for c in numeric_cols if c != treatment_col]
default_outcome = "outcome" if "outcome" in outcome_candidates else outcome_candidates[0]
outcome_col = st.sidebar.selectbox("Outcome column", outcome_candidates, index=outcome_candidates.index(default_outcome))

covariate_candidates = [c for c in numeric_cols if c not in (treatment_col, outcome_col)]
covariates = st.sidebar.multiselect("Covariates (confounders to adjust for)", covariate_candidates,
                                     default=covariate_candidates)
if not covariates:
    st.warning("Select at least one covariate.")
    st.stop()

X = df[covariates].values.astype(float)
y = df[outcome_col].values.astype(float)
treatment = df[treatment_col].values.astype(int)

with st.spinner("Fitting propensity model..."):
    ps_result = cc.propensity.fit_propensity_scores(X, treatment)
    ps = cc.propensity.clip_propensity_scores(ps_result["propensity_scores"])

st.header("Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients", f"{len(df):,}")
c2.metric("Treated", f"{treatment.sum():,} ({treatment.mean():.1%})")
c3.metric("Naive difference", f"{y[treatment==1].mean() - y[treatment==0].mean():.3f}")
support = cc.propensity.check_common_support(ps, treatment)
c4.metric("Outside common support", f"{support['n_outside_support']} ({support['frac_outside_support']:.1%})")

tab_overlap, tab_match, tab_weight, tab_dr, tab_export = st.tabs(
    ["Propensity Overlap", "Matching", "Weighting (IPTW)", "Doubly Robust (AIPW)", "Export"]
)

# --- Propensity overlap ---
with tab_overlap:
    st.subheader("Propensity score distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=ps[treatment == 1], name="Treated", opacity=0.65, histnorm="probability density",
                                marker_color=PALETTE[0]))
    fig.add_trace(go.Histogram(x=ps[treatment == 0], name="Control", opacity=0.65, histnorm="probability density",
                                marker_color=PALETTE[1]))
    fig.add_vline(x=support["lower"], line_dash="dash", line_color="gray")
    fig.add_vline(x=support["upper"], line_dash="dash", line_color="gray")
    fig.update_layout(barmode="overlay", xaxis_title="Propensity score", yaxis_title="Density",
                       template="plotly_white", height=450, title="Propensity Score Overlap")
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"Common support region: [{support['lower']:.3f}, {support['upper']:.3f}]. "
        f"{support['n_outside_support']} patients fall outside it and have no comparable "
        "counterfactual in this data."
    )

    st.subheader("Covariate balance before adjustment")
    balance_before = cc.balance.balance_table(X, treatment, feature_names=covariates)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=balance_before["smd_before"], y=balance_before["feature"], mode="markers",
                               marker=dict(size=10, color=PALETTE[1])))
    fig2.add_vline(x=0, line_color="gray")
    fig2.add_vline(x=0.1, line_dash="dash", line_color="gray")
    fig2.add_vline(x=-0.1, line_dash="dash", line_color="gray")
    fig2.update_layout(xaxis_title="Standardized mean difference", template="plotly_white",
                        height=350 + 25 * len(covariates), title="Raw Covariate Balance")
    st.plotly_chart(fig2, width="stretch")

# --- Matching ---
with tab_match:
    st.subheader("Propensity score matching")
    mc1, mc2, mc3 = st.columns(3)
    ratio = mc1.slider("Controls per treated patient (k:1)", 1, 5, 1)
    replace = mc2.checkbox("Allow reusing controls", value=False)
    caliper_mult = mc3.slider("Caliper (x SD of logit propensity)", 0.05, 1.0, 0.2, 0.05)
    logit_ps = np.log(ps / (1 - ps))
    caliper = caliper_mult * np.std(logit_ps)

    with st.spinner("Matching..."):
        match_result = cc.matching.match_on_propensity(ps, treatment, caliper=caliper, ratio=ratio, replace=replace, seed=0)

    n_matched = len(np.unique(match_result["treated_idx"]))
    st.caption(
        f"Matched {n_matched} of {match_result['n_treated_total']} treated patients "
        f"({match_result['n_unmatched']} couldn't find a control within the caliper)."
    )

    if n_matched > 0:
        att_result = cc.matching.att_from_matching(y, match_result)
        with st.spinner("Bootstrapping confidence interval..."):
            ci_result = cc.matching.bootstrap_att_ci(X, y, treatment, caliper=caliper, ratio=ratio, replace=replace, n_boot=300, seed=0)

        m1, m2 = st.columns(2)
        m1.metric("ATT (matched)", f"{att_result['att']:.3f}")
        m2.metric("95% CI", f"({ci_result['ci_low']:.3f}, {ci_result['ci_high']:.3f})")

        st.subheader("Covariate balance after matching")
        matched_rows = np.concatenate([match_result["treated_idx"], match_result["control_idx"]])
        matched_treatment = np.concatenate([np.ones(len(match_result["treated_idx"])), np.zeros(len(match_result["control_idx"]))])
        balance_after = cc.balance.balance_table(X[matched_rows], matched_treatment, feature_names=covariates)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=balance_before.set_index("feature").loc[covariates, "smd_before"],
                                  y=covariates, mode="markers", name="Before matching", marker=dict(size=10, color=PALETTE[1])))
        fig.add_trace(go.Scatter(x=balance_after.set_index("feature").loc[covariates, "smd_before"],
                                  y=covariates, mode="markers", name="After matching", marker=dict(size=10, color=PALETTE[0])))
        fig.add_vline(x=0, line_color="gray")
        fig.add_vline(x=0.1, line_dash="dash", line_color="gray")
        fig.add_vline(x=-0.1, line_dash="dash", line_color="gray")
        fig.update_layout(xaxis_title="Standardized mean difference", template="plotly_white",
                           height=350 + 25 * len(covariates), title="Love Plot: Balance Before vs After Matching")
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("No matches found — try a larger caliper.")

# --- Weighting ---
with tab_weight:
    st.subheader("Inverse probability of treatment weighting")
    wc1, wc2 = st.columns(2)
    estimand = wc1.radio("Estimand", ["ATE", "ATT"], horizontal=True)
    trim_pct = wc2.slider("Trim weights above percentile", 90.0, 100.0, 99.0, 0.5)

    weights = cc.weighting.compute_iptw_weights(ps, treatment, estimand=estimand)
    weights_trimmed = cc.weighting.trim_weights(weights, percentile=trim_pct)
    point_ate = cc.weighting.weighted_ate(y, treatment, weights_trimmed)

    with st.spinner("Bootstrapping confidence interval..."):
        ci_result = cc.weighting.bootstrap_weighted_ate_ci(X, y, treatment, estimand=estimand, trim_percentile=trim_pct, n_boot=300, seed=0)

    w1, w2, w3 = st.columns(3)
    w1.metric(f"Weighted {estimand}", f"{point_ate:.3f}")
    w2.metric("95% CI", f"({ci_result['ci_low']:.3f}, {ci_result['ci_high']:.3f})")
    w3.metric("Max weight (trimmed)", f"{weights_trimmed.max():.2f}")

    st.subheader("Covariate balance after weighting")
    balance_weighted = cc.balance.balance_table(X, treatment, feature_names=covariates, weights_after=weights_trimmed)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=balance_weighted["smd_before"], y=balance_weighted["feature"], mode="markers",
                              name="Before weighting", marker=dict(size=10, color=PALETTE[1])))
    fig.add_trace(go.Scatter(x=balance_weighted["smd_after"], y=balance_weighted["feature"], mode="markers",
                              name="After weighting", marker=dict(size=10, color=PALETTE[0])))
    fig.add_vline(x=0, line_color="gray")
    fig.add_vline(x=0.1, line_dash="dash", line_color="gray")
    fig.add_vline(x=-0.1, line_dash="dash", line_color="gray")
    fig.update_layout(xaxis_title="Standardized mean difference", template="plotly_white",
                       height=350 + 25 * len(covariates), title="Love Plot: Balance Before vs After Weighting")
    st.plotly_chart(fig, width="stretch")

# --- Doubly robust ---
with tab_dr:
    st.subheader("Comparing estimators")
    st.caption(
        "AIPW combines the propensity and outcome models: it stays consistent if "
        "EITHER model is correctly specified, not necessarily both — a hedge against "
        "getting either model exactly right, which in practice you're rarely certain of."
    )

    with st.spinner("Fitting outcome models and computing AIPW..."):
        comparison = cc.estimation.compare_estimators(X, y, treatment)
        dr_ci = cc.estimation.bootstrap_aipw_ci(X, y, treatment, n_boot=300, seed=0)

    comp_df = pd.DataFrame([
        {"Estimator": "Naive (unadjusted)", "Estimate": comparison["naive_difference"]},
        {"Estimator": "Regression only", "Estimate": comparison["regression_only"]},
        {"Estimator": "IPTW only", "Estimate": comparison["iptw_only"]},
        {"Estimator": "AIPW (doubly robust)", "Estimate": comparison["aipw"]},
    ])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=comp_df["Estimator"], y=comp_df["Estimate"],
                          marker_color=[PALETTE[1], PALETTE[2], PALETTE[3], PALETTE[0]]))
    fig.update_layout(yaxis_title="Estimated treatment effect", template="plotly_white",
                       height=450, title="Estimators Compared")
    st.plotly_chart(fig, width="stretch")

    d1, d2 = st.columns(2)
    d1.metric("AIPW estimate", f"{comparison['aipw']:.3f}")
    d2.metric("95% CI", f"({dr_ci['ci_low']:.3f}, {dr_ci['ci_high']:.3f})")
    st.dataframe(comp_df.style.format({"Estimate": "{:.3f}"}), width="stretch")

# --- Export ---
with tab_export:
    st.subheader("Download results")
    full_summary = pd.DataFrame([{
        "n_patients": len(df),
        "n_treated": int(treatment.sum()),
        "naive_difference": y[treatment == 1].mean() - y[treatment == 0].mean(),
        "regression_only": comparison["regression_only"],
        "iptw_only": comparison["iptw_only"],
        "aipw": comparison["aipw"],
        "aipw_ci_low": dr_ci["ci_low"],
        "aipw_ci_high": dr_ci["ci_high"],
        "propensity_max_smd_before": cc.balance.max_absolute_smd(X, treatment),
    }])
    st.dataframe(full_summary, width="stretch")
    st.download_button(
        "Download summary (CSV)",
        full_summary.to_csv(index=False).encode("utf-8"),
        file_name="clincausal_summary.csv",
        mime="text/csv",
    )

st.sidebar.divider()
st.sidebar.caption("Built with [clincausal](..) + [Streamlit](https://streamlit.io) · [View source on GitHub](#)")
