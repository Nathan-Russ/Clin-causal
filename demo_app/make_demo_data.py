"""
Prepares the NHEFS demo dataset for the clincausal app.

NHEFS (National Health and Nutrition Examination Survey Epidemiologic Follow-up
Study) is the running example throughout Hernán & Robins, "Causal Inference:
What If" — the standard teaching dataset for propensity score methods. The
question: does quitting smoking (`qsmk`) cause weight gain (`wt82_71`, weight
change in kg from 1971 to 1982)? Naive comparison is confounded by age, sex,
smoking intensity/duration, exercise, and other health-related covariates that
affect both the decision to quit and subsequent weight change.

Loaded from the `causaldata` package, which bundles it directly (no network
access needed).
"""

import pandas as pd
from causaldata import nhefs

COVARIATES = [
    "age", "sex", "race", "education", "smokeintensity", "smokeyrs",
    "exercise", "active", "wt71",
]


def load_nhefs_demo() -> pd.DataFrame:
    df = nhefs.load_pandas().data
    cols = ["qsmk", "wt82_71"] + COVARIATES
    df = df[cols].dropna().reset_index(drop=True)
    df = df.rename(columns={"qsmk": "treatment", "wt82_71": "outcome"})
    return df


if __name__ == "__main__":
    df = load_nhefs_demo()
    df.to_csv("sample_data/nhefs_demo.csv", index=False)
    print(f"Saved {len(df)} rows to sample_data/nhefs_demo.csv")
    print(f"Treated (quit smoking): {df['treatment'].sum()} ({df['treatment'].mean():.1%})")
    print(f"Naive difference in mean weight change: {df[df.treatment==1].outcome.mean() - df[df.treatment==0].outcome.mean():.2f} kg")
