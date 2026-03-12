"""
Statistical analysis of psycholinguistic features in suicidal vs.
non-suicidal Reddit posts.

Tests:
  - Welch's independent-samples t-test per feature
  - Cohen's d effect size
  - OLS multiple regression predicting group membership

Welch's t-test (equal_var=False) is used instead of Student's t-test
because we cannot assume equal variances between groups.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "posts_with_features.csv"

FEATURES = {
    "mean_log_freq": "Mean Log Word Frequency",
    "mean_log_cd":   "Mean Log Contextual Diversity",
    "mean_aoa":      "Mean Age of Acquisition",
}

def cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """
    Compute Cohen's d for two independent groups.

    Divides the difference between the two means by the pooled standard
    deviation — a combined measure of spread across both groups.
    """
    n1, n2   = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std

def interpret_d(d: float) -> str:

    #Return a  label for a Cohen's d value
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
    
def print_section(title: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def main():
    
    # 1.Load data
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} posts loaded")

    # 2.Split into groups
    suicide_posts = df[df["class"] == "suicide"]
    control_posts = df[df["class"] == "non-suicide"]
    print(f"  Suicide posts  : {len(suicide_posts):,}")
    print(f"  Control posts  : {len(control_posts):,}")

    # 3.Descriptive statistics — one table per feature, side by side
    print_section("DESCRIPTIVE STATISTICS")

    desc_rows = []
    for col, label in FEATURES.items():
        for group_name, group_df in [("Suicide", suicide_posts), ("Control", control_posts)]:
            s = group_df[col].dropna()
            desc_rows.append({
                "Feature": label,
                "Group":   group_name,
                "N":       len(s),
                "Mean":    round(s.mean(), 4),
                "SD":      round(s.std(), 4),
                "Median":  round(s.median(), 4),
                "Min":     round(s.min(), 4),
                "Max":     round(s.max(), 4),
            })

    desc_df = pd.DataFrame(desc_rows)

    # Print each feature block separately for readability
    for col, label in FEATURES.items():
        print(f"\n{label}")
        block = desc_df[desc_df["Feature"] == label].drop(columns="Feature")
        print(block.to_string(index=False))

    # 4.Welch's t-tests + Cohen's d
    print_section("INFERENTIAL STATISTICS — Welch's t-tests")

    test_results = []
    for col, label in FEATURES.items():
        g1 = suicide_posts[col].dropna()
        g2 = control_posts[col].dropna()

        # Welch's t-test (equal_var=False does not assume σ1 = σ2)
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)

        d = cohens_d(g1, g2)
        direction = "suicide > control" if g1.mean() > g2.mean() else "suicide < control"

        test_results.append({
            "Feature":    label,
            "t":          round(t_stat, 3),
            "p":          p_val,
            "Cohen's d":  round(d, 3),
            "Effect":     interpret_d(d),
            "Direction":  direction,
        })

    results_df = pd.DataFrame(test_results)

    # Format p-value for display
    results_df["p (formatted)"] = results_df["p"].apply(
        lambda x: "< .001" if x < .001 else f"= {x:.3f}"
    )

    print()
    for _, row in results_df.iterrows():
        print(f"Feature   : {row['Feature']}")
        print(f"  t-stat  : {row['t']}")
        print(f"  p-value : {row['p (formatted)']}")
        d_val = row["Cohen's d"]
        print(f"  Cohen's d: {d_val}  ({row['Effect']} effect, {row['Direction']})")
        print()


    # 5.OLS multiple regression predicting class (0 = control, 1 = suicide)
    print_section("OLS MULTIPLE REGRESSION")
    print("  Outcome: class (0 = non-suicide, 1 = suicide)")
    print("  Predictors: mean_log_freq, mean_log_cd, mean_aoa\n")

    # Encode class as binary integer for regression
    df["class_bin"] = (df["class"] == "suicide").astype(int)

    # Drop rows with any missing feature (OLS cannot handle NaN)
    reg_df = df[["class_bin", "mean_log_freq", "mean_log_cd", "mean_aoa"]].dropna()
    print(f"  Rows used in regression: {len(reg_df):,}")

    formula = "class_bin ~ mean_log_freq + mean_log_cd + mean_aoa"
    model   = smf.ols(formula, data=reg_df).fit()

    print(model.summary())

    # Plain coefficient summary
    print_section("REGRESSION COEFFICIENTS — Plain English")
    for name, coef, pval in zip(
        model.params.index, model.params.values, model.pvalues.values
    ):
        sig = "***" if pval < .001 else ("**" if pval < .01 else ("*" if pval < .05 else "n.s."))
        print(f"  {name:<28} β = {coef:+.4f}   p {('< .001' if pval < .001 else f'= {pval:.3f}')}  {sig}")

    print(f"\n  R² = {model.rsquared:.4f}  |  Adj. R² = {model.rsquared_adj:.4f}")
    print(
        "  Note: OLS is used here for interpretability — each β tells you how much\n"
        "  the probability of a post being suicidal changes when that feature increases\n"
        "  by one unit. For a final model, logistic regression would be preferred.\n"
    )

if __name__ == "__main__":
    main()