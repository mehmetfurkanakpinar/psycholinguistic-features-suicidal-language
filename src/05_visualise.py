
#Figures produced:
#  fig1_violins.png  — violin plots of each norm feature by class
#  fig2_scatter.png  — scatter plot of log word freq vs AoA, coloured by class
#  fig3_wordcount.png — overlapping histograms of post word count by class


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


DATA_PATH   = Path(__file__).resolve().parents[1] / "data" / "processed" / "posts_with_features.csv"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"

# Consistent colour palette: non-suicide = blue, suicide = red
PALETTE = {"non-suicide": "#4878CF", "suicide": "#D65F5F"}

# Class order for consistent x-axis ordering across all plots
CLASS_ORDER = ["non-suicide", "suicide"]


def setup():
    #Apply global plot style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def fig1_violins(df: pd.DataFrame) -> None:
    """
    Figure 1: 1×3 violin plots, one per psycholinguistic feature.
    Violin plots show the full distribution shape which is important here 
    because the distributions may be skewed or bimodal within each class.
    """
    features = [
        ("mean_log_freq", "Mean Log10 Word Frequency\n(SUBTLEXUS Lg10WF)"),
        ("mean_log_cd",   "Mean Log10 Contextual Diversity\n(SUBTLEXUS Lg10CD)"),
        ("mean_aoa",      "Mean Age of Acquisition\n(years, Kuperman et al.)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Psycholinguistic Features by Post Class",
        fontsize=15, fontweight="bold", y=1.02,
    )

    for ax, (col, ylabel) in zip(axes, features):
        plot_data = df[["class", col]].dropna()
        sns.violinplot(
            data=plot_data,
            x="class", y=col,
            hue="class",
            order=CLASS_ORDER,
            hue_order=CLASS_ORDER,
            palette=PALETTE,
            inner="box",      # show median + IQR box inside the violin
            linewidth=0.8,
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("Post Class", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel.split("\n")[0], fontsize=11, fontweight="bold")
        # Tidy x-tick labels using a FixedLocator to avoid UserWarning
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-Suicide", "Suicide"])

    fig.tight_layout()
    out = FIGURES_DIR / "fig1_violins.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def fig2_scatter(df: pd.DataFrame) -> None:
    """
    Figure 2: Scatter plot of mean_log_freq vs mean_aoa coloured by class.
    With ~224k points, alpha=0.1 is essential to reveal density structure
    rather than a solid overplotted blob.
    """
    plot_data = df[["class", "mean_log_freq", "mean_aoa"]].dropna()

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each class separately so we can control alpha and add a legend
    for cls in CLASS_ORDER:
        subset = plot_data[plot_data["class"] == cls]
        ax.scatter(
            subset["mean_log_freq"],
            subset["mean_aoa"],
            c=PALETTE[cls],
            alpha=0.1,
            s=2,           
            label=cls.replace("-", "-").title(),
            rasterized=True, 
        )

    ax.set_xlabel("Mean Log10 Word Frequency", fontsize=12)
    ax.set_ylabel("Mean Age of Acquisition (years)", fontsize=12)
    ax.set_title(
        "Word Frequency vs Age of Acquisition by Post Class\n"
        "(alpha = 0.1; each point = one post)",
        fontsize=12, fontweight="bold",
    )

    # Legend with opaque markers (override alpha so they're visible)
    handles, labels = ax.get_legend_handles_labels()
    for h in handles:
        h.set_alpha(1.0)
        h.set_sizes([40])
    ax.legend(handles, labels, title="Class", framealpha=0.9)

    fig.tight_layout()
    out = FIGURES_DIR / "fig2_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def fig3_wordcount(df: pd.DataFrame) -> None:
    """
    Figure 3: Overlapping histograms of post word count by class.
    alpha=0.5 lets both distributions show through where they overlap.
    We cap at the 99th percentile to avoid extreme outliers.
    """
    cap = int(df["word_count"].quantile(0.99))
    plot_data = df[df["word_count"] <= cap]

    fig, ax = plt.subplots(figsize=(9, 5))

    for cls in CLASS_ORDER:
        subset = plot_data[plot_data["class"] == cls]["word_count"]
        ax.hist(
            subset,
            bins=80,
            alpha=0.5,
            color=PALETTE[cls],
            label=cls.replace("-", "-").title(),
            edgecolor="none",
        )

    ax.set_xlabel("Word Count per Post", fontsize=12)
    ax.set_ylabel("Number of Posts", fontsize=12)
    ax.set_title(
        f"Distribution of Post Length by Class\n"
        f"(capped at 99th percentile = {cap} words)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(title="Class", framealpha=0.9)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    fig.tight_layout()
    out = FIGURES_DIR / "fig3_wordcount.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    
    # 1. Load data
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df):,} posts loaded")

    # 2–3. Apply global style and palette
    setup()

    # 4–6. Generate and save figures
    print("\nGenerating figures ...")
    fig1_violins(df)
    fig2_scatter(df)
    fig3_wordcount(df)

    print("\nAll figures saved to results/figures/")


if __name__ == "__main__":
    main()
