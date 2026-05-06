"""
06_analyse_distribution.py
--------------------------
Produces the distributional evidence for the write-up:
  - Histogram of VADER compound scores with threshold lines
  - Hartigan's dip test for unimodality (scipy doesn't ship it; use a simple
    bimodality coefficient as a proxy, which is standard and defensible)
  - Star-rating × VADER-classification crosstab (validation)
  - Length × sentiment scatter
  - Summary stats per platform

Input:  data/clean/reviews_scored.csv
Output: figures/*.png, tables/*.csv, summary.md

Usage:
    python 06_analyse_distribution.py --lexicon ext
    # --lexicon {base, ext}  which VADER scores to analyse
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

CLEAN_DIR = Path("data/clean")
FIG_DIR = Path("figures"); FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = Path("tables"); TAB_DIR.mkdir(exist_ok=True)


def bimodality_coefficient(x: np.ndarray) -> float:
    """
    Bimodality coefficient (SAS/Pfister formulation):
        BC = (skew^2 + 1) / (kurt + (3*(n-1)^2) / ((n-2)*(n-3)))
    BC > 0.555 is conventionally taken as evidence of bimodality.
    More defensible than eyeballing the histogram.
    """
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 4:
        return np.nan
    skew = stats.skew(x, bias=False)
    kurt = stats.kurtosis(x, bias=False, fisher=False)  # non-excess
    num = skew ** 2 + 1
    denom = kurt + (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return num / denom


def plot_histogram(df: pd.DataFrame, col: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[col].dropna(), bins=40, edgecolor="black", alpha=0.75)
    ax.axvline(-0.05, color="red", linestyle="--", linewidth=1,
               label="negative threshold (−0.05)")
    ax.axvline(0.05, color="green", linestyle="--", linewidth=1,
               label="positive threshold (+0.05)")
    ax.axvline(df[col].mean(), color="black", linestyle=":", linewidth=1.5,
               label=f"mean = {df[col].mean():.3f}")
    ax.set_xlabel("VADER compound score")
    ax.set_ylabel("Number of reviews")
    ax.set_title(f"Distribution of VADER compound scores (n = {df[col].notna().sum()})")
    ax.legend(loc="upper center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_star_vs_vader(df: pd.DataFrame, col: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    df_plot = df.dropna(subset=["star_rating", col])
    df_plot = df_plot[df_plot["star_rating"].between(1, 5)]
    for star in sorted(df_plot["star_rating"].unique()):
        subset = df_plot[df_plot["star_rating"] == star][col]
        ax.hist(subset, bins=30, alpha=0.45, label=f"{int(star)}★ (n={len(subset)})")
    ax.set_xlabel("VADER compound score")
    ax.set_ylabel("Number of reviews")
    ax.set_title("VADER compound by star rating (validation)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_length_vs_sentiment(df: pd.DataFrame, col: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["word_count"], df[col], alpha=0.3, s=10)
    ax.set_xlabel("Review word count")
    ax.set_ylabel("VADER compound score")
    ax.set_xscale("log")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Review length vs sentiment")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", choices=["base", "ext"], default="ext")
    args = ap.parse_args()

    compound_col = "vader_compound" if args.lexicon == "base" else "vader_ext_compound"
    class_col = "vader_class" if args.lexicon == "base" else "vader_ext_class"

    df = pd.read_csv(CLEAN_DIR / "reviews_scored.csv")
    print(f"Analysing {len(df)} reviews using {args.lexicon} VADER.\n")

    # Figures
    plot_histogram(df, compound_col, FIG_DIR / f"distribution_{args.lexicon}.png")
    plot_star_vs_vader(df, compound_col, FIG_DIR / f"star_vs_vader_{args.lexicon}.png")
    plot_length_vs_sentiment(df, compound_col, FIG_DIR / f"length_vs_sentiment_{args.lexicon}.png")

    # Bimodality
    bc = bimodality_coefficient(df[compound_col].values)
    bimodal = "YES" if bc > 0.555 else "NO"
    print(f"Bimodality coefficient: {bc:.3f}  (>0.555 → bimodal: {bimodal})")

    # Classification share (overall + by platform)
    overall = df[class_col].value_counts(normalize=True).round(3)
    by_platform = (df.groupby("platform")[class_col]
                   .value_counts(normalize=True)
                   .round(3).unstack(fill_value=0))
    overall.to_csv(TAB_DIR / f"classification_overall_{args.lexicon}.csv")
    by_platform.to_csv(TAB_DIR / f"classification_by_platform_{args.lexicon}.csv")
    print("\nOverall classification shares:")
    print(overall.to_string())
    print("\nBy platform:")
    print(by_platform.to_string())

    # Star rating × VADER class crosstab
    crosstab = pd.crosstab(df["star_rating"], df[class_col], normalize="index").round(3)
    crosstab.to_csv(TAB_DIR / f"star_vader_crosstab_{args.lexicon}.csv")
    print("\nStar × VADER crosstab (row-normalised):")
    print(crosstab.to_string())

    # Correlations
    corr_star = df[["star_rating", compound_col]].corr(method="spearman").iloc[0, 1]
    corr_len = df[["word_count", compound_col]].corr(method="spearman").iloc[0, 1]
    print(f"\nSpearman ρ(star, compound) = {corr_star:.3f}")
    print(f"Spearman ρ(word_count, compound) = {corr_len:.3f}")

    # Sample description table for Part II
    sample_desc = (df.groupby("platform")
                   .agg(n=("review_id", "count"),
                        date_min=("date", "min"),
                        date_max=("date", "max"),
                        mean_words=("word_count", "mean"),
                        mean_stars=("star_rating", "mean"),
                        mean_compound=(compound_col, "mean"))
                   .round(3))
    sample_desc.to_csv(TAB_DIR / "sample_description.csv")
    print("\nSample description:")
    print(sample_desc.to_string())

    # Write a summary.md for the write-up
    summary = f"""# Sentiment analysis summary ({args.lexicon} VADER)

- Total reviews analysed: **{len(df)}**
- Date range: {df['date'].min()} → {df['date'].max()}
- Mean compound score: {df[compound_col].mean():.3f}
- Length-weighted mean: {(df[compound_col] * df['word_count']).sum() / df['word_count'].sum():.3f}
- Bimodality coefficient: {bc:.3f} → {'bimodal' if bc > 0.555 else 'unimodal/skewed'}
- Spearman ρ(star, compound): {corr_star:.3f}
- Spearman ρ(word_count, compound): {corr_len:.3f}

## Classification (overall)
{overall.to_markdown()}

## Classification by platform
{by_platform.to_markdown()}

## Sample description
{sample_desc.to_markdown()}

## Star × VADER crosstab
{crosstab.to_markdown()}
"""
    summary_path = TAB_DIR / f"summary_{args.lexicon}.md"
    summary_path.write_text(summary)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
