"""
05_run_vader.py
---------------
Runs VADER sentiment analysis on the cleaned reviews. Produces:
  - Per-review compound/pos/neg/neu scores
  - Classification (Positive/Negative/Neutral) at standard ±0.05 thresholds
  - Separate columns for baseline VADER and VADER + crypto-domain lexicon

Input:  data/clean/reviews_clean.csv
Output: data/clean/reviews_scored.csv

Usage:
    python 05_run_vader.py

Domain lexicon extension:
    VADER is general-purpose and misreads some crypto/fintech vocabulary.
    The extension below adjusts scores for a small set of high-salience terms.
    Rationale: reported in the dissertation methodology as a construct-validity
    adjustment. Both baseline and extended scores are retained so the reader
    can see how sensitive results are to the adjustment.
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

CLEAN_DIR = Path("data/clean")

# Domain lexicon: values are VADER-style polarity scores in [-4, +4].
# Negative scores make the term push the compound score down, positive up.
CRYPTO_LEXICON = {
    # Strongly negative fintech/crypto terms
    "rug": -3.0,
    "rugpull": -3.5,
    "rugged": -3.0,
    "scam": -3.0,
    "scammed": -3.5,
    "scammers": -3.0,
    "frozen": -2.5,
    "freeze": -2.0,
    "locked": -2.0,
    "stuck": -2.0,
    "rekt": -2.5,
    "depeg": -2.5,
    "depegged": -2.5,
    "hacked": -3.0,
    "phishing": -3.0,
    "unresponsive": -2.5,
    # Mildly positive crypto-community terms (often misread as neutral)
    "hodl": 0.8,
    "moon": 1.5,
    "mooning": 2.0,
    "bullish": 2.0,
    # Mildly negative
    "bearish": -1.5,
    "dump": -1.5,
    "dumping": -1.5,
}


def classify(compound: float) -> str:
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def main():
    in_path = CLEAN_DIR / "reviews_clean.csv"
    if not in_path.exists():
        raise SystemExit(f"Missing {in_path}. Run 04_clean_reviews.py first.")
    df = pd.read_csv(in_path)
    print(f"Scoring {len(df)} reviews...")

    # Baseline VADER
    base = SentimentIntensityAnalyzer()

    # Extended VADER (with crypto lexicon)
    ext = SentimentIntensityAnalyzer()
    ext.lexicon.update(CRYPTO_LEXICON)

    base_rows = []
    ext_rows = []
    for text in tqdm(df["text"].fillna(""), desc="VADER"):
        b = base.polarity_scores(text)
        e = ext.polarity_scores(text)
        base_rows.append(b)
        ext_rows.append(e)

    for key in ["neg", "neu", "pos", "compound"]:
        df[f"vader_{key}"] = [r[key] for r in base_rows]
        df[f"vader_ext_{key}"] = [r[key] for r in ext_rows]

    df["vader_class"] = df["vader_compound"].apply(classify)
    df["vader_ext_class"] = df["vader_ext_compound"].apply(classify)

    out = CLEAN_DIR / "reviews_scored.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}")

    # Summary for the write-up
    print("\nBaseline VADER classification:")
    print(df["vader_class"].value_counts(normalize=True).round(3).to_string())
    print("\nExtended VADER classification:")
    print(df["vader_ext_class"].value_counts(normalize=True).round(3).to_string())
    print(f"\nMean compound (baseline): {df['vader_compound'].mean():.3f}")
    print(f"Mean compound (extended): {df['vader_ext_compound'].mean():.3f}")
    print(f"Length-weighted mean (baseline): "
          f"{(df['vader_compound'] * df['word_count']).sum() / df['word_count'].sum():.3f}")
    print(f"Length-weighted mean (extended): "
          f"{(df['vader_ext_compound'] * df['word_count']).sum() / df['word_count'].sum():.3f}")


if __name__ == "__main__":
    main()
