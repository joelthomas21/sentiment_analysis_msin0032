"""
04_clean_reviews.py
-------------------
Merges the refiltered scraped files, applies cleaning rules, produces a
single cleaned dataset ready for VADER scoring.

Reads data/raw/*_filtered.csv (produced by 00_refilter.py).
Writes data/clean/reviews_clean.csv.

Fixes from previous version:
1. Date parsing is now robust across Trustpilot/Google Play/App Store formats.
2. Rows with unparseable dates are KEPT (with a warning count), not silently
   dropped. Previously, Google Play and App Store rows vanished here because
   their date formats weren't parsing to comparable timestamps.
3. Reads the dual-classification columns (is_crypto, is_trust_concern) and
   can optionally restrict the clean set to crypto-only.

Usage:
    python 04_clean_reviews.py
    python 04_clean_reviews.py --crypto-only
    python 04_clean_reviews.py --min-words 10 --start-date 2024-01-01
"""

import argparse
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect
from rapidfuzz import fuzz
from tqdm import tqdm

DetectorFactory.seed = 42

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def load_all() -> pd.DataFrame:
    frames = []
    file_list = [
        "trustpilot_filtered.csv",
        "google_play_filtered.csv",
        "app_store_filtered.csv",
        "revolut_x_google_play_filtered.csv",
        "revolut_x_app_store_filtered.csv",
    ]
    for name in file_list:
        p = RAW_DIR / name
        if p.exists():
            df = pd.read_csv(p)
            print(f"  [+] loaded {len(df):>5} from {name}")
            frames.append(df)
        else:
            print(f"  [i] skipping {name} (not present)")
    if not frames:
        raise SystemExit("No *_filtered.csv files found. Run 00_refilter.py.")
    return pd.concat(frames, ignore_index=True, sort=False)


def robust_parse_date(val):
    """Parse a single date value across Trustpilot/Google Play/App Store formats.

    Accepts:
      - Trustpilot: '2026-04-17T13:03:41.000Z' or similar ISO with Z
      - Trustpilot read-back: '2026-04-16 16:26:26+00:00'
      - Google Play (isoformat): '2024-06-15T14:23:11' (no tz) or with ms
      - Apple RSS: '2024-12-15T10:30:00-07:00'
      - Empty/NaN: returns NaT
    """
    if pd.isna(val) or val is None or val == "":
        return pd.NaT
    try:
        return pd.to_datetime(val, utc=True)
    except (ValueError, TypeError):
        pass
    # Fall back: take first 10 chars as date-only
    try:
        return pd.to_datetime(str(val)[:10], utc=True)
    except Exception:
        return pd.NaT


def safe_detect_lang(text: str) -> str:
    try:
        return detect(text)
    except (LangDetectException, TypeError):
        return "unknown"


def drop_near_duplicates(df: pd.DataFrame, threshold: int = 90) -> pd.DataFrame:
    """Drop near-duplicate texts. O(n^2). Fine for <10k rows."""
    texts = df["text"].fillna("").tolist()
    keep = [True] * len(texts)
    for i in tqdm(range(len(texts)), desc="near-dup check"):
        if not keep[i]:
            continue
        for j in range(i + 1, len(texts)):
            if not keep[j]:
                continue
            if fuzz.ratio(texts[i], texts[j]) > threshold:
                keep[j] = False
    return df[keep].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-words", type=int, default=10)
    ap.add_argument("--start-date", default="2024-01-01")
    ap.add_argument("--skip-near-dup", action="store_true",
                    help="skip O(n^2) near-duplicate check")
    ap.add_argument("--crypto-only", action="store_true",
                    help="keep only reviews flagged is_crypto (excludes "
                         "trust-only reviews). Default keeps both.")
    args = ap.parse_args()

    print("Loading refiltered files...")
    df = load_all()
    initial = len(df)
    print(f"\nTotal raw filtered: {initial}")

    # Optionally restrict to crypto-only
    if args.crypto_only and "is_crypto" in df.columns:
        before = len(df)
        df = df[df["is_crypto"].astype(bool)].reset_index(drop=True)
        print(f"After crypto-only:         {len(df)}  (−{before - len(df)})")

    # 1. Drop missing/empty text
    df = df[df["text"].notna() & (df["text"].astype(str).str.strip() != "")].copy()
    print(f"After drop-empty:          {len(df)}")

    # 2. Exact duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    print(f"After exact-dedup:         {len(df)}  (−{before - len(df)})")

    # 3. Word-count filter
    df["word_count"] = df["text"].astype(str).str.split().str.len()
    before = len(df)
    df = df[df["word_count"] >= args.min_words].reset_index(drop=True)
    print(f"After min-words {args.min_words:>2}:        {len(df)}  (−{before - len(df)})")

    # 4. Date parsing (robust) + filter
    print("Parsing dates (robust)...")
    tqdm.pandas(desc="date parse")
    df["date"] = df["date"].progress_apply(robust_parse_date)

    n_nat = df["date"].isna().sum()
    if n_nat:
        print(f"  [!] {n_nat} rows had unparseable dates — keeping them anyway "
              f"(excluded from date range filter)")

    cutoff = pd.Timestamp(args.start_date, tz="UTC")
    before = len(df)
    # Keep if date is NaT OR date >= cutoff
    df = df[df["date"].isna() | (df["date"] >= cutoff)].reset_index(drop=True)
    print(f"After date ≥ {args.start_date}: {len(df)}  (−{before - len(df)})")

    # 5. Language detection
    print("Detecting language...")
    tqdm.pandas(desc="langdetect")
    df["lang_detected"] = df["text"].astype(str).progress_apply(safe_detect_lang)
    before = len(df)
    df = df[df["lang_detected"] == "en"].reset_index(drop=True)
    print(f"After English only:        {len(df)}  (−{before - len(df)})")

    # 6. Near-duplicate check
    if not args.skip_near_dup:
        before = len(df)
        df = drop_near_duplicates(df)
        print(f"After near-dup (>90):      {len(df)}  (−{before - len(df)})")

    # Final tidy
    df["platform"] = df["platform"].fillna("unknown").astype(str)
    df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce")

    out = CLEAN_DIR / "reviews_clean.csv"
    df.to_csv(out, index=False)
    print(f"\nFinal clean n = {len(df)}")
    print(f"Saved: {out}")

    print("\nPlatform distribution:")
    print(df["platform"].value_counts().to_string())

    if "is_crypto" in df.columns:
        print("\nCrypto vs trust-concern split:")
        print(f"  is_crypto:        {df['is_crypto'].astype(bool).sum()}")
        print(f"  is_trust_concern: {df['is_trust_concern'].astype(bool).sum()}")
        both = (df["is_crypto"].astype(bool) & df["is_trust_concern"].astype(bool)).sum()
        print(f"  both:             {both}")

    print("\nStar rating distribution:")
    print(df["star_rating"].value_counts().sort_index().to_string())

    if df["date"].notna().any():
        print(f"\nDate range (parsed): {df['date'].min()} → {df['date'].max()}")
    print(f"Mean word count: {df['word_count'].mean():.1f}")


if __name__ == "__main__":
    main()
