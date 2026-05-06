"""
02b_scrape_google_play_revolut_x.py
-----------------------------------
Scrapes Revolut X (dedicated crypto exchange app) reviews from Google Play.

Revolut X is Revolut's standalone crypto app (launched 2024). Every user
of this app is crypto-active by definition, which makes it the ideal
source to boost the `primary_focus_crypto` segment of the thematic
analysis without needing keyword filtering.

Google Play ID: com.revolut.revolutx

Output: data/raw/revolut_x_google_play_raw.csv

Usage:
    python 02b_scrape_google_play_revolut_x.py --count 2000

Methodological note for the dissertation:
    The supplementary Revolut X scrape was added after the initial
    cleaned sample yielded only n=15 primary_focus_crypto reviews in the
    stratified subsample, which is thin for per-segment comparison. The
    Revolut X review stream is entirely crypto-focused by construction,
    hence does not require crypto-keyword filtering and is not subject
    to the filter-miss problem that affects the main Revolut app scrape.
"""

import argparse
from pathlib import Path

import pandas as pd
from google_play_scraper import reviews, Sort
from tqdm import tqdm

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

APP_ID = "com.revolut.revolutx"


def scrape_for_country(country: str, count: int) -> list[dict]:
    all_rows = []
    batch_size = 200
    continuation_token = None

    pbar = tqdm(total=count, desc=f"Revolut X Play ({country})")
    while len(all_rows) < count:
        result, continuation_token = reviews(
            APP_ID,
            lang="en",
            country=country,
            sort=Sort.NEWEST,
            count=batch_size,
            continuation_token=continuation_token,
        )
        if not result:
            break
        for r in result:
            all_rows.append({
                "review_id": r.get("reviewId"),
                "platform": "google_play_revolut_x",
                "country": country,
                "date": r.get("at").isoformat() if r.get("at") else None,
                "star_rating": r.get("score"),
                "title": None,
                "text": r.get("content") or "",
                "thumbs_up": r.get("thumbsUpCount"),
                "app_version": r.get("reviewCreatedVersion"),
            })
        pbar.update(len(result))
        if continuation_token is None:
            break
    pbar.close()
    return all_rows[:count]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=2000,
                    help="total reviews per country")
    ap.add_argument("--countries", nargs="+",
                    default=["gb", "ie", "de", "fr", "es", "nl", "us"],
                    help="Play Store country codes")
    args = ap.parse_args()

    all_rows = []
    for c in args.countries:
        try:
            all_rows.extend(scrape_for_country(c, args.count))
        except Exception as e:
            print(f"  [!] {c} failed: {e}")

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No reviews scraped.")
        return

    # Every row is structurally crypto-relevant because every reviewer
    # is using a dedicated crypto app. Flag them all as is_crypto and
    # let the main cleaner/filter merge them with the main Revolut corpus.
    df["is_crypto"] = True
    df["is_trust_concern"] = False  # will be re-filtered by 00_refilter
    df["is_fx"] = False
    df["is_relevant"] = True
    df["matched_crypto"] = "revolut_x_app"
    df["matched_trust"] = ""
    df["matched_fx"] = ""

    print(f"\nTotal Revolut X reviews: {len(df)}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"Star rating distribution:")
    print(df['star_rating'].value_counts().sort_index().to_string())

    out = OUT_DIR / "revolut_x_google_play_raw.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
