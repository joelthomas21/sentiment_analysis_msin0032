"""
02_scrape_google_play.py
------------------------
Scrapes Revolut app reviews from Google Play using google-play-scraper.
This library wraps the Play Store's public review endpoints and is reliable.

Output: data/raw/google_play_raw.csv

Usage:
    python 02_scrape_google_play.py --count 3000

Notes:
- Revolut app ID on Google Play: com.revolut.revolut
- --count is the TOTAL reviews requested; crypto filter applied after.
- Filters reviews to English (lang='en') and GB/US-style countries to reduce
  translation noise. Adjust if you want broader geographic coverage.
"""

import argparse
from pathlib import Path

import pandas as pd
from google_play_scraper import reviews, Sort
from tqdm import tqdm

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

APP_ID = "com.revolut.revolut"

CRYPTO_KEYWORDS = [
    "crypto", "bitcoin", "btc", "ethereum", "eth", "stablecoin",
    "usdt", "usdc", "tether", "trading", "trader", "revolut x",
    "exchange", "wallet", "token", "blockchain", "digital asset",
    "coinbase", "binance", "defi", "nft",
]


def is_crypto_relevant(text: str) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in CRYPTO_KEYWORDS)


def scrape_for_country(country: str, count: int) -> list[dict]:
    """Scrape reviews for a given country code."""
    all_rows = []
    batch_size = 200
    continuation_token = None

    pbar = tqdm(total=count, desc=f"Google Play ({country})")
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
                "platform": "google_play",
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
    ap.add_argument("--count", type=int, default=3000,
                    help="total reviews to scrape per country")
    ap.add_argument("--countries", nargs="+",
                    default=["gb", "ie", "de", "fr", "es", "nl"],
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

    df["is_crypto"] = df["text"].fillna("").apply(is_crypto_relevant)

    print(f"\nTotal: {len(df)}")
    print(f"Crypto-relevant: {df['is_crypto'].sum()}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")

    out_all = OUT_DIR / "google_play_raw.csv"
    df.to_csv(out_all, index=False)
    print(f"Saved: {out_all}")

    out_crypto = OUT_DIR / "google_play_crypto.csv"
    df[df["is_crypto"]].to_csv(out_crypto, index=False)
    print(f"Saved (crypto-filtered): {out_crypto}")


if __name__ == "__main__":
    main()
