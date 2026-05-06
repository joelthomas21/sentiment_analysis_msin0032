"""
03_scrape_app_store.py
----------------------
Scrapes Revolut iOS app reviews from Apple App Store using Apple's public
RSS JSON endpoint. No third-party library (app-store-scraper was dropped
because it pins requests==2.23.0 which conflicts with other dependencies).

Apple endpoint:
    https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={id}/sortBy=mostRecent/json

Output: data/raw/app_store_raw.csv

Usage:
    python 03_scrape_app_store.py

Notes:
- Revolut iOS app ID: 932493382
- Apple caps the RSS feed at ~10 pages × 50 reviews = ~500 per storefront.
  Iterating across country storefronts accumulates more.
- Pages 1..10 per country. Empty page = stop for that country.
"""

import argparse
import random
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

APP_ID = 932493382

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
]

CRYPTO_KEYWORDS = [
    "crypto", "bitcoin", "btc", "ethereum", "eth", "stablecoin",
    "usdt", "usdc", "tether", "trading", "trader", "revolut x",
    "exchange", "wallet", "token", "blockchain", "digital asset",
    "coinbase", "binance", "defi", "nft",
]


def is_crypto_relevant(text: str) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in CRYPTO_KEYWORDS)


def fetch_page(country: str, page: int, session: requests.Session) -> list[dict]:
    url = (f"https://itunes.apple.com/{country}/rss/customerreviews/"
           f"page={page}/id={APP_ID}/sortBy=mostRecent/json")
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        r = session.get(url, headers=headers, timeout=30)
    except Exception as e:
        print(f"  [!] {country} p{page}: {e}")
        return []
    if r.status_code != 200:
        return []
    try:
        data = r.json()
    except ValueError:
        return []
    entries = data.get("feed", {}).get("entry", [])
    # The first entry is feed metadata (when present); real reviews have 'im:rating'
    rows = []
    for e in entries:
        if "im:rating" not in e:
            continue  # skip app metadata entry
        rating = int(e.get("im:rating", {}).get("label", 0) or 0)
        title = e.get("title", {}).get("label", "")
        content = e.get("content", {}).get("label", "")
        date = e.get("updated", {}).get("label", "")
        review_id = e.get("id", {}).get("label", "")
        author = e.get("author", {}).get("name", {}).get("label", "")
        version = e.get("im:version", {}).get("label", "")
        rows.append({
            "review_id": review_id,
            "platform": "app_store",
            "country": country,
            "date": date,
            "star_rating": rating,
            "title": title,
            "text": content,
            "user_name": author,
            "app_version": version,
        })
    return rows


def scrape_for_country(country: str, session: requests.Session,
                       max_pages: int, delay: float) -> list[dict]:
    all_rows = []
    for page in range(1, max_pages + 1):
        rows = fetch_page(country, page, session)
        if not rows:
            break
        all_rows.extend(rows)
        time.sleep(delay + random.uniform(0, 0.5))
    return all_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="+",
                    default=["gb", "ie", "us", "de", "fr", "es", "nl", "it"])
    ap.add_argument("--max-pages", type=int, default=10,
                    help="max pages per country (Apple caps at ~10)")
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()

    session = requests.Session()
    all_rows = []
    for c in tqdm(args.countries, desc="App Store countries"):
        all_rows.extend(scrape_for_country(c, session, args.max_pages, args.delay))

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No reviews scraped.")
        return

    df["is_crypto"] = (df["text"].fillna("").apply(is_crypto_relevant) |
                       df["title"].fillna("").apply(is_crypto_relevant))

    print(f"\nTotal: {len(df)}")
    print(f"Crypto-relevant: {df['is_crypto'].sum()}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")

    out_all = OUT_DIR / "app_store_raw.csv"
    df.to_csv(out_all, index=False)
    print(f"Saved: {out_all}")

    out_crypto = OUT_DIR / "app_store_crypto.csv"
    df[df["is_crypto"]].to_csv(out_crypto, index=False)
    print(f"Saved (crypto-filtered): {out_crypto}")


if __name__ == "__main__":
    main()