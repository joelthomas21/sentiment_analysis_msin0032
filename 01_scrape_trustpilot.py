"""
01_scrape_trustpilot.py
-----------------------
Scrapes Revolut reviews from Trustpilot by parsing the __NEXT_DATA__ JSON
block embedded in each review page. This is more stable than HTML scraping
because Trustpilot uses Next.js and all review data is serialised into that
JSON block.

Output: data/raw/trustpilot_raw.csv

Usage:
    python 01_scrape_trustpilot.py --max-pages 200 --delay 1.5

Notes:
- Trustpilot paginates at ~20 reviews/page, so 200 pages ≈ 4,000 raw reviews
  (crypto-filtering will reduce this substantially).
- Rate limit politely. Do NOT parallelise.
- If blocked, reduce --max-pages and increase --delay.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://www.trustpilot.com/review/revolut.com"

# Rotate through a few user-agents to look less like a single bot.
USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]

# Crypto keywords for later filtering; we scrape everything and filter after,
# because filtering at scrape time risks excluding reviews that are crypto-relevant
# but use adjacent vocabulary ("withdrawal stuck", "account frozen").
CRYPTO_KEYWORDS = [
    "crypto", "bitcoin", "btc", "ethereum", "eth", "stablecoin",
    "usdt", "usdc", "tether", "trading", "trader", "revolut x",
    "exchange", "wallet", "token", "blockchain", "digital asset",
    "coinbase", "binance", "defi", "nft",
]


def headers():
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


def fetch_page(page: int, session: requests.Session) -> list[dict]:
    url = BASE_URL if page == 1 else f"{BASE_URL}?page={page}"
    r = session.get(url, headers=headers(), timeout=30)
    if r.status_code != 200:
        print(f"  [!] page {page}: HTTP {r.status_code}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    script = soup.find("script", id="__NEXT_DATA__")
    if script is None:
        print(f"  [!] page {page}: no __NEXT_DATA__")
        return []

    data = json.loads(script.string)
    try:
        reviews = data["props"]["pageProps"]["reviews"]
    except KeyError:
        return []

    out = []
    for rv in reviews:
        out.append({
            "review_id": rv.get("id"),
            "platform": "trustpilot",
            "date": rv.get("dates", {}).get("publishedDate"),
            "star_rating": rv.get("rating"),
            "title": rv.get("title"),
            "text": rv.get("text") or "",
            "language": rv.get("language"),
            "location": rv.get("consumer", {}).get("countryCode"),
            "url": f"https://www.trustpilot.com/reviews/{rv.get('id')}",
        })
    return out


def is_crypto_relevant(text: str) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in CRYPTO_KEYWORDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=200)
    ap.add_argument("--delay", type=float, default=1.5,
                    help="seconds between requests (be polite)")
    ap.add_argument("--jitter", type=float, default=0.5)
    args = ap.parse_args()

    session = requests.Session()
    all_rows = []

    for page in tqdm(range(1, args.max_pages + 1), desc="Trustpilot"):
        rows = fetch_page(page, session)
        if not rows:
            # Empty page = hit the end, or blocked. Stop.
            print(f"  [i] stopping at page {page} (empty or blocked)")
            break
        all_rows.extend(rows)
        time.sleep(args.delay + random.uniform(0, args.jitter))

    df = pd.DataFrame(all_rows)
    df["is_crypto"] = df["text"].fillna("").apply(is_crypto_relevant) | \
                     df["title"].fillna("").apply(is_crypto_relevant)

    print(f"\nTotal reviews scraped: {len(df)}")
    print(f"Crypto-relevant after filter: {df['is_crypto'].sum()}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")

    out_all = OUT_DIR / "trustpilot_raw.csv"
    df.to_csv(out_all, index=False)
    print(f"Saved: {out_all}")

    out_crypto = OUT_DIR / "trustpilot_crypto.csv"
    df[df["is_crypto"]].to_csv(out_crypto, index=False)
    print(f"Saved (crypto-filtered): {out_crypto}")


if __name__ == "__main__":
    main()
