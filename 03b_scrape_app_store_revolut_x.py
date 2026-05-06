"""
03b_scrape_app_store_revolut_x.py
---------------------------------
Scrapes Revolut X (dedicated crypto exchange app) reviews from Apple's
App Store RSS endpoint.

iOS App ID: 6502614478

Output: data/raw/revolut_x_app_store_raw.csv

Usage:
    python 03b_scrape_app_store_revolut_x.py

Same methodological rationale as 02b: Revolut X reviews are structurally
crypto-focused and do not require keyword filtering.
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

APP_ID = 6502614478

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
]


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
    rows = []
    for e in entries:
        if "im:rating" not in e:
            continue
        rating = int(e.get("im:rating", {}).get("label", 0) or 0)
        title = e.get("title", {}).get("label", "")
        content = e.get("content", {}).get("label", "")
        date = e.get("updated", {}).get("label", "")
        review_id = e.get("id", {}).get("label", "")
        author = e.get("author", {}).get("name", {}).get("label", "")
        version = e.get("im:version", {}).get("label", "")
        rows.append({
            "review_id": review_id,
            "platform": "app_store_revolut_x",
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
    ap.add_argument("--max-pages", type=int, default=10)
    ap.add_argument("--delay", type=float, default=1.0)
    args = ap.parse_args()

    session = requests.Session()
    all_rows = []
    for c in tqdm(args.countries, desc="Revolut X App Store"):
        all_rows.extend(scrape_for_country(c, session, args.max_pages, args.delay))

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("No reviews scraped.")
        return

    # Structurally crypto; flag accordingly
    df["is_crypto"] = True
    df["is_trust_concern"] = False
    df["is_fx"] = False
    df["is_relevant"] = True
    df["matched_crypto"] = "revolut_x_app"
    df["matched_trust"] = ""
    df["matched_fx"] = ""

    print(f"\nTotal Revolut X iOS reviews: {len(df)}")
    print(f"Date range: {df['date'].min()} → {df['date'].max()}")
    print(f"Star rating distribution:")
    print(df['star_rating'].value_counts().sort_index().to_string())

    out = OUT_DIR / "revolut_x_app_store_raw.csv"
    df.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
