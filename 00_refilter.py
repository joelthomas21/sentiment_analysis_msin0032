"""
00_refilter.py
--------------
Re-applies the keyword filter to existing raw scraped files using the
robust word-boundary regex in keyword_filter.py. Produces new filtered
CSVs with TRIPLE classification: is_crypto, is_trust_concern, is_fx.

Avoids re-scraping. Takes data/raw/*_raw.csv as input.

Output: data/raw/*_filtered.csv

Usage:
    python 00_refilter.py

After this, run 04_clean_reviews.py to deduplicate, date-filter, etc.
"""

from pathlib import Path

import pandas as pd

from keyword_filter import classify

RAW_DIR = Path("data/raw")


def refilter_file(name: str) -> dict:
    path = RAW_DIR / name
    if not path.exists():
        print(f"  [!] missing {path}")
        return {"crypto": 0, "trust": 0, "fx": 0, "relevant": 0}

    df = pd.read_csv(path)
    initial = len(df)
    print(f"\n{name}: {initial} rows")

    if df.empty:
        return {"crypto": 0, "trust": 0, "fx": 0, "relevant": 0}

    # Apply filter
    results = df.apply(
        lambda row: classify(row.get("text", ""), row.get("title", "")),
        axis=1
    )
    results_df = pd.DataFrame(list(results))
    for col in ["is_crypto", "is_trust_concern", "is_fx", "is_relevant",
                "matched_crypto", "matched_trust", "matched_fx"]:
        df[col] = results_df[col].values

    # Revolut X reviews are structurally crypto by virtue of the app. Force
    # is_crypto=True regardless of keyword match (users may not explicitly
    # say "crypto" when the app itself is crypto-only).
    if "revolut_x" in name:
        df["is_crypto"] = True
        df["is_relevant"] = True
        df["matched_crypto"] = df["matched_crypto"].where(
            df["matched_crypto"].astype(bool),
            "revolut_x_app"
        )
        print(f"  [i] Revolut X app: forcing is_crypto=True for all rows")

    crypto_n = int(df["is_crypto"].sum())
    trust_n = int(df["is_trust_concern"].sum())
    fx_n = int(df["is_fx"].sum())
    relevant_n = int(df["is_relevant"].sum())

    # Overlap counts (informative for methodology)
    crypto_only = int((df["is_crypto"] & ~df["is_trust_concern"] & ~df["is_fx"]).sum())
    fx_only = int((~df["is_crypto"] & ~df["is_trust_concern"] & df["is_fx"]).sum())
    trust_only = int((~df["is_crypto"] & df["is_trust_concern"] & ~df["is_fx"]).sum())
    all_three = int((df["is_crypto"] & df["is_trust_concern"] & df["is_fx"]).sum())

    print(f"  crypto:            {crypto_n:>5}  ({crypto_n/initial:.1%})")
    print(f"  trust-concern:     {trust_n:>5}  ({trust_n/initial:.1%})")
    print(f"  fx:                {fx_n:>5}  ({fx_n/initial:.1%})")
    print(f"  relevant (union):  {relevant_n:>5}  ({relevant_n/initial:.1%})")
    print(f"  crypto-only:       {crypto_only:>5}")
    print(f"  fx-only:           {fx_only:>5}  (captured by new FX filter)")
    print(f"  trust-only:        {trust_only:>5}")
    print(f"  all three:         {all_three:>5}")

    # Save the relevant subset (any of crypto / trust / fx)
    platform = name.replace("_raw.csv", "")
    out = RAW_DIR / f"{platform}_filtered.csv"
    df[df["is_relevant"]].to_csv(out, index=False)
    print(f"  saved: {out}  (n = {relevant_n})")

    # Sample audit for each category
    for cat, col, mcol in [("crypto", "is_crypto", "matched_crypto"),
                           ("fx", "is_fx", "matched_fx")]:
        sample = df[df[col]].head(2)
        if len(sample):
            print(f"  sample {cat} matches:")
            for _, r in sample.iterrows():
                terms = (r[mcol] or "")[:60]
                text = (r["text"] or "")[:80].replace("\n", " ")
                print(f"    [{terms}] → {text}")

    return {"crypto": crypto_n, "trust": trust_n, "fx": fx_n, "relevant": relevant_n}


def main():
    totals = {"crypto": 0, "trust": 0, "fx": 0, "relevant": 0}
    file_list = [
        "trustpilot_raw.csv",
        "google_play_raw.csv",
        "app_store_raw.csv",
        # Revolut X supplementary scrapes (crypto segment boost)
        "revolut_x_google_play_raw.csv",
        "revolut_x_app_store_raw.csv",
    ]
    for name in file_list:
        counts = refilter_file(name)
        for k, v in counts.items():
            totals[k] += v

    print("\n" + "─" * 50)
    print("Total relevant reviews across all sources:")
    print(f"  crypto:        {totals['crypto']}")
    print(f"  trust-concern: {totals['trust']}")
    print(f"  fx:            {totals['fx']}")
    print(f"  relevant:      {totals['relevant']}")


if __name__ == "__main__":
    main()
