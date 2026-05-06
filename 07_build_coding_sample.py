"""
07_build_coding_sample.py
-------------------------
Builds a stratified subsample for manual thematic coding.

CODE PRESERVATION:
If `data/clean/coding_sample.xlsx` already exists from a previous run,
this script reads it and PRESERVES existing codes by matching on
review_id. For the new sample:
  - Reviews that exist in the old coded file: copy codes across
  - Reviews that are new: leave blank for manual coding

Output reports how many reviews need new coding versus how many were
preserved. Typical case after filter change: ~80% preserved,
~15–20 new reviews to code.

Usage:
    python 07_build_coding_sample.py --n-neg 50 --n-pos 50 --n-neu 20

For each NEW review while coding, fill in:
  - Each `code_*` column: 1 if theme is present, 0 if absent
  - Exactly ONE `primary_focus_*` column with 1 (the review's main topic)
  - Optionally tick multiple `secondary_focus_*` columns if relevant
  - Add emergent themes to `new_theme`
  - VADER miscode flags in `coder_notes` (vader_miscode / true_neg /
    true_pos / true_neu)
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime

import pandas as pd

CLEAN_DIR = Path("data/clean")


# ─────────────────────────────────────────────────────────────────────
# Codebook.
# ─────────────────────────────────────────────────────────────────────

CODEBOOK = [
    # (code, definition, survey_coverage)
    ("withdrawal_reliability",
     "User reports problems withdrawing fiat or crypto, or praises smooth withdrawals.",
     "partial (Q16: instant 1:1 redemption)"),
    ("account_freezing",
     "Account locked, blocked, suspended, closed without clear reason.",
     "GAP"),
    ("customer_support",
     "Support responsiveness, helpfulness, or absence thereof.",
     "GAP"),
    ("fee_transparency",
     "Fees hidden, surprise charges, or praise for transparent pricing.",
     "partial (Q7: 0% return; Q16: lower fees)"),
    ("reserve_concerns",
     "Doubt about whether Revolut holds sufficient reserves or is solvent.",
     "direct (Q7: reserves)"),
    ("peg_depeg",
     "Reference to a stablecoin losing or maintaining its peg.",
     "direct (Q7: peg risk; Q18: depeg behaviour)"),
    ("regulatory_trust",
     "Comments on Revolut's regulated status, licence, FCA/MiCA references.",
     "direct (Q7: regulatory uncertainty)"),
    ("ui_ux",
     "App interface quality, ease of use, visual design comments.",
     "partial (Q15/Q16 features)"),
    ("competitor_comparison",
     "Comparison to Coinbase, Binance, Wise, traditional banks, PayPal.",
     "direct (Q7: prefer established issuer)"),
    ("transfer_speed",
     "Speed of transfers, instant payments, or delays.",
     "partial (Q16: instant redemption)"),
    ("security_fraud",
     "Hacks, phishing, fraud protection, scams, unauthorised access.",
     "partial (Q16: banking licence backing)"),
    ("price_spread_slippage",
     "Crypto buy/sell spreads, slippage, perceived price manipulation.",
     "GAP"),
    ("fx_rate_quality",
     "Complaints or praise about Revolut's FX exchange rate, markup, weekend surcharge.",
     "direct (Q8/Q9: current FX method; Q11: cost-only switching)"),
    ("cross_border_use",
     "Use case is sending money abroad, working abroad, remittance, travel.",
     "direct (Q4: cross-border frequency; Q15: send abroad)"),
]

PRIMARY_FOCUSES = [
    ("crypto", "Review is primarily about crypto features: trading, "
               "Revolut X, coin holdings, staking, DeFi, token transfers."),
    ("fx", "Review is primarily about currency conversion, cross-border "
           "transfers, FX rates, sending money abroad, multi-currency accounts."),
    ("banking", "Review is primarily about general banking: cards, personal/"
                "joint accounts, deposits, savings, bill pay, KYC/verification, "
                "customer support (without a specific crypto or FX anchor)."),
]


def star_to_stratum(star) -> str:
    """Map star rating to stratum. 1-2★ = negative, 3★ = neutral, 4-5★ = positive."""
    if pd.isna(star):
        return "unknown"
    if star <= 2:
        return "negative"
    if star >= 4:
        return "positive"
    return "neutral"


def backup_existing(path: Path) -> Path | None:
    """Backup the existing coding_sample.xlsx so we don't lose old codes."""
    if not path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}_backup_{ts}{path.suffix}")
    shutil.copy(path, backup)
    return backup


def load_existing_codes(path: Path) -> pd.DataFrame | None:
    """Read existing coded sheet; return minimal dataframe keyed by review_id."""
    if not path.exists():
        return None
    try:
        old = pd.read_excel(path, sheet_name="reviews")
    except Exception as e:
        print(f"  [!] could not read existing codes: {e}")
        return None
    if "review_id" not in old.columns:
        return None

    # Keep only coding columns (codes, focus, notes)
    keep_cols = ["review_id"]
    for col in old.columns:
        if (col.startswith("code_") or col.startswith("primary_focus_")
                or col.startswith("secondary_focus_")
                or col in ("new_theme", "coder_notes")):
            keep_cols.append(col)

    old_codes = old[keep_cols].copy()
    # Normalise review_id for merging
    old_codes["review_id"] = old_codes["review_id"].astype(str)
    return old_codes


def merge_existing_codes(new_sample: pd.DataFrame,
                         old_codes: pd.DataFrame | None) -> tuple[pd.DataFrame, int, int]:
    """Left-join existing codes onto the new sample. Returns (df, n_preserved, n_new)."""
    if old_codes is None:
        return new_sample, 0, len(new_sample)

    # Ensure same dtype for the merge key
    new_sample["review_id"] = new_sample["review_id"].astype(str)

    # For every code column in old_codes, overwrite the zero-initialised
    # value in new_sample with the old code (if present).
    code_cols = [c for c in old_codes.columns if c != "review_id"]

    # Find which rows have matches in old_codes
    old_ids = set(old_codes["review_id"])
    new_has_old = new_sample["review_id"].isin(old_ids)
    n_preserved = int(new_has_old.sum())
    n_new = len(new_sample) - n_preserved

    # Merge: bring old values in under _old suffix, then coalesce
    merged = new_sample.merge(old_codes, on="review_id", how="left",
                              suffixes=("", "_old"))
    for col in code_cols:
        old_col = f"{col}_old"
        if old_col in merged.columns:
            # Use old value where present, otherwise keep the new (zero/empty) init
            merged[col] = merged[old_col].combine_first(merged[col])
            merged = merged.drop(columns=[old_col])

    # Coerce code columns back to int
    for col in code_cols:
        if col.startswith("code_") or col.startswith("primary_focus_") \
                or col.startswith("secondary_focus_"):
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)
        else:
            merged[col] = merged[col].fillna("").astype(object)

    return merged, n_preserved, n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-neg", type=int, default=100)
    ap.add_argument("--n-pos", type=int, default=100)
    ap.add_argument("--n-neu", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lexicon", choices=["base", "ext"], default="ext")
    args = ap.parse_args()

    compound_col = ("vader_compound" if args.lexicon == "base"
                    else "vader_ext_compound")

    out_path = CLEAN_DIR / "coding_sample.xlsx"

    # Back up existing file if present, and load old codes
    backup = backup_existing(out_path)
    if backup:
        print(f"Backed up existing file to: {backup}")
    old_codes = load_existing_codes(out_path)
    if old_codes is not None:
        print(f"Loaded existing codes for {len(old_codes)} reviews.")

    # Build new sample from the freshly refiltered data
    df = pd.read_csv(CLEAN_DIR / "reviews_scored.csv")

    # Sampling strategy with code preservation:
    #
    # When the target sample size exceeds the size of the existing coded set
    # (e.g. extending 120→240), we want every previously-coded review that
    # still qualifies for its stratum to be preserved, and the remainder of
    # the target n to be filled from un-coded reviews. This avoids losing
    # coded work to random reshuffling.
    #
    # For each stratum:
    #   1. Take all previously-coded reviews that fall in this stratum's
    #      compound band (there will be up to the initial per-stratum n)
    #   2. Shuffle the remaining un-coded pool with the fixed seed
    #   3. Take (n_target - n_preserved) from the shuffled un-coded pool
    #
    # This preserves the superset property: extending n_neg from 50→100 keeps
    # the original 50 coded negatives and adds 50 new ones.

    coded_ids = (set(old_codes["review_id"].astype(str))
                 if old_codes is not None else set())

    def stratified_sample_preserve(pool: pd.DataFrame, n_target: int,
                                   seed: int) -> pd.DataFrame:
        if len(pool) == 0:
            return pool
        pool = pool.copy()
        pool["review_id"] = pool["review_id"].astype(str)

        preserved = pool[pool["review_id"].isin(coded_ids)]
        remaining = pool[~pool["review_id"].isin(coded_ids)]

        # Take all preserved (up to n_target)
        if len(preserved) >= n_target:
            return preserved.sample(frac=1.0, random_state=seed).head(n_target)

        # Otherwise take all preserved + fill from remaining
        n_to_fill = n_target - len(preserved)
        filler = remaining.sample(frac=1.0, random_state=seed).head(n_to_fill)
        return pd.concat([preserved, filler], ignore_index=False)

    neg_pool = df[df[compound_col] < -0.3]
    pos_pool = df[df[compound_col] > 0.3]
    neu_pool = df[df[compound_col].between(-0.1, 0.1)]

    neg = stratified_sample_preserve(neg_pool, args.n_neg, args.seed)
    pos = stratified_sample_preserve(pos_pool, args.n_pos, args.seed)
    neu = stratified_sample_preserve(neu_pool, args.n_neu, args.seed)

    sample = pd.concat([neg, pos, neu], ignore_index=True)
    sample["stratum"] = (["negative"] * len(neg)
                        + ["positive"] * len(pos)
                        + ["neutral"] * len(neu))

    # Parallel star-based stratum for robustness check
    sample["stratum_star"] = sample["star_rating"].apply(star_to_stratum)

    # Initialise coding columns (will be filled in from old codes where possible)
    for code, _, _ in CODEBOOK:
        sample[f"code_{code}"] = 0
    for focus, _ in PRIMARY_FOCUSES:
        sample[f"primary_focus_{focus}"] = 0
        sample[f"secondary_focus_{focus}"] = 0
    sample["new_theme"] = ""
    sample["coder_notes"] = ""

    # Revolut X reviews are structurally crypto (dedicated crypto app).
    # Pre-assign primary_focus_crypto=1 so the coder only needs to fill in
    # themes, not figure out the segment. Coder can override if they
    # disagree, but the default saves a tick per row.
    if "platform" in sample.columns:
        rx_mask = sample["platform"].astype(str).str.contains("revolut_x", na=False)
        n_rx = int(rx_mask.sum())
        if n_rx > 0:
            sample.loc[rx_mask, "primary_focus_crypto"] = 1
            print(f"Auto-tagged {n_rx} Revolut X reviews as primary_focus_crypto "
                  f"(structurally crypto — coder still fills in themes).")

    # Merge in existing codes
    sample, n_preserved, n_new = merge_existing_codes(sample, old_codes)
    print(f"\nCode preservation:")
    print(f"  Reviews with preserved codes: {n_preserved}")
    print(f"  Reviews needing new coding:   {n_new}")

    # Flag new-to-code reviews visibly in the coder_notes (only if empty)
    if old_codes is not None:
        old_ids = set(old_codes["review_id"].astype(str))
        for idx, rid in enumerate(sample["review_id"].astype(str)):
            if rid not in old_ids and not str(sample.loc[idx, "coder_notes"]).strip():
                sample.loc[idx, "coder_notes"] = "[NEW — needs coding]"

    # Reorder columns for ergonomics
    core = ["stratum", "stratum_star", "platform", "date", "star_rating",
            compound_col, "word_count", "text"]
    primary_cols = [f"primary_focus_{f}" for f, _ in PRIMARY_FOCUSES]
    secondary_cols = [f"secondary_focus_{f}" for f, _ in PRIMARY_FOCUSES]
    code_cols = [f"code_{c}" for c, _, _ in CODEBOOK]
    tail = ["new_theme", "coder_notes"]
    ordered = core + primary_cols + secondary_cols + code_cols + tail
    rest = [c for c in sample.columns if c not in ordered]
    sample = sample[ordered + rest]

    # Write output
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        sample.to_excel(writer, sheet_name="reviews", index=False)

        codebook_df = pd.DataFrame(CODEBOOK,
                                   columns=["code", "definition", "survey_coverage"])
        codebook_df.to_excel(writer, sheet_name="codebook", index=False)

        focus_df = pd.DataFrame(PRIMARY_FOCUSES, columns=["focus", "definition"])
        focus_df.to_excel(writer, sheet_name="focus_categories", index=False)

        instructions = pd.DataFrame({
            "step": [
                "0. If preserving codes",
                "1. Primary focus",
                "2. Secondary focus",
                "3. Themes",
                "4. Emergent themes",
                "5. VADER miscode",
                "6. Notes",
            ],
            "action": [
                "Reviews carried over from a previous coded run retain their "
                "codes. Only reviews marked '[NEW — needs coding]' in "
                "coder_notes require new coding.",
                "For each review, put 1 in EXACTLY ONE primary_focus_* column "
                "(crypto / fx / banking). Pick the dominant topic.",
                "Optional: put 1 in any secondary_focus_* columns if the review "
                "spans two segments.",
                "For each code_* column, put 1 if that theme is present in "
                "the review, 0 if not. A review can have many themes.",
                "If you see a recurring theme not in the codebook, write it "
                "in the new_theme column.",
                "Write 'vader_miscode' in coder_notes if VADER got the "
                "sentiment wrong. Optionally add 'true_neg', 'true_pos', or "
                "'true_neu' to reassign the stratum.",
                "Use coder_notes for judgement calls on ambiguous reviews. "
                "When you finish coding a '[NEW]' review, delete the "
                "'[NEW — needs coding]' marker.",
            ],
        })
        instructions.to_excel(writer, sheet_name="instructions", index=False)

    print(f"\nSaved: {out_path}")
    print(f"Negative stratum: {len(neg)} | Positive: {len(pos)} | Neutral: {len(neu)}")
    print(f"Total rows: {len(sample)}")
    print("\nSheets: reviews | codebook | focus_categories | instructions")


if __name__ == "__main__":
    main()