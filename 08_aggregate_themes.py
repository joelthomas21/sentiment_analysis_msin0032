"""
08_aggregate_themes.py
----------------------
After manually coding `data/clean/coding_sample.xlsx`, run this to produce
the thematic results tables for the Part II write-up.

Handles two types of VADER correction flags written in `coder_notes`:

  - `vader_miscode`  — VADER got the sentiment wrong. Review is EXCLUDED
                       from the human-corrected valence computation.

  - `true_neg`       — Your judgement says this review is genuinely negative,
  - `true_pos`         regardless of VADER's assignment.
  - `true_neu`       — Review is reassigned to that stratum in the corrected
                       output.

Case-insensitive. Flags can appear anywhere in the `coder_notes` text
(e.g. "vader_miscode - sarcasm").

Outputs:

1. tables/thematic_results.csv/md
   Theme × stratum frequency using VADER's original stratification.

2. tables/thematic_results_corrected.csv/md
   Same table, but with:
     - `vader_miscode` reviews dropped from counts
     - `true_*` flags reassigning reviews between strata
   If any flags were found, this table is the one to report in the
   dissertation, with the uncorrected version as a robustness check.

2b. tables/thematic_results_by_star.csv/md + tables/stratum_agreement.csv
    Theme × stratum using STAR RATINGS (1-2★=neg, 3★=neu, 4-5★=pos).
    Stars represent user intent; VADER represents lexical sentiment.
    Convergence across the two = validation; divergence = finding.
    stratum_agreement shows the confusion matrix.

3. tables/thematic_by_focus.csv/md
   Theme × primary focus (crypto / fx / banking). Does not depend on VADER
   stratification, hence robust to miscoding.

4. tables/focus_distribution.csv/md
   Per primary focus: n, mean compound, valence split.

5. tables/vader_miscode_audit.md
   Count and rate of VADER miscodes flagged during coding.

Usage:
    python 08_aggregate_themes.py
"""

from pathlib import Path

import pandas as pd

CLEAN_DIR = Path("data/clean")
TAB_DIR = Path("tables"); TAB_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def load_coded() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = CLEAN_DIR / "coding_sample.xlsx"
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run 07_build_coding_sample.py first.")
    reviews = pd.read_excel(path, sheet_name="reviews")
    codebook = pd.read_excel(path, sheet_name="codebook")
    return reviews, codebook


def to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def apply_miscode_corrections(reviews: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Read `coder_notes` and apply flags.

    Returns (corrected_df, audit_dict) where corrected_df:
      - has a new column `stratum_corrected` (may differ from `stratum`)
      - has a boolean column `vader_miscode_flag`
    Rows with vader_miscode are kept in the df but `stratum_corrected` is
    set to NaN so the caller can filter them out of valence counts.
    """
    df = reviews.copy()
    notes = df["coder_notes"].fillna("").astype(str).str.lower()

    # Initialize
    df["vader_miscode_flag"] = notes.str.contains(r"\bvader_miscode\b", regex=True)
    df["stratum_corrected"] = df["stratum"]

    # Reassignment flags (mutually exclusive; later ones overwrite earlier if
    # multiple appear — unlikely in practice, but flag it)
    reassign_map = {
        "true_neg": "negative",
        "true_pos": "positive",
        "true_neu": "neutral",
    }
    multi_flag_idx = []
    for flag, stratum in reassign_map.items():
        mask = notes.str.contains(rf"\b{flag}\b", regex=True)
        # Track rows with multiple flags for audit
        for idx in df[mask].index:
            existing = [f for f in reassign_map
                        if f != flag and f in notes.loc[idx]]
            if existing:
                multi_flag_idx.append(idx)
        df.loc[mask, "stratum_corrected"] = stratum

    # For miscoded rows with no explicit true_* flag, set corrected to NaN
    # so they are dropped from valence counts
    no_reassign = df["vader_miscode_flag"] & (df["stratum_corrected"] == df["stratum"])
    df.loc[no_reassign, "stratum_corrected"] = None

    # Count reassignments: rows where corrected stratum differs from original
    # (and is not NaN). This is the count that moved INTO each stratum.
    moved = df[df["stratum_corrected"].notna() &
               (df["stratum_corrected"] != df["stratum"])]
    audit = {
        "total_coded": len(df),
        "miscode_flags": int(df["vader_miscode_flag"].sum()),
        "reassigned_to_neg": int((moved["stratum_corrected"] == "negative").sum()),
        "reassigned_to_pos": int((moved["stratum_corrected"] == "positive").sum()),
        "reassigned_to_neu": int((moved["stratum_corrected"] == "neutral").sum()),
        "dropped_no_reassignment": int(df["stratum_corrected"].isna().sum()),
        "multi_flag_rows": sorted(set(multi_flag_idx)),
    }
    audit["miscode_rate"] = round(audit["miscode_flags"] / audit["total_coded"], 3)
    return df, audit


# ─────────────────────────────────────────────────────────────────────
# Aggregations
# ─────────────────────────────────────────────────────────────────────

def aggregate_by_stratum(reviews: pd.DataFrame,
                         codebook: pd.DataFrame,
                         stratum_col: str = "stratum") -> pd.DataFrame:
    """Theme × stratum frequency. Caller supplies which stratum column to use.

    Rows with NaN in the stratum column are dropped.
    """
    df = reviews[reviews[stratum_col].notna()].copy()
    code_cols = [c for c in df.columns if c.startswith("code_")]
    for c in code_cols:
        df[c] = to_int(df[c])

    strata_counts = df[stratum_col].value_counts().to_dict()
    n_total = len(df)

    rows = []
    for c in code_cols:
        code = c.replace("code_", "")
        pos = int(df.loc[df[stratum_col] == "positive", c].sum())
        neg = int(df.loc[df[stratum_col] == "negative", c].sum())
        neu = int(df.loc[df[stratum_col] == "neutral", c].sum())
        n_pos = strata_counts.get("positive", 0)
        n_neg = strata_counts.get("negative", 0)
        n_neu = strata_counts.get("neutral", 0)
        rows.append({
            "code": code,
            "pos_freq": pos,
            "pos_pct": round(pos / n_pos, 3) if n_pos else 0,
            "neg_freq": neg,
            "neg_pct": round(neg / n_neg, 3) if n_neg else 0,
            "neu_freq": neu,
            "neu_pct": round(neu / n_neu, 3) if n_neu else 0,
            "total_freq": pos + neg + neu,
            "net_valence": round((pos - neg) / n_total, 3) if n_total else 0,
        })
    out = (pd.DataFrame(rows)
           .merge(codebook, on="code", how="left")
           .sort_values("total_freq", ascending=False)
           .reset_index(drop=True))
    return out


def aggregate_by_focus(reviews: pd.DataFrame) -> pd.DataFrame:
    """Theme × primary focus. Does NOT use VADER stratum, so robust to miscodes."""
    code_cols = [c for c in reviews.columns if c.startswith("code_")]
    focus_cols = [c for c in reviews.columns if c.startswith("primary_focus_")]
    for c in code_cols + focus_cols:
        reviews[c] = to_int(reviews[c])

    rows = []
    for c in code_cols:
        code = c.replace("code_", "")
        row = {"code": code}
        for f in focus_cols:
            focus = f.replace("primary_focus_", "")
            subset = reviews[reviews[f] == 1]
            if len(subset) == 0:
                row[f"{focus}_n"] = 0
                row[f"{focus}_pct"] = 0.0
            else:
                n = int(subset[c].sum())
                row[f"{focus}_n"] = n
                row[f"{focus}_pct"] = round(n / len(subset), 3)
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(focus_cols):
        out["_total"] = out[[f"{f.replace('primary_focus_', '')}_n"
                             for f in focus_cols]].sum(axis=1)
        out = out.sort_values("_total", ascending=False).drop("_total", axis=1)
    return out.reset_index(drop=True)


def aggregate_by_focus_with_valence(reviews: pd.DataFrame,
                                    stratum_col: str = "stratum_corrected"
                                    ) -> pd.DataFrame:
    """Theme × primary focus × valence.

    Critical for interpretation: the raw theme × focus table collapses
    positive and negative mentions into one frequency, making it impossible
    to distinguish enablers (positive theme mentions) from barriers
    (negative theme mentions). This table splits them.

    Output columns per (segment, theme):
        pos, neg, neu : count of mentions at each valence
        total         : sum
        net           : pos - neg
        net_valence   : (pos - neg) / segment_n
    """
    if stratum_col not in reviews.columns:
        stratum_col = "stratum"

    df = reviews[reviews[stratum_col].notna()].copy()
    code_cols = [c for c in df.columns if c.startswith("code_")]
    focus_cols = [c for c in df.columns if c.startswith("primary_focus_")]
    for c in code_cols + focus_cols:
        df[c] = to_int(df[c])

    rows = []
    for f in focus_cols:
        focus = f.replace("primary_focus_", "")
        seg_df = df[df[f] == 1]
        n_seg = len(seg_df)
        if n_seg == 0:
            continue
        for c in code_cols:
            code = c.replace("code_", "")
            hits = seg_df[seg_df[c] == 1]
            if len(hits) == 0:
                continue
            pos = int((hits[stratum_col] == "positive").sum())
            neg = int((hits[stratum_col] == "negative").sum())
            neu = int((hits[stratum_col] == "neutral").sum())
            total = pos + neg + neu
            rows.append({
                "segment": focus,
                "segment_n": n_seg,
                "theme": code,
                "pos": pos,
                "neg": neg,
                "neu": neu,
                "total": total,
                "net": pos - neg,
                "net_valence": round((pos - neg) / n_seg, 3) if n_seg else 0,
            })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["segment", "total"], ascending=[True, False])
    return out.reset_index(drop=True)


def focus_distribution(reviews: pd.DataFrame,
                       compound_col: str,
                       stratum_col: str = "stratum") -> pd.DataFrame:
    df = reviews[reviews[stratum_col].notna()].copy()
    focus_cols = [c for c in df.columns if c.startswith("primary_focus_")]
    for c in focus_cols:
        df[c] = to_int(df[c])

    rows = []
    for f in focus_cols:
        focus = f.replace("primary_focus_", "")
        subset = df[df[f] == 1]
        if len(subset) == 0:
            continue
        stratum_share = subset[stratum_col].value_counts(normalize=True).to_dict()
        rows.append({
            "primary_focus": focus,
            "n": len(subset),
            "pct_of_sample": round(len(subset) / len(df), 3),
            "mean_compound": round(subset[compound_col].mean(), 3),
            "pct_positive": round(stratum_share.get("positive", 0), 3),
            "pct_negative": round(stratum_share.get("negative", 0), 3),
            "pct_neutral": round(stratum_share.get("neutral", 0), 3),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def write(df: pd.DataFrame, stem: str) -> None:
    (TAB_DIR / f"{stem}.csv").write_text(df.to_csv(index=False))
    (TAB_DIR / f"{stem}.md").write_text(df.to_markdown(index=False))


def main():
    reviews, codebook = load_coded()

    # Find compound column
    if "vader_ext_compound" in reviews.columns:
        compound_col = "vader_ext_compound"
    elif "vader_compound" in reviews.columns:
        compound_col = "vader_compound"
    else:
        compound_col = None

    # Apply miscode corrections
    reviews, audit = apply_miscode_corrections(reviews)

    # 1. Theme × stratum (uncorrected — VADER's original stratification)
    by_stratum = aggregate_by_stratum(reviews, codebook, "stratum")
    write(by_stratum, "thematic_results")
    print("Saved: tables/thematic_results.csv + .md")
    print("\nTop themes (VADER stratification):")
    print(by_stratum.head(10)[["code", "pos_freq", "neg_freq",
                               "net_valence", "survey_coverage"]]
          .to_string(index=False))

    # 2. Theme × stratum (corrected — after human miscode flags applied)
    if audit["miscode_flags"] > 0 or audit["reassigned_to_neg"] != 0 or \
       audit["reassigned_to_pos"] != 0 or audit["reassigned_to_neu"] != 0:
        by_stratum_corr = aggregate_by_stratum(reviews, codebook, "stratum_corrected")
        write(by_stratum_corr, "thematic_results_corrected")
        print("\nSaved: tables/thematic_results_corrected.csv + .md")
        print("\nTop themes (human-corrected stratification):")
        print(by_stratum_corr.head(10)[["code", "pos_freq", "neg_freq",
                                        "net_valence", "survey_coverage"]]
              .to_string(index=False))
    else:
        print("\n[i] No VADER corrections flagged — skipping corrected table.")

    # 2b. Theme × stratum (star-based stratification — user intent anchor).
    # Per supervisor feedback: stars represent overall user intent; VADER
    # represents lexical sentiment. Convergence across the two = validation.
    # Divergence = a finding about VADER's limitations.
    if "stratum_star" in reviews.columns:
        by_stratum_star = aggregate_by_stratum(reviews, codebook, "stratum_star")
        write(by_stratum_star, "thematic_results_by_star")
        print("\nSaved: tables/thematic_results_by_star.csv + .md")
        print("\nTop themes (star-based stratification):")
        print(by_stratum_star.head(10)[["code", "pos_freq", "neg_freq",
                                        "net_valence", "survey_coverage"]]
              .to_string(index=False))

        # Agreement audit between VADER and star stratification
        agree_mask = reviews["stratum"] == reviews["stratum_star"]
        agreement_rate = round(agree_mask.mean(), 3)
        conf = pd.crosstab(reviews["stratum"], reviews["stratum_star"],
                           margins=True, margins_name="Total")
        write(conf.reset_index(), "stratum_agreement")
        print(f"\nVADER vs star agreement: {agreement_rate:.1%}")
        print("\nConfusion matrix (rows = VADER, cols = star):")
        print(conf.to_string())

    # 3. Theme × focus (robust — no VADER dependency)
    focus_cols_present = [c for c in reviews.columns
                          if c.startswith("primary_focus_")]
    if focus_cols_present:
        by_focus = aggregate_by_focus(reviews)
        write(by_focus, "thematic_by_focus")
        print("\nSaved: tables/thematic_by_focus.csv + .md")
        print("\nTheme × primary focus (top 8):")
        print(by_focus.head(8).to_string(index=False))

        # 3b. Theme × focus × valence — the critical interpretive table.
        # Raw theme × focus frequencies cannot distinguish "users love our FX
        # rates" (enabler) from "users hate our FX rates" (barrier). This
        # table splits each mention by valence using the corrected stratum.
        by_focus_val = aggregate_by_focus_with_valence(reviews, "stratum_corrected")
        if len(by_focus_val):
            write(by_focus_val, "thematic_by_focus_with_valence")
            print("\nSaved: tables/thematic_by_focus_with_valence.csv + .md")
            print("\nTheme × primary focus × valence (by segment, top 6 per segment):")
            for seg in by_focus_val["segment"].unique():
                sub = by_focus_val[by_focus_val["segment"] == seg].head(6)
                if len(sub):
                    n_seg = int(sub["segment_n"].iloc[0])
                    print(f"\n  {seg.upper()} (n={n_seg}):")
                    print(sub[["theme", "pos", "neg", "neu",
                               "net", "net_valence"]].to_string(index=False))

        if compound_col:
            # Both VADER-stratified and corrected focus distributions
            focus_dist = focus_distribution(reviews, compound_col, "stratum")
            write(focus_dist, "focus_distribution")
            print("\nSaved: tables/focus_distribution.csv + .md")
            print("\nPrimary focus distribution (VADER stratification):")
            print(focus_dist.to_string(index=False))

            if audit["miscode_flags"] > 0 or audit["reassigned_to_neg"] != 0 or \
               audit["reassigned_to_pos"] != 0 or audit["reassigned_to_neu"] != 0:
                focus_dist_corr = focus_distribution(reviews, compound_col,
                                                     "stratum_corrected")
                write(focus_dist_corr, "focus_distribution_corrected")
                print("\nSaved: tables/focus_distribution_corrected.csv + .md")
    else:
        print("\n[!] No primary_focus_* columns found.")

    # 4. VADER miscode audit (always written for methodology transparency)
    audit_md = f"""# VADER miscoding audit

Written during thematic coding by flagging `vader_miscode`, `true_neg`,
`true_pos`, or `true_neu` in the `coder_notes` column.

| metric | value |
|---|---|
| Total reviews coded | {audit['total_coded']} |
| `vader_miscode` flagged | {audit['miscode_flags']} |
| Miscode rate | {audit['miscode_rate']:.1%} |
| Reassigned → negative | {audit['reassigned_to_neg']:+d} |
| Reassigned → positive | {audit['reassigned_to_pos']:+d} |
| Reassigned → neutral | {audit['reassigned_to_neu']:+d} |
| Dropped (miscode, no reassignment) | {audit['dropped_no_reassignment']} |
| Rows with multiple flags | {len(audit['multi_flag_rows'])} |

Rows with multiple flags (review indices, if any): {audit['multi_flag_rows']}

## For the methodology section

A {audit['miscode_rate']:.1%} miscoding rate on the thematic subsample
(n = {audit['total_coded']}) was identified through manual review. Sources
of miscoding include sarcasm, complex negation, and context-dependent
polarity that VADER's lexicon-based approach cannot resolve. The
human-corrected valence table (`thematic_results_corrected`) is reported
as the primary result, with the uncorrected table (`thematic_results`)
retained as a robustness check. The thematic × focus cross-tabulation
(`thematic_by_focus`) does not depend on VADER stratification and is
therefore robust to this source of error.
"""
    (TAB_DIR / "vader_miscode_audit.md").write_text(audit_md)
    print("\nSaved: tables/vader_miscode_audit.md")
    print(f"\nMiscode rate: {audit['miscode_rate']:.1%} "
          f"({audit['miscode_flags']} of {audit['total_coded']})")

    # Emergent themes
    emergent = reviews["new_theme"].dropna().astype(str).str.strip()
    emergent = emergent[emergent != ""]
    if len(emergent):
        print(f"\n{len(emergent)} reviews flagged emergent themes:")
        print(emergent.value_counts().head(10).to_string())


if __name__ == "__main__":
    main()