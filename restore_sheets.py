"""
restore_sheets.py
-----------------
Takes a coding_sample.xlsx that has only the `reviews` sheet (because Excel
saved it single-sheet) and rebuilds the codebook, focus_categories, and
instructions sheets so 08_aggregate_themes.py can read it.

Place this in the same folder as your coded xlsx. Adjust INPUT_PATH if needed.
"""

from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/clean/coding_sample.xlsx")
# If your file is in a different location, change to:
# INPUT_PATH = Path("coding_sample.xlsx")

CODEBOOK = [
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
    ("crypto", "Review is primarily about crypto features."),
    ("fx", "Review is primarily about currency conversion, cross-border "
           "transfers, FX rates, sending money abroad."),
    ("banking", "Review is primarily about general banking: cards, accounts, "
                "deposits, verification, support."),
]


def main():
    if not INPUT_PATH.exists():
        raise SystemExit(f"File not found: {INPUT_PATH}")

    # Load the existing reviews sheet (the coded data)
    reviews = pd.read_excel(INPUT_PATH, sheet_name="reviews")
    print(f"Loaded {len(reviews)} coded reviews from '{INPUT_PATH.name}'")

    # Rebuild the other sheets
    codebook_df = pd.DataFrame(CODEBOOK,
                               columns=["code", "definition", "survey_coverage"])
    focus_df = pd.DataFrame(PRIMARY_FOCUSES, columns=["focus", "definition"])
    instructions = pd.DataFrame({
        "step": ["coding instructions — see original codebook"],
        "action": ["reference only; not used by 08"]
    })

    # Write all four sheets back
    with pd.ExcelWriter(INPUT_PATH, engine="openpyxl") as writer:
        reviews.to_excel(writer, sheet_name="reviews", index=False)
        codebook_df.to_excel(writer, sheet_name="codebook", index=False)
        focus_df.to_excel(writer, sheet_name="focus_categories", index=False)
        instructions.to_excel(writer, sheet_name="instructions", index=False)

    print(f"Restored codebook, focus_categories, and instructions sheets.")
    print(f"File now has 4 sheets. 08_aggregate_themes.py should run.")


if __name__ == "__main__":
    main()