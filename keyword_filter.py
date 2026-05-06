"""
keyword_filter.py
-----------------
Robust keyword matching for Revolut review filtering.

Uses word-boundary regex matching to avoid false positives. Naive substring
matching catches:
    "eth" in "whether", "something"
    "dai" in "daily"
    "defi" in "definitely"
    "aml" in "seamless"
    "matic" in "automatic"
All of these are eliminated by \\b word boundaries.

Produces TRIPLE classification aligned to the three user segments the
RevEUR decision depends on:
    is_crypto        — crypto-native segment vocabulary
    is_trust_concern — operational-trust vocabulary (any user type)
    is_fx            — FX/cross-border product vocabulary
    is_relevant      — union of the above

Rationale for triple classification:
    The dissertation analyses three user segments that drive different
    parameters of the Monte Carlo model:
      - crypto-native users       → conversion rate (additive volume)
      - FX-active users           → migration rate (cannibalisation pool)
      - general banking users     → platform-trust floor
    The filter captures all three explicitly so each segment has empirical
    coverage in the cleaned sample. Without explicit FX vocabulary, FX-
    active users only entered the sample when they also had an operational
    complaint, meaning calm FX-product criticism was systematically missed.

FX vocabulary was chosen to be SPECIFIC (product/operation terms), not
generic (travel/holiday terms). Specific FX vocabulary catches complaint
reviews without pulling in vague holiday praise.
"""

import re


# ─────────────────────────────────────────────────────────────────────
# Crypto vocabulary.
# ─────────────────────────────────────────────────────────────────────

CRYPTO_KEYWORDS_FULL = [
    "crypto", "cryptocurrency", "cryptocurrencies",
    "bitcoin", "ethereum", "stablecoin", "stablecoins",
    "tether", "blockchain", "revolut x", "defi",
    "hodl", "coinbase", "binance", "ledger", "airdrop",
    "litecoin", "ripple", "solana", "polygon", "dogecoin",
    "altcoin", "altcoins", "busd", "usdt", "usdc",
    "stakings", "staking", "liquidity pool",
    "rug pull", "rugpull", "rug pulled",
    "depeg", "depegged",
    "crypto wallet", "hot wallet", "cold wallet",
    "nft", "nfts",
]

CRYPTO_KEYWORDS_SHORT = [
    "btc", "eth", "dai", "xrp", "shib", "matic",
    "dex", "cex",
]


# ─────────────────────────────────────────────────────────────────────
# Trust/banking concern vocabulary. Operational-trust signals relevant
# across all user types.
# ─────────────────────────────────────────────────────────────────────

TRUST_CONCERN_KEYWORDS = [
    # Account access
    "frozen", "freezing", "froze my",
    "locked", "blocked my account", "account blocked",
    "suspended", "closed my account", "account closed",
    "restricted", "restriction",
    # Verification
    "verification", "verify my identity", "kyc", "aml",
    # Withdrawal/money access
    "can't withdraw", "cannot withdraw", "won't let me withdraw",
    "money stuck", "funds held", "funds frozen",
    "withdrawal stuck", "withdrawal delayed", "withdrawal blocked",
    "transfer stuck", "money missing",
    # Support
    "customer support", "customer service", "support team",
    "no response", "unresponsive", "no reply", "unhelpful support",
    # Fraud/security
    "scam", "scammed", "scammers",
    "fraud", "fraudulent", "phishing",
    "hacked", "unauthorised", "unauthorized",
]


# ─────────────────────────────────────────────────────────────────────
# FX / cross-border product vocabulary. Specific to the FX product and
# cross-border-transfer use case. Chosen to catch product complaints
# (markup, rate, spread) and not vague travel praise.
# ─────────────────────────────────────────────────────────────────────

FX_KEYWORDS = [
    # FX rates, fees, markups (the product criticism signal)
    "exchange rate", "exchange rates",
    "conversion rate", "conversion rates",
    "fx rate", "fx rates", "fx fee", "fx fees",
    "fx markup", "currency markup",
    "markup", "marked up", "mark-up", "mark up",
    "weekend rate", "weekend surcharge", "weekend markup", "weekend fee",
    "spread", "spreads", "mid-market", "mid market", "interbank rate",
    # Currency operations
    "currency conversion", "currency exchange",
    "convert currency", "convert currencies",
    "converting euros", "converting pounds", "converting dollars",
    "multi-currency", "multi currency", "multiple currencies",
    "multi-currency account", "multicurrency",
    # Cross-border transfers / remittance
    "cross-border", "cross border",
    "send abroad", "sending abroad", "sending money abroad",
    "international transfer", "international transfers",
    "international payment", "international payments",
    "overseas transfer", "overseas payment",
    "remittance", "remittances",
    "working abroad", "living abroad", "moving abroad",
    "expat", "expats", "expatriate",
    # Protocols / infrastructure (high-signal for FX-active users)
    "iban", "sepa", "swift transfer", "swift payment", "swift wire",
    # Competitor comparisons implying FX substitution
    "wise", "transferwise",
    "western union", "moneygram",
    "xe money", "remitly",
]


# ─────────────────────────────────────────────────────────────────────
# Compile patterns with word boundaries.
# ─────────────────────────────────────────────────────────────────────

def _compile_pattern(terms: list[str]) -> re.Pattern:
    """Build a case-insensitive regex with word boundaries."""
    escaped = sorted([re.escape(t) for t in terms], key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


CRYPTO_PATTERN = _compile_pattern(CRYPTO_KEYWORDS_FULL + CRYPTO_KEYWORDS_SHORT)
TRUST_PATTERN = _compile_pattern(TRUST_CONCERN_KEYWORDS)
FX_PATTERN = _compile_pattern(FX_KEYWORDS)


def classify(text: str, title: str = "") -> dict:
    """Classify a review by keyword matching.

    Returns a dict with:
        is_crypto, is_trust_concern, is_fx:  bool
        is_relevant:                         bool (union)
        matched_crypto, matched_trust, matched_fx:  comma-separated audit
    """
    combined = f"{title or ''}. {text or ''}".strip()

    crypto_matches = CRYPTO_PATTERN.findall(combined)
    trust_matches = TRUST_PATTERN.findall(combined)
    fx_matches = FX_PATTERN.findall(combined)

    return {
        "is_crypto": bool(crypto_matches),
        "is_trust_concern": bool(trust_matches),
        "is_fx": bool(fx_matches),
        "is_relevant": bool(crypto_matches or trust_matches or fx_matches),
        "matched_crypto": "; ".join(sorted({m.lower() for m in crypto_matches})),
        "matched_trust": "; ".join(sorted({m.lower() for m in trust_matches})),
        "matched_fx": "; ".join(sorted({m.lower() for m in fx_matches})),
    }


# ─────────────────────────────────────────────────────────────────────
# Self-test.
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TESTS = [
        # (text, expected_crypto, expected_fx, reason)
        ("I'm not sure whether I should trust this app", False, False,
         "'whether' contains 'eth' — FP guard"),
        ("I use this daily for everything", False, False,
         "'daily' contains 'dai' — FP guard"),
        ("The app runs on automatic updates", False, False,
         "'automatic' contains 'matic' — FP guard"),
        ("I bought some ETH last week", True, False,
         "crypto TP"),
        ("Revolut's weekend FX markup is ridiculous", False, True,
         "FX criticism — NEW: would have been missed before"),
        ("I switched to Wise because their rates are better", False, True,
         "competitor comparison — NEW"),
        ("Bad exchange rate on EUR-GBP conversion", False, True,
         "exchange rate criticism — NEW"),
        ("Great for international transfers", False, True,
         "cross-border use — NEW"),
        ("Account frozen, can't access my euros", False, False,
         "trust concern only; 'euros' alone doesn't trigger FX"),
        ("Used my SEPA transfer to send money abroad", False, True,
         "SEPA + send abroad"),
        ("Love using Revolut for holidays", False, False,
         "vague holiday praise should NOT trigger"),
        ("Great app, love the features", False, False,
         "generic positive should NOT trigger"),
        ("Bitcoin withdrawal stuck for days", True, False,
         "crypto + trust concern, no FX"),
        ("FX rates are competitive and the IBAN works", False, True,
         "positive FX review"),
    ]

    print(f"{'text':<55} {'crypto':>7} {'trust':>7} {'fx':>7}  {'exp C':>5} {'exp F':>5}  ok  reason")
    print("─" * 130)
    all_pass = True
    for text, exp_crypto, exp_fx, reason in TESTS:
        r = classify(text)
        crypto_ok = r["is_crypto"] == exp_crypto
        fx_ok = r["is_fx"] == exp_fx
        ok = "✓" if (crypto_ok and fx_ok) else "✗"
        if not (crypto_ok and fx_ok):
            all_pass = False
        print(f"{text[:53]:<55} {str(r['is_crypto']):>7} "
              f"{str(r['is_trust_concern']):>7} {str(r['is_fx']):>7}  "
              f"{str(exp_crypto):>5} {str(exp_fx):>5}  {ok:>2}  {reason}")
    print()
    print("ALL PASS" if all_pass else "SOME FAILURES — REVIEW ABOVE")