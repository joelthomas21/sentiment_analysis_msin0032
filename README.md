# Sentiment Analysis Pipeline 

Sentiment analysis of Revolut user reviews to assess baseline platform trust as a necessary condition for stablecoin adoption. Part of the MSIN0032 Management Science Dissertation, UCL.

## Overview

The pipeline scrapes, cleans, scores, and thematically codes user reviews of Revolut from five sources. It produces aggregate sentiment distributions, segment-level breakdowns, and theme-by-valence tables used in Part II of the dissertation.

## Pipeline Scripts (run in order)

| Step | Script | What it does |
|------|--------|--------------|
| — | `keyword_filter.py` | Shared module. Classifies review text into three flags: `is_crypto`, `is_trust_concern`, `is_fx`. Uses word-boundary regex to avoid false positives. Imported by other scripts, not run directly. |
| 0 | `00_refilter.py` | Applies keyword filter to all raw scraped data. Forces `is_crypto=True` on all Revolut X reviews. |
| 1 | `01_scrape_trustpilot.py` | Scrapes Revolut Trustpilot reviews via `__NEXT_DATA__` JSON extraction. |
| 2a | `02_scrape_google_play.py` | Scrapes main Revolut app reviews from Google Play (`com.revolut.revolut`). |
| 2b | `02b_scrape_google_play_revolut_x.py` | Scrapes Revolut X reviews from Google Play (`com.revolut.revolutx`). |
| 3a | `03_scrape_app_store.py` | Scrapes main Revolut app reviews from Apple App Store RSS (ID `932493382`). |
| 3b | `03b_scrape_app_store_revolut_x.py` | Scrapes Revolut X reviews from Apple App Store RSS (ID `6502614478`). |
| 4 | `04_clean_reviews.py` | Merges all five sources into a single dataset. Deduplicates, detects language (English only), parses dates, and runs fuzzy near-duplicate matching. |
| 5 | `05_run_vader.py` | Runs both baseline VADER and extended VADER (with 23 crypto-domain lexicon additions) on cleaned reviews. Outputs compound scores for both. |
| 6 | `06_analyse_distribution.py` | Produces sentiment histograms, bimodality coefficient, star-vs-VADER crosstab, Spearman correlation, and platform-level breakdowns. |
| 7 | `07_build_coding_sample.py` | Builds a stratified thematic subsample (default 100 negative, 100 positive, 40 neutral by VADER compound). Preserves existing codes when extending sample size. |
| 8 | `08_aggregate_themes.py` | Aggregates manually coded themes. Produces theme frequency tables, segment-level breakdowns, theme-by-focus-with-valence tables, VADER miscode audit, and robustness checks (star-stratified vs VADER-stratified). |

### Utility scripts

| Script | Purpose |
|--------|---------|
| `diagnose_preservation.py` | Debugging tool to verify code preservation when rebuilding the sample. |
| `restore_sheets.py` | Restores codebook/instructions sheets if they are lost during Excel operations. |

## Directory Structure

The scripts expect the following folder layout (created automatically on first run):

```
project_root/
├── data/
│   ├── raw/              # Output from scrapers (steps 1–3)
│   ├── filtered/         # Output from step 0
│   └── clean/            # Output from step 4 onwards
├── tables/               # Output tables from steps 6 and 8
├── figures/              # Output charts from step 6
├── keyword_filter.py
├── 00_refilter.py
├── 01_scrape_trustpilot.py
├── ...
├── 08_aggregate_themes.py
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10+.

```bash
# Clone the repository
git clone https://github.com/joelthomas21/sentiment-analysis-msin0032.git
cd sentiment-analysis-msin0032

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the Pipeline

**Steps 1–3 (scrapers)** should be run from a local machine, not a sandboxed environment, because Google Play and Apple App Store block non-residential IPs. Trustpilot has a login wall after approximately 10 pages.

```bash
# Scrape (run from local machine)
python 01_scrape_trustpilot.py
python 02_scrape_google_play.py
python 02b_scrape_google_play_revolut_x.py
python 03_scrape_app_store.py
python 03b_scrape_app_store_revolut_x.py

# Filter
python 00_refilter.py

# Clean and merge
python 04_clean_reviews.py

# Score sentiment
python 05_run_vader.py

# Analyse distribution
python 06_analyse_distribution.py

# Build coding sample
python 07_build_coding_sample.py

# After manual coding in the output Excel file, aggregate themes
python 08_aggregate_themes.py
```


