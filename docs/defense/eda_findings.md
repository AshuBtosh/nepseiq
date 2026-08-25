# EDA Findings

## Dataset Summary
- Stocks in basket: 11 — AHPC, CBBL, CHCL, EBL, GBIME, NABIL, NLG, NLIC, RIDI, SICL, SKBBL
- Date range: 2005-02-09 (CBBL, earliest) – 2026-08-24 (all symbols, latest pull date)
- Total rows (post `load_raw()` cleaning, incl. ADR-0007 OHLC merges + NLG listing-day fix): 34,342
- Missing values: `per_change` had 6 NaNs pre-cleaning; all other columns complete. Not
  imputed — `per_change` is not used as a feature (see Preprocessing Decisions).

## Baselines
- Class balance `target_1d`: 43.90% up / 56.10% down (n=34,331) → majority-class baseline **56.10%**
- Class balance `target_5d`: 45.86% up / 54.14% down (n=34,287) → majority-class baseline **54.14%**
- Both fall within CLAUDE.md §9's expected realistic range (52–58%); `target_1d` sits at
  the higher end.

## Findings

1. **CBBL and RIDI are flagged for different, non-interchangeable reasons.** Session
   density (rows/year of listed life) ranges from 149.9 (CBBL) to 228.5 (EBL). CBBL is a
   genuine **liquidity concern**: 21.5 years listed but the thinnest trading of any
   basket member. RIDI's density is actually mid-pack (227.1/yr, comparable to the
   banking names) — its concern is **short total history** (4.0 years, 914 rows), which
   affects walk-forward fold sizes and depth more than feature quality itself.

2. **The pooled return distribution's extreme skew/kurtosis (skew=-1.89, excess
   kurtosis=411.2) is driven disproportionately by a handful of raw-data anomalies,
   not by broad-based fat-tailed behavior.** The two single largest `|log_return|`
   values in the whole dataset are CHCL 2006-09-09 (+330.5% per_change) and
   2006-09-10 (-76.3%) — a spike-then-near-total-reversal within 24 hours that is not
   consistent with a real trading day (NEPSE's daily circuit-breaker limits make a
   genuine 330% single-day move implausible) or with a real corporate action (which
   would permanently rebase the price rather than reverting within a day). Left
   unmodified in the loader output — the "true" values cannot be recovered without
   returning to source, which ADR-0006 already established is not accessible. Flagged
   as a known, unresolved raw-data limitation; will be monitored during Day 3/4 for
   outsized influence on engineered features.

3. **`per_change` is confirmed unreliable as a feature and will not be used as one.**
   Beyond the near-duplicate cleaning already handled in `data_loader.py` (where
   `per_change=0.0` was a marker for a bad/incomplete copy), 6 of the 25 largest-move
   trading days in the entire dataset show `per_change=0.0` sitting alongside a
   computed `log_return` of -30% to -51% (e.g. CBBL 2010-11-14: `per_change=0.0` but
   close fell ~32% from the prior session). `0.0` in this column behaves as an
   unreliable/missing-data sentinel rather than a trustworthy percentage. `features.py`
   will compute returns directly from `close` prices, not from this column.

4. **Ties are correctly counted as "down" in both targets — a deliberate reading of the
   locked problem statement, not an oversight.** 5.77% of rows (1,982/34,331) have
   `close(t+1) == close(t)`. Under `target_1d = close(t+1) > close(t)`, a tie evaluates
   to 0/"down", matching the problem statement's literal wording ("will the price be
   **higher**"). Flat-day rate varies notably by symbol (AHPC 9.03% vs. RIDI 2.08%),
   itself a secondary liquidity signal worth noting alongside finding #1.

5. **Sector-level return/volatility comparisons should be read with the liquidity flags
   from finding #1 in mind, not taken at face value.** Microfinance shows the highest
   mean annualized return (~9.5%) without the highest volatility, and Hydropower shows
   negative mean return with the highest volatility (~56.5%) — but Microfinance
   contains CBBL (the flagged illiquid name) and Hydropower's negative mean is
   plausibly influenced by AHPC's and SKBBL's multi-year normalized-price declines
   (visible in the price-trend chart), which may themselves reflect unadjusted
   bonus/rights actions rather than genuine sector underperformance (see Preprocessing
   Decisions — Split/bonus adjustment, below). Sector comparison is reported as
   descriptive EDA, not as a modeling input at this stage.

## Preprocessing Decisions

- **Missing data handling:** `per_change` (6 raw NaNs, pre-cleaning) is not imputed —
  the column is excluded from modeling entirely given finding #3. No other columns had
  missing values post-`load_raw()`.
- **Duplicate rows:** 22 rows across 11 `(symbol, date)` pairs (AHPC ×2, CHCL ×9), all
  showing `row2.open == row1.close` (split-session pattern). Merged via OHLC
  aggregation. See ADR-0007. Row count: 34,353 → 34,342.
- **Listing-day artifact:** NLG 2013-07-17 had `open=0.0` with `high=low=close=275.0`
  (single-trade day). Set `open=close=275.0`.
- **Outlier handling:** No values removed or corrected. The CHCL 2006-09-09/09-10
  episode (finding #2) is left in the dataset and explicitly flagged as an unresolved
  raw-data anomaly — dropping or correcting it would require inventing a "true" value
  with no source to verify against. Will be re-examined during Day 3's leakage audit
  and Day 4 if it shows disproportionate feature importance.
- **Split/bonus adjustment:** No confirmed stock split or bonus-share event was
  identified against an authoritative source (none accessible per ADR-0006). Some
  symbols (notably AHPC, SKBBL, and early CBBL/CHCL) show step-like or multi-year
  gradual price declines in normalized terms that are consistent with unadjusted
  bonus/rights issuance, but this could not be confirmed within Day 2 scope. Documented
  as a limitation for the model card / defense, not corrected.
