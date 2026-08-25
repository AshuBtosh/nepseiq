# ADR-0007: Duplicate (symbol, date) Row Resolution via OHLC Aggregation

**Status:** Accepted
**Date:** 2026-08-25
**Day:** Day 2

## Confirmed Result
Implemented in `ml/src/data_loader.py` as `_merge_ohlc_duplicate_dates()`, applied
after existing dedup steps. Verified via re-run of `01_data_inspect.py`:
- 11 `[ohlc-merge]` events logged (AHPC ×2, CHCL ×9) — 22 rows → 11 bars
- Row count: 34,353 → 34,342
- Post-fix duplicate `(symbol, date)` check: 0
- A second, unrelated single-row fix was applied in the same pass —
  `_fix_listing_day_open()` — for NLG's listing day (2013-07-17), where
  `open=0.0` but `high=low=close=275.0` (single-trade day, source stored 0
  instead of the trade price). Set `open = close = 275.0`. Not ADR-worthy on
  its own (unambiguous single-trade artifact) but recorded here since it
  landed in the same commit. Post-fix `open <= 0` check: 0.

## Context
After `load_raw()`'s internal dedup (24 exact duplicates dropped, 45 near-duplicates
resolved by preferring non-zero `per_change`), 22 rows remained sharing a
`(symbol, date)` key — 11 pairs across AHPC (2 pairs) and CHCL (9 pairs), concentrated
in 2006–2009.

Inspection shows all 11 pairs share the same structure: the second row's `open`
exactly equals the first row's `close` (e.g. AHPC 2010-07-27: row 1 close=479.0,
row 2 open=479.0; same pattern holds for all 9 CHCL pairs). This is not random
duplication — it indicates two genuinely sequential trading intervals that the
upstream source (Aabishkar2/nepse-data, per ADR-0006) mislabeled under one shared
date, most likely a boundary artifact in how that repository's source scraping/export
handled certain historical sessions.

This must be resolved before Day 3 target construction — `target_1d`/`target_5d` for
the row immediately following an affected date would otherwise be computed against
an ambiguous `close` value.

## Decision
Merge each duplicate pair into a single daily bar using standard OHLC aggregation,
ordered by original row position (row 1 = earlier, row 2 = later, per the
open≈prior-close chaining):
- `open` = first row's `open`
- `close` = second row's `close`
- `high` = max(high1, high2)
- `low` = min(low1, low2)
- `traded_quantity` = sum
- `traded_amount` = sum
- `per_change` = recomputed from the merged open/close (not summed/averaged)

Net effect: 34,353 → 34,342 rows. Affects only AHPC and CHCL, 11 dates total.

## Alternatives Considered
| Option | Why rejected |
|---|---|
| Keep first row, drop second | Discards real trading activity (volume, the second session's range) for no reason — the chaining shows both rows are genuine |
| Keep last row, drop first | Same problem, opposite direction |
| Keep row with higher volume | Arbitrary; doesn't reflect that both intervals actually traded |
| Leave both rows as-is | Breaks the `(symbol, date)` uniqueness assumption every downstream feature/target function relies on; ambiguous `close` for target labeling |

## Consequences
**Positive:** Reconstructs one honest, complete daily bar per date. Preserves all real
volume/range information instead of discarding half of it. Downstream feature and
target code can assume `(symbol, date)` is a unique key, as originally intended.

**Negative:** Introduces an assumption (that the two rows are truly sequential same-day
sessions, not a genuine data error unrelated to time-splitting) — flagged as a stated
limitation, not hidden. Affects an immaterial 0.03% of rows, so no meaningful effect
on statistical validity either way.

## Defense Note
If asked "how did you handle duplicate records?": found 22 duplicate `(symbol, date)`
rows post-loader-dedup, all following a consistent open-equals-prior-close chain
indicating split sessions rather than random duplication — merged via standard OHLC
aggregation rather than arbitrarily dropping one side, preserving the real trading
data. Documented as ADR-0007, not silently patched in a notebook cell.
