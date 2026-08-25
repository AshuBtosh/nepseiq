# ml/data/raw/ — Frozen NEPSE OHLCV Data

## Source

[github.com/Aabishkar2/nepse-data](https://github.com/Aabishkar2/nepse-data) — a
third-party public scraper repo, ~130 NEPSE companies, updated near-daily via GitHub
Actions. Scraper source visible in the repo's `src/` folder. Licensed **MIT** (declared
in the repo's `README.md`; no standalone `LICENSE` file exists at the repo root, but the
MIT declaration is explicit and unambiguous).

This replaces PLAN.md Day 1's original source priority (Sasto Share export → Mero Share
→ NEPSE site scrape → manual CSV) — see `docs/adr/0006-data-source-pivot.md` for the
decision record. All three original sources were confirmed JS-gated or wrong-purpose
and not reachable without reverse-engineering an undocumented auth mechanism, which was
deliberately avoided.

**Date pulled:** 2026-08-25

Each symbol's file is a direct, unmodified pull of
`data/company-wise/{SYMBOL}.csv` from the repo's `main` branch — no cleaning, no
dedup, no filtering applied at this layer. This is the frozen layer per CLAUDE.md §6:
never edited in place, never regenerated after Day 1.

## Columns (as scraped)

| Column | Meaning |
|---|---|
| `published_date` | Trading date |
| `open` | Opening price |
| `high` | Day's high |
| `low` | Day's low |
| `close` | Closing price |
| `per_change` | % change (reference point not fully documented — see caveats) |
| `traded_quantity` | Shares traded |
| `traded_amount` | Turnover (NPR) |
| `status` | `sign(close - open)` per `src/utils/status.py` in the source repo — **dropped by `data_loader.py`**, see caveats below |

## Per-symbol real row counts and date ranges

(From `ml/src/data_loader.py`'s actual output, post-dedup, 2026-08-25 run.)

| Symbol | Rows | Date range |
|---|---|---|
| AHPC | 3,734 | 2009-11-25 – 2026-08-24 |
| CBBL | 3,229 | 2005-02-09 – 2026-08-24 |
| CHCL | 4,434 | 2006-06-13 – 2026-08-24 |
| EBL | 3,484 | 2011-05-26 – 2026-08-24 |
| GBIME | 2,741 | 2012-09-26 – 2026-08-24 |
| NABIL | 3,485 | 2011-05-15 – 2026-08-24 |
| NLG | 2,924 | 2013-07-17 – 2026-08-24 |
| NLIC | 3,335 | 2011-03-29 – 2026-08-24 |
| RIDI | 914 | 2022-08-15 – 2026-08-24 |
| SICL | 3,194 | 2011-03-25 – 2026-08-24 |
| SKBBL | 2,879 | 2013-10-08 – 2026-08-24 |

**Total rows across all symbols (post-dedup): 34,353**

All 11 symbols exceed the 3-year minimum target; most have 10+ years of history.
RIDI is the shortest at ~4 years.

## Known data-quality caveats (from Day 1 audit, verified against NABIL and confirmed
structurally present across the basket)

- **`status` column dropped in `data_loader.py`.** Traced to
  `src/utils/status.py` in the source repo: `status = sign(close - open)` for that
  trading day. Audited on NABIL and found **every row before 2018-02-18 has
  `status = 0` regardless of actual price movement** — the field was never populated
  for the backfilled pre-2018 history, making it unusable as a feature across the
  full date range. Where it *is* populated (2018 onward), it's redundant with a
  trivial `close - open` sign computation available directly from columns already
  kept. Dropped entirely rather than partially trusted.
- **~16% of rows have `open` outside the `[low, high]` range** (558 of 3,496 checked
  on NABIL). Possibly a known NEPSE-data quirk where older feeds report the previous
  day's close as "open" rather than the actual first trade — **unconfirmed**, not
  found documented anywhere in the source repo. Left unmodified in the raw files.
  **Flagged for Day 2 EDA, not resolved here.**
- **Duplicate rows existed in the raw scrape.** Two patterns found and handled, but
  only in `data_loader.py`'s output — **the raw CSVs in this directory are untouched
  and still contain the duplicates as scraped**:
  - Exact full-row duplicates (identical in every field) — dropped, keep-first.
  - Near-duplicates: same date/OHLC/volume, differing only in `per_change`, where
    one copy has `per_change = 0.0` and the other has the real value — the `0.0`
    copy is dropped, the real value kept. Every row this logic touches is printed
    by the loader for auditability, not applied silently.
- **A small number of rows have `close` outside `[low, high]`** (2 of 3,496 on
  NABIL) — clear scraping errors (e.g. `high`/`low` both pinned to an unrelated
  value). Left unmodified; not yet corrected or dropped.
- **Gaps of 10+ calendar days** appear at several points in NABIL's history (e.g.
  ~51 days in Mar–May 2020, ~32 days in Apr–May 2015) that line up with known real
  NEPSE closures (COVID-19 lockdown, 2015 earthquake) rather than missing-data holes
  — not cross-checked against an authoritative NEPSE trading calendar.

## Stock Basket Selection

11 symbols, picked to cover four sectors (banking, hydropower, insurance,
microfinance) rather than to maximize any single metric. The basket was chosen
before today's data pull; the reasons below distinguish what today's data actually
confirmed from what was a general sector-coverage judgment call going in.

One piece of real evidence computed today, used below: **rows per year of listed
history** (row count ÷ years between each symbol's earliest and latest date). NEPSE
trades roughly 230–250 sessions/year, so a value close to that range means the
symbol has data for close to every available session across its whole listed life —
a rough trading-frequency proxy from the row data itself, **not** a computed
turnover/volume ranking (no aggregation of `traded_amount` or `traded_quantity`
across history was done today).

**Banking**
- **NABIL** — 3,496 rows over 15.3 years, ~229 rows/year (near-full session
  coverage). One of Nepal's largest, longest-listed commercial banks; general
  sector-coverage pick, corroborated today by the density figure.
- **EBL** — 3,496 rows over 15.2 years, ~229 rows/year (near-full session
  coverage). Same reasoning as NABIL: established commercial bank, density
  figure today supports consistent trading.
- **GBIME** — 2,752 rows over 13.9 years, ~198 rows/year — the lowest density of
  the three banking picks. Sector-coverage pick (large commercial bank); today's
  data shows somewhat less consistent daily coverage than NABIL/EBL, not
  confirmed as highly liquid.

**Hydropower**
- **CHCL** — 4,446 rows over 20.2 years, ~220 rows/year (near-full session
  coverage), the longest history in the whole basket. Sector-coverage pick,
  density figure today supports it as an actively-traded, long-listed name.
- **AHPC** — 3,746 rows over 16.7 years, ~224 rows/year (near-full session
  coverage). Same reasoning: long-listed, density figure today supports
  consistent trading.
- **RIDI** — 914 rows over **only 4.0 years** (2022-08-15 onward), ~227 rows/year.
  This is a general sector-coverage pick, not a liquidity pick — it's the
  shortest history in the entire basket, barely clearing PLAN.md's 3-year
  minimum. Session density is comparable to the other two hydropower names once
  listed, but there are only 4 years of it. Flagged here explicitly, not just in
  the caveats above.

**Insurance**
- **NLIC** — 3,346 rows over 15.4 years, ~217 rows/year. Sector-coverage pick
  (established life insurer); density figure today supports regular trading,
  somewhat below the ~230 seen in the strongest names.
- **SICL** — 3,194 rows over 15.4 years, ~207 rows/year — the second-lowest
  density in the basket after CBBL. Sector-coverage pick; today's data shows
  less consistent daily coverage than most of the basket, not confirmed as
  highly liquid.
- **NLG** — 2,924 rows over 13.1 years, ~223 rows/year (near-full session
  coverage). Sector-coverage pick; density figure today supports consistent
  trading.

**Microfinance**
- **CBBL** — 3,230 rows over **21.5 years**, but only **~150 rows/year** — the
  lowest density in the entire basket by a wide margin, despite having the
  longest listed history. This is a sector-coverage pick (need a microfinance
  representative); today's data does not support treating it as a liquid,
  actively-traded name — it's the one basket member where the density figure
  actively argues against a liquidity claim, not just fails to confirm one.
- **SKBBL** — 2,879 rows over 12.9 years, ~224 rows/year (near-full session
  coverage). Sector-coverage pick; density figure today supports consistent
  trading.

No NPR turnover figures or liquidity rankings beyond the session-density figures
above were computed today, and none are claimed here.
