# ADR-0006: Data Source Pivot to Third-Party Aggregated Repository

**Status:** Accepted
**Date:** 2026-08-25
**Day:** Day 1

## Context
PLAN.md specified acquisition priority: (a) Sasto Share, (b) Mero Share, (c) NEPSE
site scrape, (d) manual CSV. All three of (a)-(c) proved inaccessible within the
acquisition budget: Sasto Share (nepsealpha.com/nepse-data) is a client-side
JS-rendered table with no exposed plain endpoint; Mero Share (meroshare.cdsc.com.np)
is a demat/IPO portal with no historical-OHLCV surface at all; NEPSE's own site is a
JS SPA requiring an undocumented rotating auth token. Per CLAUDE.md Rule 7, no
unverified source or reverse-engineered endpoint was treated as working, and none
was guessed at.

## Decision
Use github.com/Aabishkar2/nepse-data (MIT licensed, per its README) as the OHLCV
source for all 11 basket symbols, in place of sources (a)-(c). This is treated as an
extension of PLAN.md's own fallback (d) — a transparent, auditable, scraper-repo-based
export rather than a manual per-symbol UI export, chosen because its methodology is
fully public (scraper source visible in `src/`), it's updated near-daily, and its
data survived a real quality audit (duplicate check, range check, gap-vs-known-closure
check) rather than being taken on faith.

## Alternatives Considered
| Option | Why rejected |
|---|---|
| Reverse-engineer nepsealpha's AJAX endpoint or NEPSE's auth token | Violates Rule 7 — no unverified/guessed sources |
| Manual per-symbol export via nepsealpha UI | Site did not reliably respond to the filter action even under manual testing; slower with no transparency advantage over the GitHub repo |
| Other scraper repos found in search | Rejected in favor of this one for its explicit license statement, visible scraper code, and near-daily update cadence |

## Consequences
**Positive:** Auditable methodology, MIT licensed, 4-20 years of history across the
basket (RIDI shortest at ~4 years, still clears the 3-year minimum), quality-checked
before use rather than blindly trusted.
**Negative:** Not an official NEPSE feed — if the maintainer stops updating it, no
live refresh path. Acceptable since the dataset is frozen after today regardless.
Known issues (pre-2018 `status` field unusable and dropped from the loader output,
~16% `open`-out-of-range rows in NABIL concentrated in early years, duplicate/near-
duplicate rows resolved in the loader with every touched row printed for audit) are
documented, not silently cleaned away.

## Defense Note
If asked why not official NEPSE data: the official site requires an undocumented
rotating auth token — the kind of access a real outside analyst also can't get
without special arrangement. This repo's collection code is fully open source and
its output survived an independent quality audit (duplicate check, OHLC range check,
gap-vs-known-market-closure check), giving reasonable confidence it reflects true
trading history rather than being taken on faith.
