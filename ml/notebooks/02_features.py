"""
02_features.py — Day 3 runner.

Executes the feature pipeline against the FROZEN raw data and prints every
number the Day 3 Definition of Done requires. Nothing here is estimated.

Run from the repo root:

    python ml/notebooks/02_features.py | tee ml/reports/day3_full_output.txt
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ml" / "src"))

from data_loader import load_raw                      # noqa: E402
from features import (                                 # noqa: E402
    build_feature_matrix, build_targets, get_feature_names,
    NON_STATIONARY_FEATURES, MAX_LOOKBACK,
)

PROCESSED = ROOT / "ml" / "data" / "processed"
REPORTS = ROOT / "ml" / "reports"
PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 60)


def rule(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ---------------------------------------------------------------------------
rule("1. LOAD FROZEN RAW DATA")
# ---------------------------------------------------------------------------
raw = load_raw()
print(f"raw rows            : {len(raw):,}")
print(f"symbols             : {raw['symbol'].nunique()}  -> "
      f"{sorted(raw['symbol'].unique())}")
print(f"date range          : {raw['date'].min().date()} .. "
      f"{raw['date'].max().date()}")
print(f"duplicate (sym,date): {int(raw.duplicated(subset=['symbol','date']).sum())}")
print("\nrows per symbol:")
print(raw.groupby("symbol").size().sort_values(ascending=False).to_string())


# ---------------------------------------------------------------------------
rule("2. BUILD FEATURE MATRIX")
# ---------------------------------------------------------------------------
matrix = build_feature_matrix(raw)
feature_cols = get_feature_names(matrix)

print(f"\nfinal matrix shape  : {matrix.shape}")
print(f"feature count       : {len(feature_cols)}  (DoD requires >= 25)")
print(f"  stationary        : {len(feature_cols) - len(NON_STATIONARY_FEATURES)}")
print(f"  non-stationary    : {len(NON_STATIONARY_FEATURES)} (flagged, not dropped)")
print(f"max lookback (bars) : {MAX_LOOKBACK}")

print("\nrows surviving per symbol (warm-up cost is visible here):")
survive = pd.DataFrame({
    "raw": raw.groupby("symbol").size(),
    "final": matrix.groupby("symbol").size(),
})
survive["dropped"] = survive["raw"] - survive["final"]
survive["pct_kept"] = (100 * survive["final"] / survive["raw"]).round(1)
print(survive.sort_values("pct_kept").to_string())

print("\nfull feature list:")
for i, c in enumerate(feature_cols, 1):
    flag = "  [non-stationary]" if c in NON_STATIONARY_FEATURES else ""
    print(f"  {i:>2}. {c}{flag}")


# ---------------------------------------------------------------------------
rule("3. TARGET SPOT-CHECK (DoD: >= 3 rows, manual verification)")
# ---------------------------------------------------------------------------
# Spot-check on the ORIGINAL sorted raw frame, not the filtered matrix, so the
# t+1 / t+5 lookups are done by hand against the actual next trading sessions.
symbols_to_check = sorted(raw["symbol"].unique())[:3]
all_ok = True

for sym in symbols_to_check:
    sub = raw[raw["symbol"] == sym].sort_values("date").reset_index(drop=True)
    tg = build_targets(sub)
    idx = len(sub) // 2  # a row comfortably inside the history
    for i in (idx, idx + 137, idx + 401):
        if i + 5 >= len(sub):
            continue
        c0, c1, c5 = sub.loc[i, "close"], sub.loc[i + 1, "close"], sub.loc[i + 5, "close"]
        d0, d1, d5 = sub.loc[i, "date"].date(), sub.loc[i + 1, "date"].date(), sub.loc[i + 5, "date"].date()
        exp1, exp5 = int(c1 > c0), int(c5 > c0)
        got1, got5 = int(tg.loc[i, "target_1d"]), int(tg.loc[i, "target_5d"])
        ok = (exp1 == got1) and (exp5 == got5)
        all_ok &= ok
        print(f"{sym} row {i:>5} {d0}  close={c0:10.2f}")
        print(f"    t+1 {d1} close={c1:10.2f}  expected={exp1} got={got1}")
        print(f"    t+5 {d5} close={c5:10.2f}  expected={exp5} got={got5}   "
              f"{'OK' if ok else 'MISMATCH'}")

print(f"\nspot-check result: {'ALL MATCH' if all_ok else 'MISMATCH FOUND'}")
assert all_ok, "Target spot-check failed — do not proceed to Day 4."

# Tail rows must be unlabelled, never silently 0.
tail_check = build_targets(raw.sort_values(["symbol", "date"]).reset_index(drop=True))
tails = tail_check.groupby("symbol").tail(1)
print(f"last row per symbol with target_1d NA: "
      f"{int(tails['target_1d'].isna().sum())} / {len(tails)} (expect all)")


# ---------------------------------------------------------------------------
rule("4. LEAKAGE ASSERTIONS")
# ---------------------------------------------------------------------------
checks = []

checks.append(("per_change excluded from features",
               "per_change" not in feature_cols))
checks.append(("no target column inside feature set",
               not any(c.startswith("target_") for c in feature_cols)))
checks.append(("zero NaN in final feature matrix",
               int(matrix[feature_cols].isna().sum().sum()) == 0))
checks.append(("zero inf in final feature matrix",
               int(np.isinf(matrix[feature_cols].to_numpy(dtype=float)).sum()) == 0))
checks.append(("(symbol, date) unique",
               not matrix.duplicated(subset=["symbol", "date"]).any()))
checks.append(("dates monotonic within every symbol",
               bool(matrix.groupby("symbol")["date"].is_monotonic_increasing.all())))

# Truncation test — the decisive one. Rebuild features using only the first
# N rows of a symbol. If any feature peeked into the future, the values would
# differ from the same rows computed with the full history present.
test_sym = sorted(raw["symbol"].unique())[0]
full_sym = raw[raw["symbol"] == test_sym].sort_values("date")
head_n = min(400, len(full_sym) - 50)
m_trunc = build_feature_matrix(full_sym.head(head_n), dropna=False, verbose=False)
m_full = build_feature_matrix(full_sym, dropna=False, verbose=False).head(head_n)
diff = (m_trunc[feature_cols].to_numpy(dtype=float)
        - m_full[feature_cols].to_numpy(dtype=float))
max_abs = float(np.nanmax(np.abs(diff)))
checks.append((f"truncation test on {test_sym} (max abs diff={max_abs:.2e})",
               max_abs < 1e-9))

for name, passed in checks:
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
assert all(p for _, p in checks), "Leakage assertion failed."


# ---------------------------------------------------------------------------
rule("5. CLASS BALANCE ON THE PROCESSED SET")
# ---------------------------------------------------------------------------
# Day 2 recorded the baseline on the raw set. After warm-up rows are dropped
# the balance can shift slightly — Day 4 must compare against THIS number.
for t in ("target_1d", "target_5d"):
    vc = matrix[t].value_counts(normalize=True).sort_index()
    maj = float(vc.max())
    print(f"{t}: down={vc.get(0, 0)*100:.2f}%  up={vc.get(1, 0)*100:.2f}%  "
          f"-> majority-class baseline {maj*100:.2f}%")


# ---------------------------------------------------------------------------
rule("6. SAVE PROCESSED DATASET")
# ---------------------------------------------------------------------------
csv_path = PROCESSED / "features.csv"
matrix.to_csv(csv_path, index=False)
print(f"saved: {csv_path}  ({csv_path.stat().st_size / 1e6:.2f} MB)")

try:
    pq_path = PROCESSED / "features.parquet"
    matrix.to_parquet(pq_path, index=False)
    print(f"saved: {pq_path}  ({pq_path.stat().st_size / 1e6:.2f} MB)")
except Exception as e:
    print(f"parquet skipped ({type(e).__name__}: {e}) — CSV is the source of truth")

meta = {
    "generated_by": "ml/notebooks/02_features.py",
    "raw_rows": int(len(raw)),
    "processed_rows": int(len(matrix)),
    "rows_dropped_warmup_and_tail": int(len(raw) - len(matrix)),
    "n_features": len(feature_cols),
    "n_non_stationary": len(NON_STATIONARY_FEATURES),
    "max_lookback_bars": int(MAX_LOOKBACK),
    "symbols": sorted(matrix["symbol"].unique().tolist()),
    "date_min": str(matrix["date"].min().date()),
    "date_max": str(matrix["date"].max().date()),
    "features": feature_cols,
    "non_stationary_features": NON_STATIONARY_FEATURES,
    "targets": ["target_1d", "target_5d"],
    "excluded_columns": ["per_change (unreliable — Day 2 finding)"],
}
meta_path = PROCESSED / "features_metadata.json"
meta_path.write_text(json.dumps(meta, indent=2))
print(f"saved: {meta_path}")

rule("DAY 3 PIPELINE RUN COMPLETE")
print("Copy this entire output into ml/reports/day3_full_output.txt")
