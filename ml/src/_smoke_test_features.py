"""
Synthetic smoke test for features.py.

This validates the CODE ONLY. Every number it prints comes from randomly
generated data, not from the NepseIQ dataset. It exists so the pipeline is
known-good before it touches the frozen raw data.
"""
import numpy as np
import pandas as pd
from features import (
    build_feature_matrix, build_targets, get_feature_names,
    NON_STATIONARY_FEATURES,
)

rng = np.random.default_rng(42)


def make_fake(symbol: str, n: int, start_price: float) -> pd.DataFrame:
    # Business days with some deliberately removed, to mimic NEPSE holidays.
    dates = pd.bdate_range("2015-01-01", periods=int(n * 1.6))
    dates = dates[rng.random(len(dates)) > 0.13][:n]
    assert len(dates) == n, f"date generation short: {len(dates)} != {n}"
    ret = rng.normal(0, 0.02, n)
    close = start_price * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    qty = rng.integers(100, 50000, n).astype(float)
    return pd.DataFrame({
        "symbol": symbol, "date": dates, "open": open_, "high": high,
        "low": low, "close": close, "traded_quantity": qty,
        "traded_amount": qty * close * (1 + rng.normal(0, 0.002, n)),
        "per_change": rng.normal(0, 1, n),  # present but must be ignored
    })


raw = pd.concat(
    [make_fake("FAKE_A", 900, 500.0),
     make_fake("FAKE_B", 700, 120.0),
     make_fake("FAKE_C", 400, 1800.0)],
    ignore_index=True,
)

print("=" * 68)
print("SYNTHETIC SMOKE TEST — numbers below are from RANDOM data")
print("=" * 68)
print(f"synthetic input rows: {len(raw):,}  symbols: {raw['symbol'].nunique()}")
print()

matrix = build_feature_matrix(raw)
feature_cols = get_feature_names(matrix)

print()
print(f"FEATURE COUNT: {len(feature_cols)}  (DoD requires >= 25)")
print(f"  stationary    : {len(feature_cols) - len(NON_STATIONARY_FEATURES)}")
print(f"  non-stationary: {len(NON_STATIONARY_FEATURES)} (flagged, not dropped)")
print()
for i, c in enumerate(feature_cols, 1):
    flag = "  [non-stationary]" if c in NON_STATIONARY_FEATURES else ""
    print(f"  {i:>2}. {c}{flag}")

# --- Leakage assertions --------------------------------------------------
print()
print("-" * 68)
print("LEAKAGE ASSERTIONS")
print("-" * 68)

assert "per_change" not in feature_cols, "per_change leaked into features"
print("  [PASS] per_change excluded from feature set")

assert not any(c.startswith("target_") for c in feature_cols)
print("  [PASS] no target column present in feature set")

assert matrix[feature_cols].isna().sum().sum() == 0
print("  [PASS] zero NaN in final feature matrix")

assert not matrix.duplicated(subset=["symbol", "date"]).any()
print("  [PASS] (symbol, date) is unique")

# Truncation test: features for the first N rows must be IDENTICAL whether or
# not the future rows exist. If any feature peeked forward, they would differ.
head_n = 300
truncated = raw[raw["symbol"] == "FAKE_A"].sort_values("date").head(head_n)
full = raw[raw["symbol"] == "FAKE_A"].sort_values("date")

m_trunc = build_feature_matrix(truncated, dropna=False, verbose=False)
m_full = build_feature_matrix(full, dropna=False, verbose=False).head(head_n)

cmp_cols = [c for c in feature_cols]
diff = (m_trunc[cmp_cols].to_numpy(dtype=float)
        - m_full[cmp_cols].to_numpy(dtype=float))
max_abs = np.nanmax(np.abs(diff))
assert max_abs < 1e-9, f"FUTURE PEEK DETECTED: max abs diff {max_abs}"
print(f"  [PASS] truncation test: max abs diff = {max_abs:.2e}")
print("         (features on first 300 rows identical with and without")
print("          the remaining future rows present -> no look-ahead)")

# --- Target spot-check ---------------------------------------------------
print()
print("-" * 68)
print("TARGET SPOT-CHECK (synthetic)")
print("-" * 68)

sub = raw[raw["symbol"] == "FAKE_A"].sort_values("date").reset_index(drop=True)
tg = build_targets(sub)
ok = True
for i in [50, 200, 455]:
    c_now = sub.loc[i, "close"]
    c_1 = sub.loc[i + 1, "close"]
    c_5 = sub.loc[i + 5, "close"]
    exp1, exp5 = int(c_1 > c_now), int(c_5 > c_now)
    got1, got5 = int(tg.loc[i, "target_1d"]), int(tg.loc[i, "target_5d"])
    ok &= (exp1 == got1) and (exp5 == got5)
    print(f"  row {i:>3}: close={c_now:8.2f} | t+1={c_1:8.2f} exp={exp1} got={got1}"
          f" | t+5={c_5:8.2f} exp={exp5} got={got5}")
assert ok, "target mismatch"
print("  [PASS] all 3 spot-checked rows match hand computation")

tail = tg[tg["symbol"] == "FAKE_A"].tail(5)
assert tail["target_5d"].isna().all(), "tail rows should be unlabelled"
print("  [PASS] last 5 rows have target_5d = NA (no future available)")

print()
print("=" * 68)
print("ALL ASSERTIONS PASSED — pipeline is code-correct.")
print("Row counts above are SYNTHETIC. Real numbers require the frozen data.")
print("=" * 68)
