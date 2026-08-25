"""
Day 2 - Task 1: Data inspection.
Run this FIRST, before the full EDA notebook is written, so the notebook
is built against the *actual* schema data_loader.py produces, not an
assumed one.

Run from repo root:
    python ml/notebooks/01_data_inspect.py > ml/reports/day2_inspect_output.txt
(or just run in a notebook cell / venv and paste the printed output back)
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import load_raw  # noqa: E402  -- adjust import if your function name differs

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = load_raw()  # adjust call signature if data_loader.py expects args (e.g. a raw dir path)

print("=" * 80)
print("SHAPE")
print("=" * 80)
print(df.shape)

print("\n" + "=" * 80)
print("DTYPES")
print("=" * 80)
print(df.dtypes)

print("\n" + "=" * 80)
print("HEAD (10 rows)")
print("=" * 80)
print(df.head(10))

print("\n" + "=" * 80)
print("COLUMN NAMES (exact, for copy-paste)")
print("=" * 80)
print(list(df.columns))

print("\n" + "=" * 80)
print("PER-SYMBOL ROW COUNTS")
print("=" * 80)
symbol_col = "symbol" if "symbol" in df.columns else None
if symbol_col:
    print(df[symbol_col].value_counts())
else:
    print("!! No 'symbol' column found - update this script with the actual column name.")

print("\n" + "=" * 80)
print("PER-SYMBOL DATE RANGE")
print("=" * 80)
date_col = "date" if "date" in df.columns else None
if symbol_col and date_col:
    print(df.groupby(symbol_col)[date_col].agg(["min", "max", "count"]))
else:
    print("!! Update column names above - could not find 'date' and/or 'symbol'.")

print("\n" + "=" * 80)
print("MISSING VALUES PER COLUMN")
print("=" * 80)
print(df.isna().sum())

print("\n" + "=" * 80)
print("DUPLICATE (symbol, date) ROWS")
print("=" * 80)
if symbol_col and date_col:
    dupes = df.duplicated(subset=[symbol_col, date_col], keep=False)
    print(f"Duplicate rows: {dupes.sum()}")
    if dupes.sum() > 0:
        print(df[dupes].sort_values([symbol_col, date_col]).head(20))

print("\n" + "=" * 80)
print("BASIC PRICE SANITY (describe on numeric columns)")
print("=" * 80)
print(df.describe())

print("\n" + "=" * 80)
print("ZERO / NEGATIVE VOLUME OR PRICE CHECK")
print("=" * 80)
for col in ["open", "high", "low", "close", "volume"]:
    if col in df.columns:
        n_zero = (df[col] <= 0).sum()
        print(f"{col}: {n_zero} rows with value <= 0")
