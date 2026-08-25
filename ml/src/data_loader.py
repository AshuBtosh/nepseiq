"""Loads raw NEPSE OHLCV CSVs (ml/data/raw/) into a single tidy long-format DataFrame.

Source: github.com/Aabishkar2/nepse-data (MIT). See docs/adr/0006-data-source-pivot.md
and ml/data/raw/README.md for provenance and known data-quality caveats.
"""
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

COLUMNS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "per_change",
    "traded_quantity",
    "traded_amount",
]

# Rows are "near-duplicates" of each other when everything matches except
# per_change (one copy carries the real value, the other a stray 0.0).
NEAR_DUP_KEY = ["symbol", "date", "open", "high", "low", "close", "traded_quantity", "traded_amount"]


def load_raw() -> pd.DataFrame:
    frames = []
    for csv_path in sorted(RAW_DIR.glob("*.csv")):
        symbol = csv_path.stem
        df = pd.read_csv(csv_path)
        df["symbol"] = symbol
        df = df.rename(columns={"published_date": "date"})
        df = df.drop(columns=["status"])
        frames.append(df[COLUMNS])

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])

    combined = _drop_exact_duplicates(combined)
    combined = _resolve_near_duplicates(combined)
    combined = _merge_ohlc_duplicate_dates(combined)
    combined = _fix_listing_day_open(combined)

    combined = combined.sort_values(["symbol", "date"]).reset_index(drop=True)
    return combined


def _drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(keep="first")
    dropped = before - len(df)
    if dropped:
        print(f"[dedup] dropped {dropped} exact full-row duplicate(s)")
    return df.reset_index(drop=True)


def _resolve_near_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    drop_indices = []
    for _, group in df.groupby(NEAR_DUP_KEY):
        if len(group) < 2:
            continue

        zero_rows = group[group["per_change"] == 0.0]
        nonzero_rows = group[group["per_change"] != 0.0]

        if len(group) == 2 and len(zero_rows) == 1 and len(nonzero_rows) == 1:
            zero_idx = zero_rows.index[0]
            keep_row = nonzero_rows.iloc[0]
            drop_row = zero_rows.iloc[0]
            print(
                f"[near-dup] {keep_row['symbol']} {keep_row['date'].date()}: "
                f"keeping per_change={keep_row['per_change']}, "
                f"dropping per_change={drop_row['per_change']} "
                f"(open={keep_row['open']} high={keep_row['high']} "
                f"low={keep_row['low']} close={keep_row['close']})"
            )
            drop_indices.append(zero_idx)
        else:
            print(
                f"[near-dup] UNHANDLED group of {len(group)} rows for "
                f"{group.iloc[0]['symbol']} {group.iloc[0]['date'].date()} "
                f"(per_change values: {group['per_change'].tolist()}) — left as-is"
            )

    if drop_indices:
        df = df.drop(index=drop_indices)
    return df.reset_index(drop=True)


def _merge_ohlc_duplicate_dates(df: pd.DataFrame) -> pd.DataFrame:
    merged_rows = []
    drop_indices = []
    for (symbol, date), group in df.groupby(["symbol", "date"], sort=False):
        if len(group) < 2:
            continue

        first_row = group.iloc[0]
        last_row = group.iloc[-1]
        open_ = first_row["open"]
        close = last_row["close"]

        merged_rows.append({
            "symbol": symbol,
            "date": date,
            "open": open_,
            "high": group["high"].max(),
            "low": group["low"].min(),
            "close": close,
            "per_change": (close - open_) / open_ * 100,
            "traded_quantity": group["traded_quantity"].sum(),
            "traded_amount": group["traded_amount"].sum(),
        })
        print(f"[ohlc-merge] {symbol} {date.date()}: merged {len(group)} rows into one bar")
        drop_indices.extend(group.index.tolist())

    if drop_indices:
        df = df.drop(index=drop_indices)
        df = pd.concat([df, pd.DataFrame(merged_rows, columns=COLUMNS)], ignore_index=True)
    return df.reset_index(drop=True)


def _fix_listing_day_open(df: pd.DataFrame) -> pd.DataFrame:
    mask = (df["open"] == 0) & (df["high"] == df["low"]) & (df["low"] == df["close"])
    for _, row in df[mask].iterrows():
        print(f"[open-fix] {row['symbol']} {row['date'].date()}: open 0.0 -> {row['close']}")
    df.loc[mask, "open"] = df.loc[mask, "close"]
    return df


def print_report(df: pd.DataFrame) -> None:
    for symbol, group in df.groupby("symbol"):
        print(f"\n{symbol}")
        print(f"  rows: {len(group)}")
        print(f"  date range: {group['date'].min().date()} - {group['date'].max().date()}")
        missing = group.isna().sum()
        missing = missing[missing > 0]
        if missing.empty:
            print("  missing values: none")
        else:
            for col, count in missing.items():
                print(f"  missing values in {col}: {count}")

    print(f"\nTotal rows across all symbols: {len(df)}")


if __name__ == "__main__":
    data = load_raw()
    print_report(data)
