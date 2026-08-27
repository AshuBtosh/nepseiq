"""
diagnose_extreme_returns.py — one-off diagnostic for the Day 5 backtest crash.

The backtest failed with a negative equity value, which only happens if some
symbol's forward_return_1d came out near -100% on a single "row-to-row" step.
A real single NEPSE session cannot move that much (circuit breakers). The two
most likely explanations, both already known data risks:

  1. Sparse trading (ADR-0006 flagged CBBL, RIDI): if a symbol didn't trade
     for weeks, the "next row" in the sorted series is not the next calendar
     day, so a "1-day" feature/target actually spans a large real gap.
  2. Unadjusted bonus/rights shares (Parking Lot, accepted limitation): a
     corporate action can make the raw close jump or crater without a
     genuine trading loss.

This script does not fix anything. It finds and prints the actual offending
row(s) so the fix is chosen from real data, not guessed.
"""

import glob
import os

import pandas as pd

PROCESSED_DIR = os.path.join("data", "processed")


def load_close():
    candidates = sorted(
        glob.glob(os.path.join(PROCESSED_DIR, "*.parquet"))
        + glob.glob(os.path.join(PROCESSED_DIR, "*.csv"))
    )
    path = candidates[-1]
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, columns=["symbol", "date", "close"])
    else:
        df = pd.read_csv(path, usecols=["symbol", "date", "close"],
                         parse_dates=["date"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def main():
    df = load_close()
    df["prev_date"] = df.groupby("symbol")["date"].shift(1)
    df["gap_days"] = (df["date"] - df["prev_date"]).dt.days
    df["fwd_close"] = df.groupby("symbol")["close"].shift(-1)
    df["fwd_date"] = df.groupby("symbol")["date"].shift(-1)
    df["gap_to_next_days"] = (df["fwd_date"] - df["date"]).dt.days
    df["forward_return_1d"] = df["fwd_close"] / df["close"] - 1.0

    # Only look inside the actual backtest window to match what crashed.
    window = df[(df["date"] >= "2017-05-08") &
                (df["date"] <= "2026-08-17")].copy()

    extreme = window[window["forward_return_1d"] <= -0.30].sort_values(
        "forward_return_1d"
    )

    print(f"Rows in backtest window: {len(window)}")
    print(f"Rows with forward_return_1d <= -30%: {len(extreme)}")
    print()
    if extreme.empty:
        print("No extreme drops found in this window at the -30% threshold. "
              "Lowering threshold to -10% to widen the net:")
        extreme = window[window["forward_return_1d"] <= -0.10].sort_values(
            "forward_return_1d"
        )
        print(f"Rows with forward_return_1d <= -10%: {len(extreme)}")

    cols = ["symbol", "date", "close", "fwd_date", "fwd_close",
            "forward_return_1d", "gap_to_next_days"]
    print(extreme[cols].head(30).to_string(index=False))

    print("\n--- Same diagnostic for extreme GAINS (>= +100%), for completeness ---")
    extreme_up = window[window["forward_return_1d"] >= 1.0].sort_values(
        "forward_return_1d", ascending=False
    )
    print(f"Rows with forward_return_1d >= +100%: {len(extreme_up)}")
    print(extreme_up[cols].head(15).to_string(index=False))

    print("\n--- Trading-gap distribution (days between consecutive rows per symbol) ---")
    print(df.groupby("symbol")["gap_days"].describe())


if __name__ == "__main__":
    main()
