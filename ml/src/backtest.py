"""
backtest.py — Day 5 backtest for NepseIQ.

SCOPE (binding, per ADR-0009)
------------------------------
Only target_1d, only RandomForest, is backtested here. target_5d did not beat
its baseline in Day 4 walk-forward validation and is not a served model per
ADR-0009 -- it is not eligible for a backtest that implies it's tradeable.
This script hard-asserts that scope; it does not take a --target flag for 5d.

DATA SOURCE
-----------
Predictions come from reports/day5_oof_predictions.csv, produced by the Day 5
patch to train.py. These are genuine out-of-fold predictions -- each prediction
was made by a model that never saw that row during training. The backtest
window is therefore the union of all five test folds:
2017-05-08 -> 2026-08-17 (see Day 5 train re-run output). Dates before the
first test fold have no OOF prediction and are correctly excluded -- there is
no such thing as an out-of-sample backtest on in-sample rows.

FORWARD RETURN -- COMPUTED, NOT ASSUMED
----------------------------------------
target_1d's exact construction lives in features.py, which is not in scope for
this file. Rather than assume its definition, this script computes the forward
1-day return directly from the processed close-price series
(close.shift(-1)/close - 1, per symbol) and cross-checks its sign against the
OOF file's y_true. If the two disagree on sign at more than a small tolerance,
that means this script's return convention doesn't match the label's
convention, and the equity curve would be wrong -- so it stops and prints a
warning instead of silently proceeding. Expect near-100% agreement, not
necessarily exactly 100% (edge cases: exact-zero moves, rounding).

STRATEGY
--------
Daily rebalance. On each date in the backtest window, go long, equal-weight,
every symbol whose predicted P(up) exceeds the threshold; hold cash for the
rest of the day's capital if no symbol qualifies. Because target_1d's horizon
is exactly one session, every opened position is necessarily closed the same
cycle -- there is no "hold longer" option consistent with what the model
actually predicts. Transaction costs (ml/src/costs.py) are charged on both the
entry and the exit of every position, every day it's held.

BENCHMARK
---------
Equal-weight buy across all 11 symbols on the first backtest date, held
untouched to the last date. One buy leg per symbol at entry, one sell leg per
symbol at exit. No rebalancing, no daily costs.

WHAT THIS DOES NOT MODEL (stated, not hidden)
----------------------------------------------
- Capital gains tax (PLAN.md Day 5 scopes costs to commission + SEBON + DP only;
  see costs.py docstring for why CGT is excluded).
- Slippage / liquidity impact / ability to actually fill at the closing price.
- Position limits, circuit breakers, or NEPSE's daily price-band rules.
- Whether DP charge applies to both legs or the sell only (costs.py applies
  both, a deliberately conservative assumption -- see costs.py docstring).
These are limitations to state in model_card.md, not gaps to quietly patch.
"""

from __future__ import annotations
import costs
import matplotlib.pyplot as plt

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")


REPORTS_DIR = "reports"
OOF_PATH = os.path.join(REPORTS_DIR, "day5_oof_predictions.csv")
PROCESSED_DIR = os.path.join("data", "processed")

FROZEN_TARGET = "target_1d"
FROZEN_MODEL = "RandomForest"   # per ADR-0009 / ADR-0004 tie-break

DEFAULT_CAPITAL = 1_000_000.0
DEFAULT_THRESHOLDS = [0.50, 0.55, 0.60]

SIGN_MISMATCH_TOLERANCE = 0.02  # abort if >2% of rows disagree on direction

# Known data anomalies excluded from the STRATEGY simulation only (never from
# the underlying price series or the buy-and-hold benchmark). Found via
# diagnose_extreme_returns.py: NLIC dropped -30.6% close-to-close on two
# genuinely consecutive sessions (gap_to_next_days == 1), which exceeds any
# real NEPSE circuit breaker. This is the same class of thing already
# documented as an accepted limitation (PROGRESS.md Parking Lot:
# "Unadjusted bonus/rights share detection -- no authoritative NEPSE
# corporate-actions source available") -- treated the same way ADR-0007's
# investigation treated the CHCL 2006 anomaly: flagged and excluded from the
# specific downstream use where it would otherwise distort the result, not
# silently corrected in the underlying data. One row, out of 21,728 OOF rows
# in the backtest window.
KNOWN_DATA_ANOMALIES = [
    ("NLIC", "2018-03-27"),
]

# Defensive floor, not a substitute for the exclusion above. This exists so
# an UNKNOWN future anomaly of the same kind fails loudly with a printed flag
# instead of crashing the whole run on a negative-equity ValueError. Any time
# this actually triggers, treat it as a signal to run the diagnostic again --
# it means there's another anomaly that hasn't been identified and excluded.
MAX_DAILY_PORTFOLIO_LOSS = -0.50

# Once equity falls below this fraction of starting capital, stop trading
# entirely and hold whatever remains in cash for the rest of the window.
# Fixed per-trade costs (the Rs 25 DP charge, see costs.py) don't scale down
# with position size. Once a losing streak has shrunk the account far enough,
# that flat fee stops being a small percentage and becomes many multiples of
# the remaining capital -- every subsequent "trade" is a fraction-of-a-paisa
# position that no real broker or exchange would execute. Continuing to
# compound through that is not a stress test, it's a floating-point artifact.
# A real account would be effectively wiped out and done trading; this floor
# makes the simulation stop the same way reality would, rather than
# compounding through negative-equity fiction. Reported explicitly as
# "went bankrupt on <date>" wherever it triggers -- this is itself a finding
# about the threshold, not something to hide.
BANKRUPTCY_FRACTION = 0.01


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_processed_close() -> pd.DataFrame:
    """Load symbol/date/close from the same processed file train.py used."""
    import glob
    candidates = sorted(
        glob.glob(os.path.join(PROCESSED_DIR, "*.parquet"))
        + glob.glob(os.path.join(PROCESSED_DIR, "*.csv"))
    )
    if not candidates:
        raise FileNotFoundError(f"No processed file found in {PROCESSED_DIR}/")
    path = candidates[-1]
    print(f"[load] close prices from {path}")
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, columns=["symbol", "date", "close"])
    else:
        df = pd.read_csv(path, usecols=["symbol", "date", "close"],
                         parse_dates=["date"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def compute_forward_returns(close_df: pd.DataFrame) -> pd.DataFrame:
    """
    forward_return_1d at (symbol, date) = return earned by holding from this
    date's close to the NEXT date's close, for that symbol. Requires the full
    close series (not just OOF rows) so shift(-1) is correct at fold boundaries.
    """
    df = close_df.copy()
    df["forward_return_1d"] = (
        df.groupby("symbol")["close"].shift(-1) / df["close"] - 1.0
    )
    return df


def load_oof_predictions() -> pd.DataFrame:
    if not os.path.exists(OOF_PATH):
        raise FileNotFoundError(
            f"{OOF_PATH} not found. Re-run the Day 5 patched train.py first."
        )
    df = pd.read_csv(OOF_PATH, parse_dates=["date"])
    df = df[(df["target"] == FROZEN_TARGET) & (
        df["model"] == FROZEN_MODEL)].copy()
    if df.empty:
        raise ValueError(
            f"No OOF rows found for target={FROZEN_TARGET}, model={FROZEN_MODEL}. "
            f"Check reports/day5_oof_predictions.csv contents."
        )
    print(f"[load] {len(df)} OOF rows for {FROZEN_MODEL} / {FROZEN_TARGET}")
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def sanity_check_direction(oof: pd.DataFrame, close_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge OOF predictions with computed forward returns and verify the sign of
    forward_return_1d agrees with y_true. This is the check that catches a
    return-convention mismatch before it silently corrupts the equity curve.
    """
    fwd = compute_forward_returns(close_df)
    merged = oof.merge(fwd[["symbol", "date", "forward_return_1d"]],
                       on=["symbol", "date"], how="left")

    missing = merged["forward_return_1d"].isna().sum()
    if missing:
        print(f"[warn] {missing} OOF rows have no forward return "
              f"(likely the last trading date per symbol) -- dropping them.")
    merged = merged.dropna(subset=["forward_return_1d"]).copy()

    computed_up = (merged["forward_return_1d"] > 0).astype(int)
    mismatch = (computed_up != merged["y_true"]).mean()
    print(f"[sanity] sign(forward_return_1d) vs y_true mismatch rate: "
          f"{mismatch:.4%}")

    if mismatch > SIGN_MISMATCH_TOLERANCE:
        raise RuntimeError(
            f"Mismatch rate {mismatch:.4%} exceeds tolerance "
            f"{SIGN_MISMATCH_TOLERANCE:.2%}. This means the forward return "
            f"computed here does not match target_1d's actual definition in "
            f"features.py. DO NOT trust the equity curve below until this is "
            f"resolved -- check features.py's target construction."
        )
    return merged


def exclude_known_anomalies(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Remove KNOWN_DATA_ANOMALIES rows from the strategy-eligible prediction
    set. Never applied to close_df / the buy-and-hold benchmark -- the
    underlying price history is left exactly as-is; only the decision to
    treat that single day as a tradeable signal is excluded.
    """
    out = merged.copy()
    out["_date_str"] = out["date"].dt.strftime("%Y-%m-%d")
    before = len(out)
    for symbol, date_str in KNOWN_DATA_ANOMALIES:
        hit = (out["symbol"] == symbol) & (out["_date_str"] == date_str)
        n_hit = int(hit.sum())
        if n_hit:
            print(f"[exclude] {symbol} {date_str}: removing {n_hit} row(s) "
                  f"from strategy simulation -- known unadjusted-price "
                  f"anomaly, see KNOWN_DATA_ANOMALIES docstring")
        out = out[~hit]
    out = out.drop(columns=["_date_str"])
    print(f"[exclude] strategy-eligible rows: {before} -> {len(out)}")
    return out


# --------------------------------------------------------------------------
# Strategy simulation
# --------------------------------------------------------------------------

def simulate_strategy(merged: pd.DataFrame, threshold: float,
                      capital: float) -> pd.DataFrame:
    """
    Daily equal-weight long-only strategy, cash when no symbol qualifies.
    Every position is opened and closed within the same day-cycle (horizon=1),
    so round-trip transaction costs are charged every day a symbol is held.

    Returns a per-day DataFrame: date, gross_return, cost_drag, net_return,
    n_positions, equity (starting at `capital`).
    """
    daily_rows = []
    equity = capital
    bankrupt = False
    bankruptcy_floor = capital * BANKRUPTCY_FRACTION

    for date, day_df in merged.groupby("date"):
        if bankrupt:
            daily_rows.append({
                "date": date, "n_positions": 0,
                "gross_return": 0.0, "cost_drag": 0.0, "net_return": 0.0,
                "equity": equity,
            })
            continue

        picks = day_df[day_df["proba"] > threshold]
        n = len(picks)

        if n == 0:
            daily_rows.append({
                "date": date, "n_positions": 0,
                "gross_return": 0.0, "cost_drag": 0.0, "net_return": 0.0,
                "equity": equity,
            })
            continue

        position_size = equity / n
        gross_return = picks["forward_return_1d"].mean()  # equal-weight

        # Entry cost on the pre-trade allocation; exit cost on what that
        # position is actually worth after the day's move -- not the same
        # figure for both legs, since a losing position exits at a smaller
        # notional than it entered at.
        total_cost = 0.0
        for _, row in picks.iterrows():
            exit_value = max(
                position_size * (1.0 + row["forward_return_1d"]), 0.0)
            total_cost += costs.leg_cost(position_size, "buy")["total"]
            total_cost += costs.leg_cost(exit_value, "sell")["total"]
        cost_drag = total_cost / equity if equity > 0 else 0.0

        net_return = gross_return - cost_drag

        # Defensive floor -- see MAX_DAILY_PORTFOLIO_LOSS docstring. Should
        # never actually trigger given KNOWN_DATA_ANOMALIES exclusion; if it
        # does, that's a signal there's another unidentified anomaly, not a
        # normal outcome to accept quietly.
        if net_return < MAX_DAILY_PORTFOLIO_LOSS:
            print(f"[floor-triggered] {date.date()}: net_return {net_return:.4f} "
                  f"floored to {MAX_DAILY_PORTFOLIO_LOSS:.4f} -- "
                  f"symbols today: {sorted(picks['symbol'].tolist())} -- "
                  f"RUN diagnose_extreme_returns.py, this is not expected.")
            net_return = MAX_DAILY_PORTFOLIO_LOSS

        equity = equity * (1.0 + net_return)

        if equity <= bankruptcy_floor:
            print(f"[bankrupt] {date.date()}: equity {equity:,.2f} fell below "
                  f"{BANKRUPTCY_FRACTION:.0%} of starting capital "
                  f"({bankruptcy_floor:,.2f}) -- halting further trading for "
                  f"the remainder of the backtest at threshold={threshold}.")
            bankrupt = True

        daily_rows.append({
            "date": date, "n_positions": n,
            "gross_return": gross_return, "cost_drag": cost_drag,
            "net_return": net_return, "equity": equity,
        })

    return pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)


def simulate_buy_and_hold(close_df: pd.DataFrame, start_date, end_date,
                          capital: float) -> pd.DataFrame:
    """
    Equal-weight buy across all symbols present at start_date, held to
    end_date. One buy leg per symbol at entry, one sell leg per symbol at exit.
    No rebalancing in between -- pure daily mark-to-market of the equity curve.
    """
    window = close_df[(close_df["date"] >= start_date)
                      & (close_df["date"] <= end_date)].copy()
    symbols = sorted(window["symbol"].unique())
    n = len(symbols)
    per_symbol_capital = capital / n

    entry_prices, shares, entry_cost_total = {}, {}, 0.0
    for sym in symbols:
        sym_df = window[window["symbol"] == sym].sort_values("date")
        if sym_df.empty:
            continue
        first_row = sym_df.iloc[0]
        entry_prices[sym] = first_row["close"]
        shares[sym] = per_symbol_capital / first_row["close"]
        entry_cost_total += costs.leg_cost(per_symbol_capital, "buy")["total"]

    dates = sorted(window["date"].unique())
    daily_rows = []
    equity_after_entry_cost = capital - entry_cost_total

    for d in dates:
        day_prices = window[window["date"] == d].set_index("symbol")["close"]
        mtm = sum(shares.get(sym, 0.0) * day_prices.get(sym, entry_prices.get(sym, 0.0))
                  for sym in symbols)
        daily_rows.append({"date": d, "equity": mtm})

    bh = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    # Rescale so day-0 equity reflects entry costs, then apply exit costs to
    # the final value.
    scale = equity_after_entry_cost / bh["equity"].iloc[0]
    bh["equity"] = bh["equity"] * scale

    exit_value = bh["equity"].iloc[-1]
    exit_cost_total = sum(
        costs.leg_cost(exit_value / n, "sell")["total"] for _ in symbols
    )
    bh.loc[bh.index[-1], "equity"] = exit_value - exit_cost_total

    print(f"[buy-hold] entry cost total: {entry_cost_total:,.2f}  "
          f"exit cost total: {exit_cost_total:,.2f}")
    return bh


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def compute_metrics(equity_curve: pd.DataFrame, capital: float,
                    is_strategy: bool) -> dict:
    eq = equity_curve["equity"].values
    total_return = (eq[-1] / capital) - 1.0

    running_max = np.maximum.accumulate(eq)
    drawdown = (eq - running_max) / running_max
    max_drawdown = drawdown.min()

    metrics = {
        "total_return_pct": total_return * 100.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "final_equity": eq[-1],
    }

    if is_strategy:
        traded_days = equity_curve[equity_curve["n_positions"] > 0]
        metrics["n_trading_days"] = len(traded_days)
        metrics["n_total_days"] = len(equity_curve)
        metrics["win_rate_pct"] = (
            (traded_days["net_return"] > 0).mean() * 100.0
            if len(traded_days) else float("nan")
        )
        metrics["avg_positions_per_day"] = (
            traded_days["n_positions"].mean() if len(traded_days) else 0.0
        )
        metrics["total_gross_return_pct"] = (
            (1 + equity_curve["gross_return"]).prod() - 1
        ) * 100.0
        metrics["cumulative_cost_drag_pct"] = (
            metrics["total_gross_return_pct"] - metrics["total_return_pct"]
        )

    return metrics


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=DEFAULT_THRESHOLDS)
    args = ap.parse_args()

    os.makedirs(REPORTS_DIR, exist_ok=True)

    print(f"[scope] target={FROZEN_TARGET}  model={FROZEN_MODEL}  "
          f"(per ADR-0009 -- target_5d is not backtested)")

    close_df = load_processed_close()
    oof = load_oof_predictions()
    merged = sanity_check_direction(oof, close_df)
    merged = exclude_known_anomalies(merged)

    start_date, end_date = merged["date"].min(), merged["date"].max()
    print(f"[window] backtest period: {start_date.date()} -> {end_date.date()} "
          f"({merged['date'].nunique()} trading days)")

    bh_curve = simulate_buy_and_hold(
        close_df, start_date, end_date, args.capital)
    bh_metrics = compute_metrics(bh_curve, args.capital, is_strategy=False)

    print("\n=== BUY-AND-HOLD BENCHMARK ===")
    for k, v in bh_metrics.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v}")

    sensitivity_rows = []
    strategy_curves = {}

    for thr in args.thresholds:
        curve = simulate_strategy(merged, thr, args.capital)
        strategy_curves[thr] = curve
        m = compute_metrics(curve, args.capital, is_strategy=True)
        m["threshold"] = thr
        sensitivity_rows.append(m)

        print(f"\n=== STRATEGY threshold={thr} ===")
        for k, v in m.items():
            print(f"  {k}: {v:,.4f}" if isinstance(
                v, float) else f"  {k}: {v}")

    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(f"{REPORTS_DIR}/day5_threshold_sensitivity.csv",
                          index=False)

    for thr, curve in strategy_curves.items():
        curve.to_csv(f"{REPORTS_DIR}/day5_equity_curve_thr{thr:.2f}.csv",
                     index=False)
    bh_curve.to_csv(
        f"{REPORTS_DIR}/day5_equity_curve_buyhold.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bh_curve["date"], bh_curve["equity"], label="Buy & Hold",
            linewidth=2, color="black")
    for thr, curve in strategy_curves.items():
        ax.plot(curve["date"], curve["equity"], label=f"Strategy (thr={thr})")
    ax.set_title(f"NepseIQ Backtest — {FROZEN_MODEL} / {FROZEN_TARGET}\n"
                 f"({start_date.date()} to {end_date.date()}, "
                 f"costs included)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (NPR)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{REPORTS_DIR}/day5_equity_curve.png", dpi=130)
    plt.close(fig)

    print(f"\n[export] {REPORTS_DIR}/day5_threshold_sensitivity.csv")
    print(f"[export] {REPORTS_DIR}/day5_equity_curve_*.csv")
    print(f"[export] {REPORTS_DIR}/day5_equity_curve.png")
    print("\n[done]")


if __name__ == "__main__":
    main()
