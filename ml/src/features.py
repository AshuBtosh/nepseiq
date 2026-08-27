"""
features.py — NepseIQ feature engineering pipeline (Day 3).

Turns the tidy long-format output of `data_loader.load_raw()` into a
model-ready feature matrix with the two binary targets from ADR-0003.

DESIGN CONTRACT (this is defense material — read before changing anything):

  1. LEAK-FREE BY CONSTRUCTION. Every feature at row t is computed from
     data at times <= t only. Every rolling/ewm call in this file uses
     pandas' default trailing window (right-closed, ending at t). There is
     not a single `.shift(-n)` anywhere outside the target block, and the
     target block is fenced off in `build_targets()` so it is trivially
     auditable. See docs/defense/leakage_audit.md.

  2. PER-SYMBOL ISOLATION. Every rolling, ewm, cumulative and lag operation
     is done inside a `groupby("symbol")`. NABIL's 50-day SMA never sees an
     EBL price. Without this, pooling 11 symbols into one frame would smear
     the end of one symbol's history into the start of the next.

  3. NO per_change. Day 2 established that the raw `per_change` column from
     the upstream source (ADR-0006) is unreliable. Every return-based
     feature here is derived from `close` directly. `per_change` is dropped.

  4. NO FITTED TRANSFORMS. No scaling, no imputation-by-mean, no encoding
     that learns statistics from the data. Anything that must be *fit* is
     deferred to the Day 4 sklearn Pipeline so it is fit per training fold
     only (ADR-0003 §3). This module is pure, stateless arithmetic.

  5. ROW-BASED WINDOWS, NOT CALENDAR WINDOWS. NEPSE trades ~230-250 sessions
     a year and the basket has uneven session density (ADR-0006: CBBL is
     sparse). All windows here count *trading sessions*, not calendar days.
     A "20-day SMA" is 20 traded bars. This is the standard convention for
     technical indicators and is stated explicitly in the leakage audit.

Author: Ashutosh — NepseIQ DSML Capstone
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

# Columns expected from data_loader.load_raw(). Documented in ADR-0007.
REQUIRED_COLUMNS = [
    "symbol",
    "date",
    "open",
    "high",
    "low",
    "close",
    "traded_quantity",
    "traded_amount",
]

# Windows. Kept as module constants so the leakage audit and the viva answer
# reference the same numbers the code actually uses.
SMA_WINDOWS = (5, 10, 20, 50)
EMA_SPANS = (12, 26)
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_WINDOW, BB_STD = 20, 2
ATR_PERIOD = 14
VOLUME_WINDOWS = (5, 20)
LAGS = (1, 2, 3, 4, 5)
ROLL_WINDOWS = (5, 10, 20)

# The longest lookback any feature uses. Rows before this per symbol are
# warm-up rows and get dropped.
MAX_LOOKBACK = max(max(SMA_WINDOWS), MACD_SLOW + MACD_SIGNAL)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_input(df: pd.DataFrame) -> None:
    """Fail loudly and early if the loader output is not what we expect."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required column(s): {missing}. "
            f"Got columns: {sorted(df.columns)}"
        )

    if df.duplicated(subset=["symbol", "date"]).any():
        n = int(df.duplicated(subset=["symbol", "date"]).sum())
        raise ValueError(
            f"{n} duplicate (symbol, date) rows found. ADR-0007 resolved these "
            f"inside load_raw(); if they are back, the loader regressed."
        )

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise TypeError("`date` must be datetime64, not " + str(df["date"].dtype))

    if (df["close"] <= 0).any():
        n = int((df["close"] <= 0).sum())
        raise ValueError(f"{n} rows have close <= 0; returns would be undefined.")


# ---------------------------------------------------------------------------
# Indicator primitives (single-series, trailing-window only)
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Wilder's RSI. Uses ewm(alpha=1/period, adjust=False), which is the
    recursive smoothing Wilder defined: each value depends only on the
    previous smoothed value and the current delta. Strictly backward-looking.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 means an unbroken run of up-days -> RSI is 100 by definition.
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series,
         period: int = ATR_PERIOD) -> pd.Series:
    """
    Average True Range, Wilder-smoothed. True Range at t uses the t-1 close,
    which is past data — no leak.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume: running sum of signed volume. cumsum() is a prefix
    sum — value at t depends only on rows <= t.
    """
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


# ---------------------------------------------------------------------------
# Per-symbol feature block
# ---------------------------------------------------------------------------

def _features_for_symbol(g: pd.DataFrame) -> pd.DataFrame:
    """
    Build every feature for ONE symbol. `g` must already be sorted by date.

    Feature families and their stationarity flag (whether the value is
    comparable across symbols with different price levels):

      price/return   -> stationary
      raw MA levels  -> NON-stationary (kept because PLAN.md Day 3 names them,
                        but see the note at the bottom of this file)
      MA ratios      -> stationary
      oscillators    -> stationary (bounded or normalised)
      volume ratios  -> stationary
      raw volume MAs -> NON-stationary
    """
    out = pd.DataFrame(index=g.index)

    close = g["close"]
    high = g["high"]
    low = g["low"]
    open_ = g["open"]
    volume = g["traded_quantity"]
    amount = g["traded_amount"]

    prev_close = close.shift(1)

    # --- Price & return features (all from close, never per_change) --------
    out["ret_1d"] = close.pct_change()
    out["log_ret_1d"] = np.log(close / prev_close)
    out["hl_range_pct"] = (high - low) / close
    out["close_open_gap_pct"] = (close - open_) / open_
    out["overnight_gap_pct"] = (open_ - prev_close) / prev_close
    out["close_position_in_range"] = (close - low) / (high - low).replace(0.0, np.nan)

    # --- Simple moving averages + their stationary ratio forms -------------
    for w in SMA_WINDOWS:
        sma = close.rolling(window=w, min_periods=w).mean()
        out[f"sma_{w}"] = sma
        out[f"close_sma_{w}_ratio"] = close / sma
    out["sma_5_20_ratio"] = out["sma_5"] / out["sma_20"]
    out["sma_20_50_ratio"] = out["sma_20"] / out["sma_50"]

    # --- Exponential moving averages ---------------------------------------
    for s in EMA_SPANS:
        ema = close.ewm(span=s, adjust=False, min_periods=s).mean()
        out[f"ema_{s}"] = ema
        out[f"close_ema_{s}_ratio"] = close / ema

    # --- RSI ----------------------------------------------------------------
    out[f"rsi_{RSI_PERIOD}"] = _rsi(close, RSI_PERIOD)

    # --- MACD ---------------------------------------------------------------
    ema_fast = close.ewm(span=MACD_FAST, adjust=False, min_periods=MACD_FAST).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False, min_periods=MACD_SLOW).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=MACD_SIGNAL, adjust=False, min_periods=MACD_SIGNAL).mean()
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd - macd_signal
    # Normalised by price so it is comparable across symbols.
    out["macd_hist_pct"] = (macd - macd_signal) / close

    # --- Bollinger Bands ----------------------------------------------------
    bb_mid = close.rolling(window=BB_WINDOW, min_periods=BB_WINDOW).mean()
    bb_sd = close.rolling(window=BB_WINDOW, min_periods=BB_WINDOW).std(ddof=0)
    bb_upper = bb_mid + BB_STD * bb_sd
    bb_lower = bb_mid - BB_STD * bb_sd
    out["bb_width"] = (bb_upper - bb_lower) / bb_mid
    out["bb_pct_b"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0.0, np.nan)

    # --- ATR ----------------------------------------------------------------
    atr = _atr(high, low, close, ATR_PERIOD)
    out[f"atr_{ATR_PERIOD}"] = atr
    out[f"atr_{ATR_PERIOD}_pct"] = atr / close

    # --- Volume features ----------------------------------------------------
    for w in VOLUME_WINDOWS:
        vma = volume.rolling(window=w, min_periods=w).mean()
        out[f"volume_ma_{w}"] = vma
        out[f"volume_ratio_{w}"] = volume / vma.replace(0.0, np.nan)
    out["obv"] = _obv(close, volume)
    out["obv_change_5"] = out["obv"].diff(5)
    # VWAP proxy: average traded price for the session vs the close.
    vwap = amount / volume.replace(0.0, np.nan)
    out["vwap_close_ratio"] = vwap / close

    # --- Lag features (t-1 .. t-5) ------------------------------------------
    for lag in LAGS:
        out[f"ret_1d_lag_{lag}"] = out["ret_1d"].shift(lag)
    for lag in (1, 2, 3):
        out[f"volume_ratio_5_lag_{lag}"] = out["volume_ratio_5"].shift(lag)

    # --- Rolling statistics of returns --------------------------------------
    for w in ROLL_WINDOWS:
        out[f"ret_mean_{w}"] = out["ret_1d"].rolling(window=w, min_periods=w).mean()
        out[f"ret_std_{w}"] = out["ret_1d"].rolling(window=w, min_periods=w).std(ddof=0)
        out[f"cum_ret_{w}"] = (close / close.shift(w)) - 1.0

    # Volatility-of-volatility: is the recent regime calmer or wilder?
    out["vol_ratio_5_20"] = out["ret_std_5"] / out["ret_std_20"].replace(0.0, np.nan)

    return out


# ---------------------------------------------------------------------------
# Targets — THE ONLY FORWARD-LOOKING CODE IN THIS FILE
# ---------------------------------------------------------------------------

def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the two binary targets from ADR-0003.

        target_1d = 1 if close[t+1] > close[t] else 0
        target_5d = 1 if close[t+5] > close[t] else 0

    Ties (close[t+h] == close[t]) are labelled 0 — "not up". NEPSE has
    genuine flat sessions on thin symbols, so this matters. The convention is
    stated here rather than left implicit.

    These use .shift(-h) and are therefore forward-looking BY DEFINITION —
    that is what a supervised label is. The critical property is that no
    target is ever fed back in as a feature. Enforced by `get_feature_names()`,
    which excludes anything starting with "target_".
    """
    out = df[["symbol", "date"]].copy()
    grp = df.groupby("symbol", sort=False)["close"]
    for horizon in (1, 5):
        future_close = grp.shift(-horizon)
        out[f"target_{horizon}d"] = (future_close > df["close"]).astype("int8")
        # Rows where the future does not exist yet must be NaN, not 0.
        out.loc[future_close.isna(), f"target_{horizon}d"] = pd.NA
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    dropna: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Main entry point. Takes the tidy long-format frame from
    `data_loader.load_raw()` and returns the feature matrix.

    Deliberately takes a DataFrame rather than importing data_loader, so the
    two modules stay decoupled and features.py is testable in isolation.

    Parameters
    ----------
    df : tidy long-format OHLCV, one row per (symbol, date).
    dropna : drop warm-up rows and unlabelled tail rows. Default True.
    verbose : print row-count accounting.

    Returns
    -------
    DataFrame with symbol, date, the OHLCV passthrough columns, all engineered
    features, and target_1d / target_5d.
    """
    validate_input(df)

    work = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    n_start = len(work)

    # Explicit per-symbol loop rather than groupby().apply(). Slower, but the
    # isolation is visible at a glance and the behaviour does not shift between
    # pandas versions — which matters for a result that has to be reproducible
    # on defense day.
    blocks = []
    for symbol, g in work.groupby("symbol", sort=False):
        blocks.append(_features_for_symbol(g))
    feats = pd.concat(blocks).reindex(work.index)

    targets = build_targets(work)

    keep = ["symbol", "date", "open", "high", "low", "close",
            "traded_quantity", "traded_amount"]
    matrix = pd.concat(
        [work[keep], feats, targets[["target_1d", "target_5d"]]], axis=1
    )

    # Division-by-near-zero on thin trading days can produce infinities.
    feature_cols = get_feature_names(matrix)
    matrix[feature_cols] = matrix[feature_cols].replace([np.inf, -np.inf], np.nan)

    n_after_build = len(matrix)

    if dropna:
        matrix = matrix.dropna(subset=feature_cols + ["target_1d", "target_5d"])
        matrix = matrix.reset_index(drop=True)

    if verbose:
        print(f"[features] input rows            : {n_start:,}")
        print(f"[features] after feature build   : {n_after_build:,}")
        print(f"[features] after warm-up dropna  : {len(matrix):,}")
        print(f"[features] rows dropped          : {n_after_build - len(matrix):,}")
        print(f"[features] engineered features   : {len(feature_cols)}")
        print(f"[features] max lookback (bars)   : {MAX_LOOKBACK}")

    return matrix


def get_feature_names(matrix: pd.DataFrame) -> list[str]:
    """
    The model-facing feature columns: everything that is not an identifier,
    not raw passthrough OHLCV, and not a target.

    This is the single source of truth for what the model is allowed to see.
    Day 4 must call this rather than hand-listing columns.
    """
    excluded = {
        "symbol", "date",
        "open", "high", "low", "close",
        "traded_quantity", "traded_amount",
        "per_change",
        "target_1d", "target_5d",
    }
    return [c for c in matrix.columns if c not in excluded]


# Feature families that carry an absolute price/volume level and are therefore
# NOT comparable across symbols trading at different prices. Flagged here, not
# dropped — whether to exclude them is a Day 4 modelling decision, not a Day 3
# feature-engineering one.
NON_STATIONARY_FEATURES = [
    *[f"sma_{w}" for w in SMA_WINDOWS],
    *[f"ema_{s}" for s in EMA_SPANS],
    "macd", "macd_signal", "macd_hist",
    f"atr_{ATR_PERIOD}",
    *[f"volume_ma_{w}" for w in VOLUME_WINDOWS],
    "obv", "obv_change_5",
]
